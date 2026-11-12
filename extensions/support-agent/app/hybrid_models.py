"""CPU-only, local ONNX inference with one resident model and complete windows.

This module never downloads a model or changes retrieval mode on failure. The
index/retrieval layer owns any persistent embedding cache and its provenance.
"""
import gc
import hashlib
import importlib
import json
from pathlib import Path
from threading import RLock
import time


QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
REQUIRED_FILES = ("model.onnx", "tokenizer.json", "config.json")


class LocalModels:
    """Reuse the active model; release it before loading the other model.

    ``close`` unloads resident state and may be followed by another inference.
    Metadata contains counts/timing, never source documents or user queries.
    """

    def __init__(self, model_dir, threads=2, max_length=512, query_prefix=QUERY_PREFIX):
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be a positive integer")
        if type(max_length) is not int or not 8 <= max_length <= 512:
            raise ValueError("max_length must be an integer between 8 and 512")
        self.model_dir = Path(model_dir)
        if not isinstance(query_prefix, str):
            raise ValueError("query_prefix must be text")
        self.query_prefix = query_prefix
        self.threads, self.max_length = threads, max_length
        self.stride = min(32, max_length // 4)
        self._lock = RLock()
        self._session = self._tokenizer = self._active = None
        self._numpy = None
        self._configs = {}
        self._loads = {name: 0 for name in ("embedding", "reranker")}
        self._load_seconds = {name: 0.0 for name in self._loads}
        self._inference_seconds = {name: 0.0 for name in self._loads}
        self._windows = {name: 0 for name in ("embedding_documents", "embedding_queries", "rerank_pairs", "rerank_query_segments")}
        self._pair_strides = set()
        self._peak_sessions = 0
        self._validate_files()

    def _validate_files(self):
        for name in ("embedding", "reranker"):
            for filename in REQUIRED_FILES:
                path = self.model_dir / name / filename
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Local {name} model is incomplete: missing {filename}. No fallback was performed.")
            try:
                config = json.loads((self.model_dir / name / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                raise RuntimeError(f"Local {name} config.json is invalid.") from None
            if not isinstance(config, dict):
                raise RuntimeError(f"Local {name} config.json must be an object.")
            self._configs[name] = config

    def _release(self):
        self._session = self._tokenizer = self._active = None
        gc.collect()

    def close(self):
        with self._lock:
            self._release()

    def _load(self, name):
        if self._active == name:
            return
        self._release()
        self._validate_files()
        started = time.perf_counter()
        try:
            self._numpy = importlib.import_module("numpy")
            ort = importlib.import_module("onnxruntime")
            tokenizer_module = importlib.import_module("tokenizers")
        except ImportError:
            raise RuntimeError("Local inference requires numpy, tokenizers and onnxruntime. No fallback was performed.") from None
        try:
            tokenizer = tokenizer_module.Tokenizer.from_file(str(self.model_dir / name / "tokenizer.json"))
            tokenizer.no_padding()
            tokenizer.no_truncation()
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.enable_cpu_mem_arena = False
            options.enable_mem_pattern = False
            session = ort.InferenceSession(str(self.model_dir / name / "model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
            supported = {"input_ids", "attention_mask", "token_type_ids"}
            if not {item.name for item in session.get_inputs()} <= supported:
                raise ValueError("unsupported model inputs")
            if "CPUExecutionProvider" not in session.get_providers():
                raise ValueError("CPU provider unavailable")
        except Exception:
            # The other session was already released; failed loading cannot keep
            # a stale model resident or silently produce lexical-only results.
            raise RuntimeError(f"Unable to load the local {name} ONNX model. No fallback was performed.") from None
        self._session, self._tokenizer, self._active = session, tokenizer, name
        self._loads[name] += 1
        self._load_seconds[name] += time.perf_counter() - started
        self._peak_sessions = max(self._peak_sessions, 1)

    def _run(self, encoding):
        np = self._numpy
        if len(encoding.ids) > self.max_length:
            raise RuntimeError("Tokenizer window exceeds the configured model context.")
        values = {"input_ids": encoding.ids, "attention_mask": encoding.attention_mask,
                  "token_type_ids": encoding.type_ids}
        feed = {}
        for item in self._session.get_inputs():
            dtype = np.int32 if item.type == "tensor(int32)" else np.int64
            feed[item.name] = np.asarray([values[item.name]], dtype=dtype)
        started = time.perf_counter()
        outputs = self._session.run(None, feed)
        self._inference_seconds[self._active] += time.perf_counter() - started
        if not outputs:
            raise RuntimeError("The local model returned no output.")
        return outputs[0]

    @staticmethod
    def _texts(values):
        if not isinstance(values, (list, tuple)) or any(not isinstance(value, str) for value in values):
            raise ValueError("texts/passages must be a list of strings")
        return values

    @staticmethod
    def _all_windows(encoding):
        # Hugging Face tokenizers attaches all overflow windows to the primary
        # encoding; each includes the model's required special tokens.
        return [encoding, *encoding.overflowing]

    def embed(self, texts, query=False):
        texts = self._texts(texts)
        with self._lock:
            if not texts:
                np = importlib.import_module("numpy")
                return np.empty((0, int(self._configs["embedding"].get("hidden_size", 512))), dtype=np.float32)
            self._load("embedding")
            np = self._numpy
            tokenizer = self._tokenizer
            tokenizer.enable_truncation(max_length=self.max_length, stride=self.stride, strategy="longest_first")
            result = []
            for text in texts:
                windows = self._all_windows(tokenizer.encode((self.query_prefix if query else "") + text))
                vectors = []
                for window in windows:
                    values = np.asarray(self._run(window))
                    if values.ndim != 3 or values.shape[0] != 1 or not values.shape[1]:
                        raise RuntimeError("Embedding model must return [1, tokens, hidden_size] token representations.")
                    vector = values[0, 0].astype(np.float32, copy=True)
                    dimension = self._configs["embedding"].get("hidden_size", 512)
                    if vector.shape != (dimension,) or not np.isfinite(vector).all():
                        raise RuntimeError("Embedding output has an invalid dimension or non-finite values.")
                    vectors.append(vector)
                pooled = np.mean(np.stack(vectors), axis=0)
                norm = np.linalg.norm(pooled)
                if not np.isfinite(norm) or norm <= 0:
                    raise RuntimeError("Embedding output cannot be normalized.")
                result.append(pooled / norm)
                self._windows["embedding_queries" if query else "embedding_documents"] += len(windows)
            return np.stack(result).astype(np.float32, copy=False)

    def _query_segments(self, query):
        tokenizer = self._tokenizer
        # Reserve at least half the pair window for passage tokens. Slice the
        # original query by tokenizer offsets, not decoded/generated text.
        specials = tokenizer.num_special_tokens_to_add(True)
        budget = (self.max_length - specials) // 2
        if budget < 1:
            raise RuntimeError("Tokenizer special tokens leave no room for a query/passage pair.")
        tokenizer.enable_truncation(max_length=budget, stride=min(self.stride, budget - 1), strategy="longest_first")
        windows = self._all_windows(tokenizer.encode(query, add_special_tokens=False))
        if len(windows) == 1:
            return [query]
        segments = []
        for index, window in enumerate(windows):
            offsets = [(start, end) for start, end in window.offsets if end > start]
            if not offsets:
                raise RuntimeError("Cannot locate an overflow query window in the original text.")
            start = 0 if index == 0 else offsets[0][0]
            end = len(query) if index == len(windows) - 1 else offsets[-1][1]
            segments.append(query[start:end])
        return segments

    def rerank(self, query, passages):
        passages = self._texts(passages)
        if not isinstance(query, str) or not query.strip():
            raise ValueError("rerank query must be non-empty text")
        with self._lock:
            if not passages:
                return []
            self._load("reranker")
            np, tokenizer = self._numpy, self._tokenizer
            segments = self._query_segments(query)
            self._windows["rerank_query_segments"] += len(segments)
            scores = []
            for passage in passages:
                logits = []
                for segment in segments:
                    # Offset slicing can change wordpiece boundaries slightly.
                    # Recheck the actual pair budget rather than letting the
                    # tokenizer panic when stride is >= remaining passage room.
                    tokenizer.no_truncation()
                    query_tokens = len(tokenizer.encode(segment, add_special_tokens=False).ids)
                    room = self.max_length - tokenizer.num_special_tokens_to_add(True) - query_tokens
                    if room < 1:
                        raise RuntimeError("A query window leaves no passage space; no query text was discarded.")
                    stride = min(self.stride, room - 1)
                    self._pair_strides.add(stride)
                    tokenizer.enable_truncation(max_length=self.max_length, stride=stride, strategy="only_second")
                    for window in self._all_windows(tokenizer.encode(segment, passage)):
                        values = np.asarray(self._run(window))
                        if values.size != 1 or not np.isfinite(values).all():
                            raise RuntimeError("Reranker must return one finite logit per query/passage pair.")
                        logits.append(float(values.reshape(-1)[0]))
                        self._windows["rerank_pairs"] += 1
                scores.append(max(logits))
            return scores

    def metadata(self):
        with self._lock:
            manifest = self.model_dir / "manifest.json"
            return {"backend": "local-onnx-cpu", "model_dir": str(self.model_dir),
                    "model_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
                    "active_model": self._active, "active_sessions": int(self._session is not None),
                    "peak_active_sessions": self._peak_sessions, "session_loads": dict(self._loads),
                    "session_load_seconds": dict(self._load_seconds), "inference_seconds": dict(self._inference_seconds),
                    "window_counts": dict(self._windows), "threads": self.threads, "max_length": self.max_length,
                    "stride": self.stride, "batch_size": 1, "cpu_arena": False, "memory_pattern": False,
                    "observed_pair_strides": sorted(self._pair_strides),
                    "embedding_pooling": "CLS per full overflow window; arithmetic mean then L2 normalization",
                    "query_prefix": self.query_prefix,
                    "rerank_window_policy": "Original query offset windows reserve half the context; only_second passage windows; max single logit over all pairs",
                    "persistent_embedding_cache": "owned by retrieval/index layer"}
