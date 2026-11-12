"""Real tokenizer windows, fake inference tensors; no downloads or model weights."""
import json
from pathlib import Path
from types import SimpleNamespace
import weakref

import numpy as np
import onnxruntime
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from app.hybrid_models import LocalModels, QUERY_PREFIX


@pytest.fixture
def files(tmp_path):
    vocab = {"[UNK]": 0, "[CLS]": 1, "[SEP]": 2}
    vocab.update({f"t{index}": index + 3 for index in range(200)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
                                                              special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    for name in ("embedding", "reranker"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "model.onnx").write_bytes(b"fake inference file; never opened by ONNX")
        tokenizer.save(str(folder / "tokenizer.json"))
        (folder / "config.json").write_text(json.dumps({"hidden_size": 3, "num_labels": 1}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text('{"fixture":true}', encoding="utf-8")
    return tmp_path


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(live=weakref.WeakSet(), loads=[], calls=[], output_mode="normal", load_error=False)

    class Session:
        def __init__(self, path, sess_options, providers):
            assert len(state.live) == 0, "The prior model must be released before loading another one."
            if state.load_error:
                raise RuntimeError("simulated broken model")
            self.name = Path(path).parent.name
            assert providers == ["CPUExecutionProvider"]
            assert sess_options.enable_cpu_mem_arena is False
            assert sess_options.enable_mem_pattern is False
            assert sess_options.intra_op_num_threads == 2
            assert sess_options.inter_op_num_threads == 1
            state.live.add(self)
            state.loads.append(self.name)

        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name=name, type="tensor(int64)")
                    for name in ("input_ids", "attention_mask", "token_type_ids")]

        def run(self, selected, feed):
            assert all(value.dtype == np.int64 and value.ndim == 2 and value.shape[0] == 1 for value in feed.values())
            state.calls.append((self.name, {key:value.copy() for key,value in feed.items()}))
            if self.name == "embedding":
                values = np.zeros((1, feed["input_ids"].shape[1], 3), dtype=np.float32)
                values[0, 0] = [feed["input_ids"].sum(), feed["input_ids"].shape[1], 1]
                if state.output_mode == "zero":
                    values.fill(0)
                elif state.output_mode == "nan":
                    values[0, 0, 0] = np.nan
                elif state.output_mode == "wrong_shape":
                    values = np.ones((1, 3), dtype=np.float32)
                return [values]
            value = float(feed["input_ids"].max())
            if state.output_mode == "nan":
                value = float("nan")
            return [np.asarray([[value]], dtype=np.float32)]

    monkeypatch.setattr(onnxruntime, "InferenceSession", Session)
    return state


def test_initialization_is_lazy_and_missing_models_fail_explicitly(files, runtime):
    model = LocalModels(files)
    assert model.metadata()["active_sessions"] == 0
    assert runtime.loads == []
    (files / "reranker/model.onnx").unlink()
    with pytest.raises(RuntimeError, match="incomplete.*No fallback"):
        LocalModels(files)


def test_single_session_is_reused_then_released_before_model_switch(files, runtime):
    model = LocalModels(files)
    assert model.embed(["t1 t2"]).shape == (1, 3)
    model.embed(["t3"], query=True)
    assert runtime.loads == ["embedding"]
    assert model.rerank("t2", ["t3", "t4"]) == [6.0, 7.0]
    model.embed(["t1"])
    assert runtime.loads == ["embedding", "reranker", "embedding"]
    assert model.metadata()["peak_active_sessions"] == 1
    model.close()
    assert model.metadata()["active_sessions"] == 0 and not runtime.live
    model.close()


def test_embedding_overflows_cover_document_tail_and_mean_then_normalize(files, runtime):
    model = LocalModels(files, max_length=16)
    text = " ".join(f"t{index}" for index in range(55))
    result = model.embed([text])
    calls = [feed for name,feed in runtime.calls if name == "embedding"]
    assert len(calls) > 1
    assert max(feed["input_ids"].shape[1] for feed in calls) <= 16
    observed = set(np.concatenate([feed["input_ids"].ravel() for feed in calls]))
    assert set(range(3, 58)) <= observed
    vectors = np.asarray([[feed["input_ids"].sum(), feed["input_ids"].shape[1], 1] for feed in calls], dtype=np.float32)
    expected = vectors.mean(axis=0)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)
    assert model.metadata()["window_counts"]["embedding_documents"] == len(calls)


def test_query_embedding_prefix_and_overflow_tail_are_retained(files, runtime):
    model = LocalModels(files, max_length=16)
    text = " ".join(f"t{index}" for index in range(80))
    model.embed([text], query=True)
    # The fixture maps the prepended Chinese instruction to UNK; tail tokens
    # still occur in later windows. The actual model tokenizer handles Chinese.
    ids = np.concatenate([feed["input_ids"].ravel() for _,feed in runtime.calls])
    assert 0 in ids and 82 in ids
    metadata = model.metadata()
    assert metadata["query_prefix"] == QUERY_PREFIX
    assert metadata["window_counts"]["embedding_queries"] > 1


def test_reranker_uses_passage_overflow_max_instead_of_first_window(files, runtime):
    model = LocalModels(files, max_length=16)
    passage = " ".join(f"t{index}" for index in range(50))
    assert model.rerank("t1", [passage]) == [52.0]
    calls = [feed for name,feed in runtime.calls if name == "reranker"]
    assert len(calls) > 1
    assert max(feed["input_ids"].shape[1] for feed in calls) <= 16
    observed = set(np.concatenate([feed["input_ids"].ravel() for feed in calls]))
    assert set(range(3, 53)) <= observed
    assert model.metadata()["window_counts"]["rerank_pairs"] == len(calls)


def test_long_query_and_long_passage_both_keep_their_tail(files, runtime):
    model = LocalModels(files, max_length=16)
    query = " ".join(f"t{index}" for index in range(25, 65))
    passage = " ".join(f"t{index}" for index in range(80, 130))
    assert model.rerank(query, [passage]) == [132.0]
    first, second = set(), set()
    for _, feed in runtime.calls:
        ids, types = feed["input_ids"][0], feed["token_type_ids"][0]
        first.update(ids[types == 0])
        second.update(ids[types == 1])
    assert set(range(28, 68)) <= first
    assert set(range(83, 133)) <= second
    assert model.metadata()["window_counts"]["rerank_query_segments"] > 1


@pytest.mark.parametrize("bad_mode", ["zero", "nan", "wrong_shape"])
def test_invalid_embedding_output_never_becomes_a_usable_vector(files, runtime, bad_mode):
    runtime.output_mode = bad_mode
    with pytest.raises(RuntimeError):
        LocalModels(files).embed(["t1"])


def test_invalid_reranker_output_fails_without_lexical_fallback(files, runtime):
    runtime.output_mode = "nan"
    with pytest.raises(RuntimeError, match="finite logit"):
        LocalModels(files).rerank("t1", ["t2"])


def test_failed_switch_does_not_keep_stale_embedding_session(files, runtime):
    model = LocalModels(files)
    model.embed(["t1"])
    runtime.load_error = True
    with pytest.raises(RuntimeError, match="Unable to load.*No fallback"):
        model.rerank("t1", ["t2"])
    assert model.metadata()["active_sessions"] == 0
    assert not runtime.live


def test_empty_batch_avoids_loading_and_metadata_never_contains_input(files, runtime):
    model = LocalModels(files)
    assert model.embed([]).shape == (0, 3)
    assert model.rerank("t1", []) == []
    assert runtime.loads == []
    model.embed(["private-example-text"])
    assert "private-example-text" not in json.dumps(model.metadata())


@pytest.mark.parametrize("arguments", [{"threads":0}, {"threads":True}, {"max_length":1024}, {"max_length":1}])
def test_invalid_runtime_limits_rejected(files, arguments):
    with pytest.raises(ValueError):
        LocalModels(files, **arguments)
