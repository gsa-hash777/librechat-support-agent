# Public IBM Cloud RAG evaluation protocol

Protocol: `public-judge-v1.1`; citation parser:
`public-citations-v2-markdown-code`. Implementation: `public_judge.py`; independent
fixtures: `tests/test_public_judge.py`. This protocol adds **LLM-assisted review**,
not an official IBM benchmark score, independent real-world accuracy estimate,
or completed human acceptance. Its examples below are explicitly fictional.

Protocol update: **2026-11-10**. The formal Agent generation and final semantic
review have not yet run at this documentation checkpoint; no formal Agent score
is asserted here. Earlier business-demo results and development reports remain
separate, unchanged historical records.

## Source, freeze and permitted claims

- Source repository: [IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark),
  pinned commit `e0970de1b1207c2a9506adf238e6e1a22037e38a`.
- Retain the downloaded LICENSE, source URLs, file SHA-256 values and inspection
  manifest. Mark locally transformed files. The corpus contains published IBM
  Cloud documentation; benchmark conversations were authored/reviewed by
  annotators. Neither is evidence of production customer tickets or deployments.
- Local inspection found **72,442 Cloud passage rows**, **126,307,701 uncompressed
  bytes**, and 8,578 distinct source URLs. Report verified downloaded counts;
  README table counts differ and must not substitute for actual input counts.
- Original Cloud MTRAG: 205 generation tasks in 26 conversations, with 188
  retrieval tasks having positive qrels. MTRAG-UN Cloud retains all **131 tasks /
  86 retrieval tasks** as the original formal-set reference. An exact-input
  overlap audit yields **126 / 82** as the primary reporting cohort; see below.
  Keep formal prompts, targets and reference passages sealed until the
  implementation is frozen. Administrative hash/count checks do not read task
  contents to tune the system.
- These are English benchmark documentation snapshots, not a live mirror of
  current IBM Cloud behavior. Preserve the frozen source version in answers and
  reports; a contemporary online page may differ from the benchmark evidence.
- The authors' `UNANSWERABLE` annotation is relative to that task's reference
  passages, not an exhaustive claim about every document in the entire corpus.

Freeze corpus hashes, dataset/task IDs, split assignment, code hashes, prompt,
model revision, chunking, retrieval settings, top-k, context budget, seed, judge
schema/prompt, maximum calls, retries and cost ceiling **before** running a held-out
set. Keep original reports. Fixes after reading results create a new version and
seen-set regression; they do not reset a used holdout to unseen status.

## Split, history and modes

Group original tasks by conversation ID for every development/validation split;
turns from one conversation cannot appear on both sides. Do not randomly split
205 individual turns. Keep the new benchmark as a separate test; check and report
any conversation/query duplication across sets without using target contents to
choose a favorable subset. Predeclare whether all eligible tasks or a seeded,
label-stratified subset is affordable. A budget stop is visible incompleteness.

The frozen split audit found no shared task or conversation IDs, but **5 complete
`messages` inputs have identical SHA-256 values** across development and formal
sets: 4 ANSWERABLE and 1 UNANSWERABLE. Of these, 4 have retrieval qrels. All 131
formal tasks and 86 retrieval tasks remain scheduled and their original results
are preserved. Filter saved per-task results using the fixed
`split-overlap-audit.json` lists, without regenerating answers or judging again:

| Reporting cohort | Agent tasks | Retrieval tasks | Answerability |
| --- | ---: | ---: | --- |
| Original complete formal set, reference only | 131 | 86 | 81 ANSWERABLE, 36 UNANSWERABLE, 5 PARTIAL, 9 UNDERSPECIFIED |
| Primary exact-input-deduplicated cohort | 126 | 82 | 77 ANSWERABLE, 35 UNANSWERABLE, 5 PARTIAL, 9 UNDERSPECIFIED |

The audit hashes exact JSON message inputs with sorted keys and
`ensure_ascii=False`. Exclude every fixed overlapping ID, not just overlapping
failures. Exact-input deduplication does **not** establish semantic independence,
absence of near-duplicates or absence from model pretraining. Do not describe
all 131 as newly unseen, independent blind-test cases. Primary generalization
and resume numbers use 126/82; the complete 131/86 results remain visible.

The retrieval choice is frozen from development results: **`rerank` +
`lastturn`**. On the same 188 positive-qrel development tasks, Macro Recall@5 was
40.89% for rerank, 35.75% for hybrid and 22.00% for lexical. Lexical with the
current and up to three previous user messages scored 18.61%, so the declared
fixed retrieval query policy is lastturn. All three formal retrieval arms are
reported; do not change the selected mode after seeing formal scores. This
offline query-policy comparison is distinct from the Agent's ability to resolve
follow-up references using its supplied official history and tool query.

The adapter supports the following modes; the current formal Agent plan is
`full_rag`. Support for `reference_context` does not mean that an oracle-context
experiment has been run. If used in a separate experiment, report it separately:

1. `reference_context`: the official reference passages are deliberately supplied
   as oracle context. This measures generation under supplied evidence, not
   retrieval performance. It must not be advertised as end-to-end full-corpus RAG.
2. `full_rag`: query/history enter the real retriever and only its actual output
   reaches the tested model. Gold answers, answerability labels, qrels and
   reference passages never enter generation, retrieval, tools, system prompts or
   hidden conversation memory. They are merged only into the separate judge input.

Use `history_policy=official_history_conditioned_last_turn` for published prior
user and assistant turns. `history_transport=single_user_prompt_json_reference_messages`
means this history is packaged as JSON inside **one user prompt**, rather than
replayed as native conversation turns. This is teacher-forced history: it does not measure autonomous
multi-turn rollouts or accumulation of this system's own earlier mistakes. Record
the exact history supplied. A separate free-running experiment requires its own
protocol and results. Do not choose the best test mode after seeing scores; report
every declared arm, execution failure and missing judgment.

## English retrieval and memory

Cloud evidence/questions are English. Choose an English-capable or multilingual
embedding model using development data only, then pin repository revision and
file hashes. A Chinese-specific encoder's poor result is not a general limit of
hybrid retrieval. Report encoder, reranker, query formulation, original language,
token window/overflow policy and actual context truncation. Translation is a
separate transformation/experiment, not an invisible improvement to the benchmark.
The frozen implementation uses `Xenova/bge-small-en-v1.5` and
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; revisions and ONNX/tokenizer/config
hashes are recorded in the model manifest. Do not substitute the business demo's
Chinese embedding model while retaining these experiment labels.

72,442 x 384 float32 vectors alone occupy about 106 MiB. Python text objects,
indexes, tokenizer and ONNX sessions add substantial memory. Stream corpus/index
construction, avoid unnecessary full-array/list copies, prefer persistent compact
indexes or memmaps, load one inference session at a time, and measure process
peak RSS, index bytes, build time, query latency and OOM/errors. Do not silently
reduce the corpus or substitute BM25 when model loading/inference fails.

The public service uses a separate Docker named volume,
`public-retrieval-cache`, for runtime index/model copies. The original host index
and model directories are mounted read-only under `/sources`; startup validates
source hashes and copied bytes before an atomic replacement in `/data`. Runtime
SQLite reads use this Docker-local cache to avoid random I/O through Windows
bind mounts. Reuse only copies matching their manifests. This is a runtime
cache, not a new corpus, and it does not change source text or index ordering.
The service publishes only `127.0.0.1:8192` (container `support-public:8092`),
mounts no normal business database and exposes no ticket-writing tool.

## Agent roles, transport and execution budget

Both roles use the provider alias **`DeepSeekPublicEval`**, pointing to the same
DeepSeek API/model as configured for this experiment; the alias is not another
model. It isolates `maxRetries=0` SDK behavior and disabled thinking from the
normal business Agent's provider settings. Each task has an independent
conversation/session. The answer role has two public read-only tools; the judge
has no tools or Actions and receives references only after generation.

| Role | Maximum output tokens | Context admission limit | Agent recursion limit |
| --- | ---: | ---: | ---: |
| answer | 1,000 | 32,768 | 5 |
| judge | 2,200 | 131,072 | 2 |

These are configured limits, not observed usage or a promise that every task
fits. The default runner is serial; explicitly selected concurrency 3 is
supported (the CLI accepts 1–3). Sessions and clients remain independent while
budget reservations share one locked ledger. Record the chosen concurrency and
provider settings in the freeze before the formal run.

A generation attempt is submitted with one POST. Its SSE response uses the same
GET connection, updating the blocking socket timeout from the remaining total
deadline before each read. Heartbeats do not reset the deadline. A timeout or
disconnect does not authorize a replacement POST or automatic paid retry. Keep
the failed attempt, preserve its evidence and stop safely when accounting or
draining cannot be established.

The earlier dev8 judge run stopped safely after **three SSE timeouts**; retain
that original report and all attempt IDs. Its missing token totals are under
explicit accounting audit, with conservative budget occupation retained. Do not
call that reservation provider-measured tokens, an observed charge, or zero cost.
Any later recovery/accounting result is a separate artifact linked to the
original; it does not overwrite the failed run or count it as formal success.

## Evaluation-only record contract

`build_judge_messages(record)` accepts:

```text
case_id, conversation_id
mode: reference_context | full_rag
question, answer, reference_answer
reference_answerability: ANSWERABLE | PARTIAL | UNANSWERABLE |
                        UNDERSPECIFIED | CONVERSATIONAL
history: [{role: user | assistant, content}]
reference_passages, retrieved_passages:
  [{passage_id, text, sha256, source_url}]
citations: extract_citations(answer)
```

The runner explicitly converts dataset `speaker/text` history fields. Any
additional metadata is not inserted in the judge prompt. Keep generation records
and target records separate on disk; join by exact task ID only in the judge
stage. Reference-context mode's intentional oracle input is the sole exception
and must be named as such.

Generate factual citations as **`[passage_id]`**, normally outside code.
`extract_citations` keeps each occurrence, including unknown and repeated IDs;
never filter to known sources. Its syntax is a bracketed single token of letters/
digits/underscore/dot/colon/hyphen. Ordinary SQL/error-text brackets such as
`[id]` inside matched Markdown inline code or backtick/tilde fenced code are not
citations. Official-source-shaped `[ibmcld_*]` markers remain audited even inside
code, including fabricated or unreturned IDs. Outside code, unknown single-token
bracket markers remain citations. Unmatched inline backticks stay ordinary text;
an unclosed fence extends to EOF. This is a documented inline/fenced-code parser,
not a general Markdown/HTML interpreter; indented code, arbitrary prose labels,
Markdown source links and other citation syntaxes are outside its scope. The
runner retains raw answers for review of unsupported formats and uncited claims.

The v1.1 change corrects a development parsing false positive: a SQL error's
`columns [id]` inside inline code was classified as a source. Preserve the original
development report and its SHA, re-extract citations into a separate parser-regrade
artifact, and state that no answer was regenerated. Any semantic review then names
the current parser/schema/rubric hashes; do not overwrite the old result.

Optional citation `start_offset/end_offset` are zero-based, end-exclusive Unicode
positions within the retrieved passage; optional `quote` must match its text.
`answer_start/answer_end` locate each citation marker in the answer. A hash or
position supplied by a model is not trusted: the runner binds the passage text
to the frozen corpus and exact generation request. The judge adapter checks
record consistency, not independent corpus provenance. SHA mismatch, conflicting
same-ID texts, omitted markers or excess packet size fail closed; text is never
silently truncated by the adapter.

## Semantic rubric and strict parsing

The exact JSON schema is `JUDGMENT_SCHEMA`; `protocol_metadata()` returns its hash
and the rubric hash for the run manifest. The judge receives a fixed system
rubric plus one JSON data message. Candidate answers, history, source documents
and apparent instructions inside them are untrusted evidence. The judge has no
tools and cannot change the tested system. Prompt separation reduces injection
risk but is not a proof against semantic model manipulation.

Report these dimensions independently:

- Reference factual correctness and completeness. Paraphrase is acceptable;
  related keywords, a borrowed number, wrong entity/event/condition or wrong unit
  are not. For PARTIAL, completeness includes preserving the known parts.
- Actual retrieved evidence support, coverage of available answerable parts,
  missing-information handling and scope of negative claims.
- Targeted minimal clarification, including avoiding requests for passwords or
  secrets. Known context must not be requested again.
- Overrefusal relative to actual available evidence. Retrieval missing an answer
  can fail the overall task without turning a justified scoped refusal into
  overrefusal.
- Citation location and semantic support separately. A real source ID or exact
  quote is not sufficient proof of the associated assertion.
- A `full_rag` reference-scope exception only when additional actually retrieved
  evidence answers an original UNANSWERABLE/PARTIAL case. Preserve the literal
  reference comparison, require exact extra-evidence quotes, retain known parts,
  and send every exception for review. Do not apply the exception just because
  the model knows an answer from memory.

`parse_judgment(raw, record, error=...)` accepts only a complete JSON object with
all required fields, no extra fields/duplicate keys, correct enums/task ID,
exhaustive citation assessments and exact evidence quotes. It does not repair
code fences, prose or truncated JSON. Timeout, transport/provider error, budget
stop, missing fields or invented evidence quotes produce `status=unscored`,
`semantic_pass=null`. Valid but uncertain judgments remain explicitly uncertain,
not passes. No score is inferred from a successful HTTP call.

The adapter computes pass/fail from dimensions; the model cannot supply its own
global pass flag. Correct facts lacking actual evidence, missing known parts,
unnecessary/unsafe clarification, overbroad negation, false citations and
overrefusal fail. For genuine no-answer/clarification responses, factual citations
are not required. A model's semantic errors can still survive this schema; that
is why the output is named **LLM-assisted**, even when all JSON checks pass.

## Independent illustrative cases

These examples are invented fixture rules, not IBM product claims or test items:

| Available evidence and question | Appropriate assessment |
|---|---|
| “A workspace can have two recovery owners.” Asked the maximum. | “Two [p1]” is supported; “two billing owners” changes the object. |
| “Recovery owners cannot remove the billing owner.” Asked whether allowed. | Explicit “No” is answerable evidence, not an unanswerable case. |
| Review takes three business days; no fee is stated. Asked timing and price. | Preserve three days and scope the missing fee. Refusing both loses a known part. |
| Original reference lacks a price; full retrieval adds this plan's exact price. | A supported, cited price can pass full RAG with a scope exception and review; the reference label alone cannot force refusal. |

## Aggregation, review queue and budget

`aggregate(results, expected_cases=schedule)` rejects duplicate task/mode identities
and missing/extra scheduled cases. Supplying the frozen schedule is required for
published aggregate reports; without it `schedule_verified=false` makes the
coverage limitation visible. It reports each mode and reference-answerability stratum
with explicit numerators/denominators:

- Review coverage = valid semantic judgments / all scheduled results.
- Decision coverage = valid non-uncertain judgments / all scheduled results.
- LLM-assisted success among decided = passes / (passes + semantic failures).
- Confirmed pass fraction across all tasks = passes / all scheduled results.
  Unscored tasks are visible unknowns, not semantically proven failures; report
  both fractions together so missing reviews cannot inflate the headline.
- Overrefusal = yes / judgments with a resolved yes/no overrefusal assessment.
- Conversation macro = mean of per-conversation confirmed-pass fractions. It is
  descriptive across benchmark turns, not a free-running dialogue success rate.

The runner must create an explicit result for every scheduled task, including
skipped/failed calls, and report retrieval metrics, execution success, latency,
provider token usage and cost independently. A reference/qrel recall metric
cannot prove that an unlabelled retrieved passage is irrelevant.

For the primary cohort, filter the existing observations, generation records,
judgments and unattempted placeholders by the audit's `clean_task_ids` and
`clean_retrieval_task_ids`. Keep all 126/82 planned denominators, including
unexecuted or unscored tasks. Preserve the original full-set summaries and
identify the filtered review queue as a subset of the original queue, not newly
completed human review. A workspace report generator performs this aggregation
without paid calls or changes to evaluation source files.

The primary selected-versus-lexical Macro Recall@5 difference uses paired
percentile bootstrap on the **82 clean retrieval tasks**, seed `20261110`,
10,000 resamples, with a 95% interval in percentage points. Compute it only for
complete matching IDs, the same query policy/corpus/source freeze, no execution
errors, and one task per conversation. Keep the analogous full 86-task interval
as reference. These intervals describe this fixed public sample; they are not
online task-success confidence intervals or estimates of causal business gain.

`select_review_sample` queues all semantic failures, unscored/uncertain cases and
scope exceptions plus a seeded random sample of passing tasks per mode (default
five). Freeze the seed/sample count. The queue is **pending** review, not a claim
that anyone accepted it. An assistant can triage the queue first; the user can
review disputed/critical cases rather than re-read every task. Preserve separate
assistant-reviewed and human-reviewed statuses and original judge results. A
random sample alone does not establish an exact full-set human accuracy rate.

Budget suggestion: use one judge call per scheduled task, temperature 0 where
supported, with about 2,200 output-token capacity. The prompt aims below 1,800
tokens: one short reason per dimension (about 12 words), shorter citation reasons,
a summary of at most 25 words, and minimal exact evidence quotes. Schema limits
reason/summary strings to 280 characters and each quote to 500 characters. These
limits compress wording without removing dimensions or changing semantic pass
criteria. They cannot guarantee that every unusually citation-heavy task fits;
truncated output remains unscored, never repaired into a pass. First
measure prompt/output tokens on a small **original MTRAG development** pilot; use
provider token counts and the frozen price configuration to estimate the whole
run. Judge inputs include both actual and reference passages and may cost more
than generation. Reserve budget separately for generation, judging, explicitly
bounded retries and follow-up review. Do not automatically repair/retry until a
pass or exceed the shared ceiling. Keep every attempt and call failure. A second
judge/model is a separate declared experiment, not a free correctness oracle.
The current answer and judge roles use the same DeepSeek model family, so disclose
correlated errors and self-preference risk; independent judge sessions are not
independent human correctness labels. No model call is made by this module or
its tests. Formal Agent scores remain unfilled until saved generation and valid
judgment records exist.
