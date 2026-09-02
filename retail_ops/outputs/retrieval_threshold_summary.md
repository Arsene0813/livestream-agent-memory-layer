# Retrieval Threshold Inspection Summary

This file summarizes prototype retrieval score distributions over file-backed retail evidence.

The retrieval corpus combines generated Demo 1 and Demo 2 retail memory facts with selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.

The retrieval evidence consists of selected observations manually transcribed from the Meituan merchant backend, together with generated local memory facts; the experiment runs on the repository snapshot rather than a live backend connection.

The purpose is to inspect retrieval-threshold behavior. It is not a production-level threshold validation, a broad LLM benchmark, or proof that retrieved evidence is sufficient for an operating conclusion.

## Corpus

- Retrieval units loaded: 307
- Unit definition: one generated memory fact or one chunked field-contract/source-note segment; this is not a store count or a count of independent business observations.
- Corpus SHA-256: `3e42fa1cc2457385a4cebba21bd2bbc1b92f9397cd4b6749161da70ad715a278`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Run commit: `aa47fa938b7fe48ae090950f35de57c4527988c4`
- Experiment scope SHA-256: `be776b9bd5dde8c9b5cd42cb6738211f1bdbe82a3350c847002a0d6e0997334f`
- Applicability note: the scope hash identifies the clean experiment-relevant code and input snapshot. The run commit is retained for navigation. Later unrelated commits do not invalidate the result. Run `python3 eval/check_retrieval_result_applicability.py` to check the current scope.
- Retrieval threshold cases: 29
- Embedding model: `bge-m3` via local Ollama
- Generated memory fact sources:
  - `retail_ops/outputs/generated_retail_memory_facts.json`
  - `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- Field-contract sources:
  - `retail_ops/data/DATA_DICTIONARY.md`
  - `retail_ops/data/demo2_source_notes.md`

## Case Groups

| Case type | Purpose |
|---|---|
| positive_supported | Queries with expected supporting evidence in the current generated retail facts or field-contract notes. |
| negative_unsupported | Queries that should not have enough evidence in the current corpus. |
| hard_negative_boundary | Queries that may retrieve related facts but still require refusal or qualification. |
| entity_period_mismatch | Queries that mention an entity, period, or demo scope not supported by the retrieved fact. |
| ambiguous_comparison | Broad comparison queries where multiple records may be relevant. |

## Score Distribution by Case Type

| Case type | Cases | Top-1 min | Top-1 p25 | Top-1 median | Top-1 p75 | Top-1 max | Median margin | Expected hit@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ambiguous_comparison | 4 | 0.5450 | 0.5471 | 0.5499 | 0.5588 | 0.5790 | 0.0051 | 3/4 |
| entity_period_mismatch | 4 | 0.5925 | 0.6089 | 0.6199 | 0.6304 | 0.6453 | 0.0143 | 1/4 |
| hard_negative_boundary | 7 | 0.5119 | 0.5658 | 0.6328 | 0.6726 | 0.7161 | 0.0079 | 4/7 |
| negative_unsupported | 6 | 0.4796 | 0.4842 | 0.4946 | 0.5285 | 0.5666 | 0.0052 | 0/6 |
| positive_supported | 8 | 0.5858 | 0.6319 | 0.6494 | 0.7067 | 0.7570 | 0.0289 | 8/8 |

## Threshold Interpretation

A reference threshold can be inspected for the trade-off between unsupported retrieval noise and supported evidence retention.

An exploratory reference threshold from this small corpus is around `0.5802`, midway between the positive-supported p25 score `0.6319` and the negative-unsupported p75 score `0.5285`. This is an inspection reference, not a production cutoff or an answer-decision rule.

High scores in hard-negative boundary cases are expected. They show that a semantically related fact can be retrieved even when the correct answer should still refuse or qualify the requested conclusion.

For that reason, retrieval thresholding must be paired with entity, period, slot, source-path, and answer-boundary checks.

Entity/period mismatch cases should not be treated as answerable merely because a related store or metric is retrieved.

Ambiguous comparison cases should be narrowed by metric and operating question before the system makes a comparison.

Because the current corpus is small, this score distribution is used for inspection of retrieval behavior and failure modes rather than production-level threshold validation.

## Outputs

- Detail CSV: `retail_ops/outputs/retrieval_score_distribution.csv`
- Score distribution plot: `retail_ops/outputs/retrieval_score_distribution.png`
