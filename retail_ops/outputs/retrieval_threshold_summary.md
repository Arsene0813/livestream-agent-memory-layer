# Retrieval Threshold Inspection Summary

This file summarizes prototype retrieval score distributions over file-backed retail evidence.

The retrieval corpus combines generated Demo 1 and Demo 2 retail memory facts with selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.

The retrieval evidence consists of selected observations manually transcribed from the Meituan merchant backend, together with generated local memory facts; the experiment runs on the repository snapshot rather than a live backend connection.

The purpose is to inspect retrieval-threshold behavior. It is not a production-level threshold validation, a broad LLM benchmark, or proof that retrieved evidence is sufficient for an operating conclusion.

## Corpus

- Retrieval units loaded: 302
- Unit definition: one generated memory fact or one chunked field-contract/source-note segment; this is not a store count or a count of independent business observations.
- Corpus SHA-256: `1f8fd5a7b42d875d2a4f72105d1f43db0a9e05883a6a8cc3fafc4717799eb850`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Run commit: `8e697d2be6bc2eef96ce5252fb6ee9fbca62c80d`
- Experiment scope SHA-256: `d92a7bb3ca12d08fac907a89ad1178b5b6782e136e1f6ef5760028bdd4e68431`
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
| ambiguous_comparison | 4 | 0.5279 | 0.5374 | 0.5427 | 0.5549 | 0.5852 | 0.0043 | 3/4 |
| entity_period_mismatch | 4 | 0.5555 | 0.5776 | 0.5999 | 0.6238 | 0.6504 | 0.0088 | 1/4 |
| hard_negative_boundary | 7 | 0.5055 | 0.5711 | 0.6271 | 0.6784 | 0.7139 | 0.0085 | 5/7 |
| negative_unsupported | 6 | 0.4566 | 0.4933 | 0.4957 | 0.5290 | 0.5561 | 0.0085 | 0/6 |
| positive_supported | 8 | 0.5809 | 0.6150 | 0.6441 | 0.7108 | 0.7593 | 0.0213 | 8/8 |

## Threshold Interpretation

A reference threshold can be inspected for the trade-off between unsupported retrieval noise and supported evidence retention.

An exploratory reference threshold from this small corpus is around `0.5720`, midway between the positive-supported p25 score `0.6150` and the negative-unsupported p75 score `0.5290`. This is an inspection reference, not a production cutoff or an answer-decision rule.

High scores in hard-negative boundary cases are expected. They show that a semantically related fact can be retrieved even when the correct answer should still refuse or qualify the requested conclusion.

For that reason, retrieval thresholding must be paired with entity, period, slot, source-path, and answer-boundary checks.

Entity/period mismatch cases should not be treated as answerable merely because a related store or metric is retrieved.

Ambiguous comparison cases should be narrowed by metric and operating question before the system makes a comparison.

Because the current corpus is small, this score distribution is used for inspection of retrieval behavior and failure modes rather than production-level threshold validation.

## Outputs

- Detail CSV: `retail_ops/outputs/retrieval_score_distribution.csv`
- Score distribution plot: `retail_ops/outputs/retrieval_score_distribution.png`
