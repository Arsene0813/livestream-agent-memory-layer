# Retrieval Threshold Inspection Summary

This file summarizes prototype retrieval score distributions over file-backed retail evidence.

The retrieval corpus combines generated Demo 1 and Demo 2 retail memory facts with selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.

The current project does not connect to the live Meituan backend. The evidence used here comes from manually structured Meituan-style backend data and generated local memory facts.

The purpose is to inspect retrieval-threshold behavior. It is not a production-level threshold validation, a broad LLM benchmark, or proof that retrieved evidence is sufficient for an operating conclusion.

## Corpus

- Retrieval documents loaded: 282
- Corpus SHA-256: `142368acc56e40a7dee55aabd65e4bfea719f29052b1c7d84a82b4a5654726f3`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Generated from commit: `09b4e118a7a2f0103555854f9cd41850530924b5`
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
| ambiguous_comparison | 4 | 0.5275 | 0.5399 | 0.5444 | 0.5572 | 0.5941 | 0.0021 | 3/4 |
| entity_period_mismatch | 4 | 0.5538 | 0.5696 | 0.5897 | 0.6156 | 0.6489 | 0.0061 | 1/4 |
| hard_negative_boundary | 7 | 0.5071 | 0.5688 | 0.6319 | 0.6739 | 0.7153 | 0.0068 | 5/7 |
| negative_unsupported | 6 | 0.4582 | 0.4933 | 0.4971 | 0.5292 | 0.5602 | 0.0176 | 0/6 |
| positive_supported | 8 | 0.6049 | 0.6243 | 0.6510 | 0.7071 | 0.7571 | 0.0337 | 8/8 |

## Threshold Interpretation

A reference threshold can be inspected for the trade-off between unsupported retrieval noise and supported evidence retention.

An exploratory reference threshold from this small corpus is around `0.5767`, midway between the positive-supported p25 score `0.6243` and the negative-unsupported p75 score `0.5292`. This is an inspection reference, not a production cutoff or an answer-decision rule.

High scores in hard-negative boundary cases are expected. They show that a semantically related fact can be retrieved even when the correct answer should still refuse or qualify the requested conclusion.

For that reason, retrieval thresholding must be paired with entity, period, slot, source-path, and answer-boundary checks.

Entity/period mismatch cases should not be treated as answerable merely because a related store or metric is retrieved.

Ambiguous comparison cases should be narrowed by metric and operating question before the system makes a comparison.

Because the current corpus is small, this score distribution is used for inspection of retrieval behavior and failure modes rather than production-level threshold validation.

## Outputs

- Detail CSV: `retail_ops/outputs/retrieval_score_distribution.csv`
- Score distribution plot: `retail_ops/outputs/retrieval_score_distribution.png`
