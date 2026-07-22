# Retrieval Threshold Inspection Summary

This file summarizes prototype retrieval score distributions over file-backed retail evidence.

The retrieval corpus combines generated Demo 1 and Demo 2 retail memory facts with selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.

The retrieval evidence consists of selected observations manually transcribed from the Meituan merchant backend, together with generated local memory facts; the experiment runs on the repository snapshot rather than a live backend connection.

The purpose is to inspect retrieval-threshold behavior. It is not a production-level threshold validation, a broad LLM benchmark, or proof that retrieved evidence is sufficient for an operating conclusion.

## Corpus

- Retrieval documents loaded: 283
- Corpus SHA-256: `db4af4728ae54a15f2332e77459c151fd2e32288817e16fba0f401c4a789b473`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Execution commit: `cb263686677068e1e8e0a8bde28e467a39405964`
- Provenance note: the corpus SHA-256 identifies the evidence snapshot; the execution commit identifies the code state used for the run.
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
| ambiguous_comparison | 4 | 0.5275 | 0.5390 | 0.5438 | 0.5572 | 0.5941 | 0.0021 | 3/4 |
| entity_period_mismatch | 4 | 0.5574 | 0.5705 | 0.5897 | 0.6158 | 0.6498 | 0.0083 | 1/4 |
| hard_negative_boundary | 7 | 0.5071 | 0.5677 | 0.6292 | 0.6747 | 0.7105 | 0.0068 | 5/7 |
| negative_unsupported | 6 | 0.4552 | 0.4943 | 0.4971 | 0.5292 | 0.5602 | 0.0161 | 0/6 |
| positive_supported | 8 | 0.6049 | 0.6243 | 0.6507 | 0.7090 | 0.7553 | 0.0382 | 8/8 |

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
