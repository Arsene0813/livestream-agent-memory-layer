# Retrieval Threshold Inspection Summary

This file summarizes prototype retrieval score distributions over file-backed retail evidence.

The retrieval corpus combines generated Demo 1 and Demo 2 retail memory facts with selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.

The retrieval evidence consists of selected observations manually transcribed from the Meituan merchant backend, together with generated local memory facts; the experiment runs on the repository snapshot rather than a live backend connection.

The purpose is to inspect retrieval-threshold behavior. It is not a production-level threshold validation, a broad LLM benchmark, or proof that retrieved evidence is sufficient for an operating conclusion.

## Corpus

- Retrieval units loaded: 286
- Unit definition: one generated memory fact or one chunked field-contract/source-note segment; this is not a store count or a count of independent business observations.
- Corpus SHA-256: `39f39e7f10c35c9c849bbc577020dedbf3bd08f6bf24e6f8816c264bcefdc6e6`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Execution commit: `be35bfb8e47ed940f655e31a39c5c6feda6cc37d`
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
| ambiguous_comparison | 4 | 0.5279 | 0.5377 | 0.5429 | 0.5573 | 0.5945 | 0.0026 | 2/4 |
| entity_period_mismatch | 4 | 0.5552 | 0.5702 | 0.5896 | 0.6155 | 0.6497 | 0.0068 | 1/4 |
| hard_negative_boundary | 7 | 0.5054 | 0.5712 | 0.6270 | 0.6787 | 0.7137 | 0.0090 | 5/7 |
| negative_unsupported | 6 | 0.4566 | 0.4952 | 0.4978 | 0.5290 | 0.5588 | 0.0181 | 0/6 |
| positive_supported | 8 | 0.6044 | 0.6248 | 0.6506 | 0.7109 | 0.7594 | 0.0344 | 8/8 |

## Threshold Interpretation

A reference threshold can be inspected for the trade-off between unsupported retrieval noise and supported evidence retention.

An exploratory reference threshold from this small corpus is around `0.5769`, midway between the positive-supported p25 score `0.6248` and the negative-unsupported p75 score `0.5290`. This is an inspection reference, not a production cutoff or an answer-decision rule.

High scores in hard-negative boundary cases are expected. They show that a semantically related fact can be retrieved even when the correct answer should still refuse or qualify the requested conclusion.

For that reason, retrieval thresholding must be paired with entity, period, slot, source-path, and answer-boundary checks.

Entity/period mismatch cases should not be treated as answerable merely because a related store or metric is retrieved.

Ambiguous comparison cases should be narrowed by metric and operating question before the system makes a comparison.

Because the current corpus is small, this score distribution is used for inspection of retrieval behavior and failure modes rather than production-level threshold validation.

## Outputs

- Detail CSV: `retail_ops/outputs/retrieval_score_distribution.csv`
- Score distribution plot: `retail_ops/outputs/retrieval_score_distribution.png`
