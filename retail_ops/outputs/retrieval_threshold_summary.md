# Retrieval Threshold Calibration Summary

This file summarizes prototype retrieval score distributions over file-backed retail evidence.

The retrieval corpus combines generated Demo 1 and Demo 2 retail memory facts with selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.

The current project does not connect to the live Meituan backend. The evidence used here comes from manually structured Meituan-style backend data and generated local memory facts.

The purpose is to inspect retrieval-threshold behavior. It is not a production-level threshold validation, a broad LLM benchmark, or proof that retrieved evidence is sufficient for an operating conclusion.

## Corpus

- Retrieval documents loaded: 236
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
| ambiguous_comparison | 4 | 0.5326 | 0.5333 | 0.5379 | 0.5447 | 0.5519 | 0.0058 | 3/4 |
| entity_period_mismatch | 4 | 0.5809 | 0.6055 | 0.6307 | 0.6478 | 0.6481 | 0.0132 | 1/4 |
| hard_negative_boundary | 7 | 0.5057 | 0.5860 | 0.6335 | 0.6754 | 0.7149 | 0.0298 | 5/7 |
| negative_unsupported | 6 | 0.4559 | 0.4943 | 0.4968 | 0.5293 | 0.5567 | 0.0062 | 0/6 |
| positive_supported | 8 | 0.5666 | 0.6121 | 0.6416 | 0.7088 | 0.7567 | 0.0380 | 8/8 |

## Threshold Interpretation

This calibration is an offline embedding-retrieval inspection over the current file-backed retail evidence corpus. It does not set the runtime behavior of `/chat_retail_ops_demo2_kb`.

The current Demo 2 endpoint selects generated Demo 2 memory facts through file-backed entity, slot, and scope logic. Answer safety still depends on entity, period, slot, source-path, and answer-boundary checks rather than score alone.

A candidate threshold should reduce unsupported retrieval noise while keeping most supported evidence available for answer generation. One rough candidate threshold from this small corpus is around `0.5707`, midway between the positive-supported p25 score `0.6121` and the negative-unsupported p75 score `0.5293`.

This is a prototype calibration reference, not a production cutoff. High scores in hard-negative boundary cases are expected. They show that a semantically related fact can be retrieved even when the correct answer should still refuse or qualify the requested conclusion.

For that reason, retrieval thresholding must be paired with entity, period, slot, source-path, and answer-boundary checks. Entity/period mismatch cases should not be treated as answerable merely because a related store or metric is retrieved. Ambiguous comparison cases should be narrowed by metric and operating question before the system makes a comparison.

Because the current corpus is small, this score distribution is used for prototype calibration rather than production-level threshold validation.

## Outputs

- Detail CSV: `retail_ops/outputs/retrieval_score_distribution.csv`
- Score distribution plot: `retail_ops/outputs/retrieval_score_distribution.png`
