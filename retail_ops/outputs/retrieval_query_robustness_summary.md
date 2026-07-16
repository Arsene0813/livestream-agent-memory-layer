# Retrieval Query Wording-Variation Stress-Test Summary

## Purpose

This small-corpus stress test records how retrieval behavior changes when the same query intent is expressed with deterministic wording variations.

It is a diagnostic evaluation for the current file-backed retail decision-support prototype.

## Inputs

- Cases: `eval/retrieval_threshold_cases.json`
- Retail memory facts: `retail_ops/outputs/generated_retail_memory_facts.json`
- Demo 2 memory facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- Dictionary context: `retail_ops/data/DATA_DICTIONARY.md`
- Demo 2 source notes: `retail_ops/data/demo2_source_notes.md`
- Corpus documents: 282
- Corpus SHA-256: `142368acc56e40a7dee55aabd65e4bfea719f29052b1c7d84a82b4a5654726f3`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Generated from commit: `e598a640edceda32f538cda8dfe17cf634f4bcfc`
- Embedding model: `bge-m3`
- Reference threshold: `0.5767`
- Reference threshold source: `retail_ops/outputs/retrieval_threshold_summary.md`

## Variant Types

Each original query is evaluated with deterministic wording variants:

- `original`
- `shortened`
- `paraphrased`
- `typo_punctuation_noise`
- `keyword_order_changed`

## Results by Case Type

| case_type | variant_count | expected_hit_at_5_count | expected_hit_at_5_rate | above_reference_threshold_count | above_reference_threshold_rate | top1_changed_non_original_count | top1_changed_non_original_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_comparison | 16 | 12 | 75.0% | 5 | 31.25% | 5 | 41.67% |
| entity_period_mismatch | 18 | 13 | 72.22% | 12 | 66.67% | 5 | 35.71% |
| hard_negative_boundary | 33 | 28 | 84.85% | 23 | 69.7% | 8 | 30.77% |
| negative_unsupported | 30 | 0 | 0.0% | 0 | 0.0% | 4 | 16.67% |
| positive_supported | 34 | 34 | 100.0% | 34 | 100.0% | 2 | 7.69% |

## Threshold Sweep

This sweep is not an optimization procedure. It shows how many query variants remain above several simple threshold values.

| threshold | variants_above_threshold | variants_above_threshold_rate |
| --- | --- | --- |
| 0.5 | 117 | 89.31% |
| 0.55 | 87 | 66.41% |
| 0.6 | 61 | 46.56% |
| 0.65 | 34 | 25.95% |
| 0.7 | 14 | 10.69% |

The full threshold sweep by case type is stored in:

- `retail_ops/outputs/retrieval_query_threshold_sweep.csv`

## Interpretation Boundary

Supported cases should generally retain expected evidence in top-k under small wording changes.

Hard-negative, entity/period-mismatch, and ambiguous comparison cases may still remain semantically close to valid evidence. That behavior reinforces the current design: retrieval threshold is useful as one signal, but it cannot be treated as an answer-decision rule.

Unsupported cases should not become answerable merely because wording changes.

Final answer behavior should still depend on entity, period, slot, source-path, and interpretation-boundary checks.
