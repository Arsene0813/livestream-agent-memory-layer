# Retrieval Query Robustness Summary

## Purpose

This small-corpus check inspects whether retrieval behavior remains reasonably stable when the same query intent is expressed with small wording changes.

It does not prove production-level retrieval robustness. It is a diagnostic evaluation for the current file-backed retail decision-support prototype.

## Inputs

- Cases: `eval/retrieval_threshold_cases.json`
- Retail memory facts: `retail_ops/outputs/generated_retail_memory_facts.json`
- Demo 2 memory facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- Dictionary context: `retail_ops/data/DATA_DICTIONARY.md`
- Demo 2 source notes: `retail_ops/data/demo2_source_notes.md`
- Embedding model: `bge-m3`
- Reference threshold: `0.5707`

## Variant Types

Each original query is evaluated with deterministic wording variants:

- `original`
- `shortened`
- `paraphrased`
- `typo_punctuation_noise`
- `keyword_order_changed`

## Robustness by Case Type

| case_type | variant_count | expected_hit_at_5_count | expected_hit_at_5_rate | above_reference_threshold_count | above_reference_threshold_rate | top1_changed_non_original_count | top1_changed_non_original_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_comparison | 16 | 13 | 81.25% | 2 | 12.5% | 4 | 33.33% |
| entity_period_mismatch | 17 | 17 | 100.0% | 17 | 100.0% | 3 | 23.08% |
| hard_negative_boundary | 33 | 33 | 100.0% | 18 | 54.55% | 8 | 30.77% |
| negative_unsupported | 30 | 0 | 0.0% | 3 | 10.0% | 4 | 16.67% |
| positive_supported | 35 | 35 | 100.0% | 27 | 77.14% | 1 | 3.7% |

## Threshold Sweep

This sweep is not an optimization procedure. It shows how many query variants remain above several simple threshold values.

| threshold | variants_above_threshold | variants_above_threshold_rate |
| --- | --- | --- |
| 0.5 | 112 | 85.5% |
| 0.55 | 92 | 70.23% |
| 0.6 | 48 | 36.64% |
| 0.65 | 35 | 26.72% |
| 0.7 | 13 | 9.92% |

The full threshold sweep by case type is stored in:

- `retail_ops/outputs/retrieval_query_threshold_sweep.csv`

## Interpretation Boundary

Supported cases should generally retain expected evidence in top-k under small wording changes.

Hard-negative, entity/period-mismatch, and ambiguous comparison cases may still remain semantically close to valid evidence. That behavior reinforces the current design: retrieval threshold is useful as one signal, but it cannot be treated as an answer-decision rule.

Unsupported cases should not become answerable merely because wording changes.

Final answer behavior should still depend on entity, period, slot, source-path, and interpretation-boundary checks.
