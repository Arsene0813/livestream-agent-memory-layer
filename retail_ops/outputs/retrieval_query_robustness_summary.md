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
- Retrieval units: 302

- Unit definition: one generated memory fact or one chunked field-contract/source-note segment; this is not a store count or a count of independent business observations.
- Corpus SHA-256: `76d706b500fc81090a2d00b9e3069e4c46c1d6a2cd052a439ec8b9df7302aa77`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Run commit: `2dbc9694c57ae644de3e0b898ae3c244904404bc`
- Experiment scope SHA-256: `0a0d6c96ec670ef5892ef48da1552426da10a4d6353767ba843d549b5246a41f`
- Applicability note: the scope hash identifies the clean experiment-relevant code and input snapshot. The run commit is retained for navigation. Later unrelated commits do not invalidate the result. Run `python3 eval/check_retrieval_result_applicability.py` to check the current scope.
- Embedding model: `bge-m3`
- Reference threshold: `0.5776`
- Reference threshold source: `retail_ops/outputs/retrieval_threshold_summary.md`
- Reference threshold mode: `summary_source`

## Variant Types

Each original query is evaluated with deterministic wording variants:

- `original`
- `shortened`
- `paraphrased`
- `typo_punctuation_noise`
- `keyword_order_changed`

## Expected-Hit Contract

For each non-negative case, `expected_hit_at_5` is true only when at least one top-5 retrieval unit satisfies all applicable `entity_id`, slot, period, and expected-term constraints.

`negative_unsupported` cases are always recorded without an expected evidence hit. Semantic similarity or a single matching keyword is not sufficient.

## Results by Case Type

| case_type | variant_count | expected_hit_at_5_count | expected_hit_at_5_rate | above_reference_threshold_count | above_reference_threshold_rate | top1_changed_non_original_count | top1_changed_non_original_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ambiguous_comparison | 16 | 11 | 68.75% | 5 | 31.25% | 3 | 25.0% |
| entity_period_mismatch | 18 | 2 | 11.11% | 14 | 77.78% | 4 | 28.57% |
| hard_negative_boundary | 33 | 25 | 75.76% | 23 | 69.7% | 6 | 23.08% |
| negative_unsupported | 30 | 0 | 0.0% | 0 | 0.0% | 7 | 29.17% |
| positive_supported | 34 | 34 | 100.0% | 34 | 100.0% | 4 | 15.38% |

## Threshold Sweep

This sweep is not an optimization procedure. It shows how many query variants remain above several simple threshold values.

| threshold | variants_above_threshold | variants_above_threshold_rate |
| --- | --- | --- |
| 0.5 | 115 | 87.79% |
| 0.55 | 88 | 67.18% |
| 0.6 | 64 | 48.85% |
| 0.65 | 36 | 27.48% |
| 0.7 | 14 | 10.69% |

The full threshold sweep by case type is stored in:

- `retail_ops/outputs/retrieval_query_threshold_sweep.csv`

## Interpretation Boundary

Supported cases should generally retain expected evidence in top-k under small wording changes.

Hard-negative, entity/period-mismatch, and ambiguous comparison cases may still remain semantically close to valid evidence. That behavior reinforces the current design: retrieval threshold is useful as one signal, but it cannot be treated as an answer-decision rule.

Unsupported cases should not become answerable merely because wording changes.

`top1_changed_non_original_rate` is descriptive of the current corpus and embedding runtime; it is not evidence of model improvement. The experiment records the model name but does not fingerprint the local Ollama model binary.

Final answer behavior should still depend on entity, period, slot, source-path, and interpretation-boundary checks.
