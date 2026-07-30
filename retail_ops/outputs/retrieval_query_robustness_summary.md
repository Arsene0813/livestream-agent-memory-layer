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
- Corpus SHA-256: `1f8fd5a7b42d875d2a4f72105d1f43db0a9e05883a6a8cc3fafc4717799eb850`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Run commit: `77ed40a33b5f4b43afa2d5e3c39e309cb157bca8`
- Experiment scope SHA-256: `b0139d4052b6b9be67a1b0eef08a11e14167528599a5508552fb4e77c06d47d1`
- Applicability note: the scope hash identifies the clean experiment-relevant code and input snapshot. The run commit is retained for navigation. Later unrelated commits do not invalidate the result. Run `python3 eval/check_retrieval_result_applicability.py` to check the current scope.
- Embedding model: `bge-m3`
- Reference threshold: `0.572`
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
| ambiguous_comparison | 16 | 13 | 81.25% | 5 | 31.25% | 5 | 41.67% |
| entity_period_mismatch | 18 | 3 | 16.67% | 15 | 83.33% | 2 | 14.29% |
| hard_negative_boundary | 33 | 25 | 75.76% | 23 | 69.7% | 6 | 23.08% |
| negative_unsupported | 30 | 0 | 0.0% | 1 | 3.33% | 9 | 37.5% |
| positive_supported | 34 | 34 | 100.0% | 34 | 100.0% | 2 | 7.69% |

## Threshold Sweep

This sweep is not an optimization procedure. It shows how many query variants remain above several simple threshold values.

| threshold | variants_above_threshold | variants_above_threshold_rate |
| --- | --- | --- |
| 0.5 | 115 | 87.79% |
| 0.55 | 88 | 67.18% |
| 0.6 | 61 | 46.56% |
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
