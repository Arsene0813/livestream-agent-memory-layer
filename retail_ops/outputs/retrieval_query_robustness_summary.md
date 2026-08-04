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
- Retrieval units: 307

- Unit definition: one generated memory fact or one chunked field-contract/source-note segment; this is not a store count or a count of independent business observations.
- Corpus SHA-256: `d11e5c2eb1ff26461056cc2412b8e827e31ec57d2b378e799b998cc3de33b19c`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Run commit: `1c542a7fd2bea59da232a0873584db21a861aff1`
- Experiment scope SHA-256: `b47dee68bcb61051317215f5f698bc3a79e0d49759453ded4f977d39189a8ece`
- Applicability note: the scope hash identifies the clean experiment-relevant code and input snapshot. The run commit is retained for navigation. Later unrelated commits do not invalidate the result. Run `python3 eval/check_retrieval_result_applicability.py` to check the current scope.
- Embedding model: `bge-m3`
- Reference threshold: `0.5805`
- Reference threshold source: `retail_ops/outputs/retrieval_threshold_summary.md`
- Reference threshold mode: `cli_override`

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
| ambiguous_comparison | 16 | 13 | 81.25% | 3 | 18.75% | 4 | 33.33% |
| entity_period_mismatch | 18 | 3 | 16.67% | 13 | 72.22% | 3 | 21.43% |
| hard_negative_boundary | 33 | 24 | 72.73% | 23 | 69.7% | 9 | 34.62% |
| negative_unsupported | 30 | 0 | 0.0% | 1 | 3.33% | 5 | 20.83% |
| positive_supported | 34 | 30 | 88.24% | 34 | 100.0% | 0 | 0.0% |

## Threshold Sweep

This sweep is not an optimization procedure. It shows how many query variants remain above several simple threshold values.

| threshold | variants_above_threshold | variants_above_threshold_rate |
| --- | --- | --- |
| 0.5 | 115 | 87.79% |
| 0.55 | 91 | 69.47% |
| 0.6 | 59 | 45.04% |
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
