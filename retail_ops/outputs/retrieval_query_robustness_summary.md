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
- Corpus SHA-256: `3e42fa1cc2457385a4cebba21bd2bbc1b92f9397cd4b6749161da70ad715a278`
- Corpus builder: `eval/retail_retrieval_corpus.py::load_retail_retrieval_documents`
- Run commit: `aa47fa938b7fe48ae090950f35de57c4527988c4`
- Experiment scope SHA-256: `2cafa72199b3c71b147258b9c71887c2ac59c12977051ea2da4e48d2fbe1060b`
- Applicability note: the scope hash identifies the clean experiment-relevant code and input snapshot. The run commit is retained for navigation. Later unrelated commits do not invalidate the result. Run `python3 eval/check_retrieval_result_applicability.py` to check the current scope.
- Embedding model: `bge-m3`
- Reference threshold: `0.5802`
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
| ambiguous_comparison | 16 | 13 | 81.25% | 3 | 18.75% | 4 | 33.33% |
| entity_period_mismatch | 17 | 3 | 17.65% | 16 | 94.12% | 3 | 23.08% |
| hard_negative_boundary | 34 | 20 | 58.82% | 24 | 70.59% | 11 | 40.74% |
| negative_unsupported | 30 | 0 | 0.0% | 1 | 3.33% | 8 | 33.33% |
| positive_supported | 34 | 30 | 88.24% | 34 | 100.0% | 0 | 0.0% |

## Threshold Sweep

This sweep is not an optimization procedure. It shows how many query variants remain above several simple threshold values.

| threshold | variants_above_threshold | variants_above_threshold_rate |
| --- | --- | --- |
| 0.5 | 115 | 87.79% |
| 0.55 | 92 | 70.23% |
| 0.6 | 64 | 48.85% |
| 0.65 | 37 | 28.24% |
| 0.7 | 14 | 10.69% |

The full threshold sweep by case type is stored in:

- `retail_ops/outputs/retrieval_query_threshold_sweep.csv`

## Interpretation Boundary

Supported cases should generally retain expected evidence in top-k under small wording changes.

Hard-negative, entity/period-mismatch, and ambiguous comparison cases may still remain semantically close to valid evidence. That behavior reinforces the current design: retrieval threshold is useful as one signal, but it cannot be treated as an answer-decision rule.

Unsupported cases should not become answerable merely because wording changes.

`top1_changed_non_original_rate` is descriptive of the current corpus and embedding runtime; it is not evidence of model improvement. The experiment records the model name but does not fingerprint the local Ollama model binary.

Final answer behavior should still depend on entity, period, slot, source-path, and interpretation-boundary checks.
