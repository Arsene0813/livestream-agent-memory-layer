# Retail Operations Experiment Results

Reviewer-facing companion: `retail_ops/EXPERIMENTS.md` explains the purpose of each analytical check.

## Compact Validation Summary

| Validation target | Review value | Current result |
|---|---|---|
| Metric contract | Checks whether selected Meituan backend fields keep canonical names, documented meanings, and generated-fact structure. | Passed. |
| Demo 1 diagnostic | Shows that Store A monthly movement can be structured as a multi-metric operating profile. | Implemented. |
| Demo 2 diagnostic SQL | Places selected B-F March 2026 store-period records under one reporting window and one field contract. | Implemented. |
| Demo 2 memory facts | Converts SQL diagnostic evidence into retrieval-facing facts with source fields, observed values, and limitation notes. | Implemented. |
| Answer-boundary behavior | Checks whether answers preserve metric definitions, entity scope, period scope, and comparison limits. | Implemented as offline scenario checks. |
| Demo 2 endpoint behavior | Checks whether the file-backed Demo 2 endpoint answers supported B-F questions and keeps unsupported requests outside current scope. | Implemented as endpoint-level checks. |
| Retrieval threshold inspection | Makes score behavior inspectable across supported, unsupported, mismatch, and ambiguous retail queries. | Completed as offline inspection. |
| Query robustness inspection | Inspects whether small wording changes still retrieve the expected evidence path. | Completed as offline inspection. |
| Future comparability-gate contract | Documents the planned pairwise gate as a question-specific future stage. | Contract documented. |
| RAC grounded review | Makes factor expansion, evidence routing, critique, fact checking, and confidence updates visible over local project evidence. | Implemented as deterministic local-evidence scaffold. |

These checks are designed for a staged decision-support prototype. They show whether the current data path preserves field meanings, source scope, diagnostic boundaries, and reviewer-traceable evidence.

Each record below follows the same structure: question, evidence path, expected behavior, current result, and checked-by command.

## Experiment 1: Store A Month-over-Month Diagnostic

Question: Can selected Meituan backend metrics for one store be organized into a month-over-month diagnostic without changing the backend metric meanings?

Evidence path:

- Source data: `retail_ops/data/store_a_monthly_metrics.csv`
- SQL: `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- Output: `retail_ops/outputs/store_a_demo1_sql_output.csv`
- Generated facts: `retail_ops/outputs/generated_retail_memory_facts.json`

Expected behavior:

The output may describe observed month-over-month movement, but it should not attribute performance change to one metric alone.

Current result: Implemented.

Checked by:

- `python3 retail_ops/scripts/validate_retail_data_contract.py`

## Experiment 2: Demo 2 Same-Period Store Diagnostic

Question: Can selected B-F store-period rows be placed under one March 2026 reporting window and one field contract before any stronger comparison is attempted?

Evidence path:

- Source data: `retail_ops/data/demo2_store_period_metrics.csv`
- SQL: `retail_ops/sql/02_demo2_cross_store_comparability.sql`
- Output: `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

Expected behavior:

The SQL output should include `comparison_scope_flag` and `comparison_limit_notes`, while staying at row-level same-period diagnostic scope.

Current result: Implemented.

Checked by:

- `python3 eval/eval_retail_demo2_scope_boundary.py`

Result path:

- `eval/results/eval_retail_demo2_scope_boundary_result.txt`

## Experiment 2A: Demo 2 Guardrail Sensitivity Check

Question: Are the current Demo 2 `comparison_limit_notes` stable under small threshold changes?

Evidence path:

- Script: `retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py`
- Input: `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- Output CSV: `retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv`
- Output note: `retail_ops/outputs/demo2_guardrail_sensitivity_result.txt`

Expected behavior:

The check should not optimize thresholds or turn them into peer-selection rules. It should only show whether the current threshold-based guardrail notes are sensitive to small threshold shifts.

Current result: Implemented.

In the current B-F sample, all five stores have guardrail notes that change under at least one +/- 5 percentage-point sensitivity scenario. This means the current thresholds should be treated as diagnostic warnings, not stable peer-comparison rules.

Checked by:

- `python3 retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py`

## Experiment 3: Demo 2 Memory-Fact Generation

Question: Can the Demo 2 diagnostic output be converted into retrieval-facing memory facts without losing source fields, observed values, or limitation notes?

Evidence path:

- Generator: `retail_ops/scripts/generate_demo2_retail_memory_facts.py`
- Generated facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- Eval: `eval/eval_retail_demo2_facts.py`

Expected behavior:

Generated facts should preserve canonical field names and expose the main evidence slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `top3_sku_product_mix_note`
- `single_metric_attribution_guard`

Current result: Implemented.

Checked by:

- `python3 eval/eval_retail_demo2_facts.py`

Result path:

- `eval/results/eval_retail_demo2_facts_result.txt`

## Experiment 4: Demo 2 Answer-Boundary Contract Check

Question: Can expected answer patterns preserve metric boundaries when Demo 2 evidence is used?

Evidence path:

- Eval: `eval/eval_retail_demo2_answer_behavior.py`
- SQL output: `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- Generated facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

Expected behavior:

The answer-boundary contract should preserve these rules:

- `activity_cost_ratio_pct` is not traditional ROI or profit margin.
- `top3_sku_transaction_amount_share_pct` is not full product-category sales share.
- Search-entry evidence is treated as one visibility-to-entry signal within the current store-period profile.
- Activity evidence describes operating-tool usage, not automatic promotion-transfer logic.
- `same_period_diagnostic_ready` is not a finished pairwise comparability decision.
- `region_type` is weak context only.

Current result: Implemented as offline scenario checks.

Checked by:

- `python3 eval/eval_retail_demo2_answer_behavior.py`

Result path:

- `eval/results/eval_retail_demo2_answer_behavior_result.txt`

## Experiment 4A: Demo 2 Endpoint-Boundary Contract Check

Question: Does the implemented `/chat_retail_ops_demo2_kb` endpoint preserve the same evidence boundaries when answering from file-backed Demo 2 memory facts?

Evidence path:

- Endpoint: `/chat_retail_ops_demo2_kb`
- API implementation: `api/main.py`
- Generated facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- Eval: `eval/eval_retail_demo2_endpoint_behavior.py`

Expected behavior:

- supported Store B-F questions return file-backed Demo 2 facts;
- cross-store B-F questions stay at same-period diagnostic scope;
- all-48-store questions are refused;
- best-store ranking and final operating recommendations are refused;
- out-of-Demo-2 entities are refused.

Current result: Implemented as endpoint-level boundary checks.

Checked by:

- `python3 eval/eval_retail_demo2_endpoint_behavior.py`

Result path:

- `eval/results/eval_retail_demo2_endpoint_behavior_result.txt`

## Experiment 5: Retrieval Threshold Inspection

Question: Can the current retrieval threshold be explained from score distributions rather than isolated successful examples?

Evidence path:

- Cases: `eval/retrieval_threshold_cases.json`
- Analysis script: `eval/analyze_retail_embedding_score_distribution.py`
- Output CSV: `retail_ops/outputs/retrieval_score_distribution.csv`
- Output plot: `retail_ops/outputs/retrieval_score_distribution.png`
- Summary: `retail_ops/outputs/retrieval_threshold_summary.md`

Expected behavior:

The analysis should make retrieval behavior inspectable across supported, unsupported, hard-negative, entity/period-mismatch, and ambiguous comparison queries.

A high retrieval score should not be treated as sufficient evidence for an operating conclusion.

Current result: Implemented as an offline small-corpus retrieval inspection.

Boundary:

This is not the runtime threshold logic for `/chat_retail_ops_demo2_kb`, not a production-level retrieval benchmark, and not proof that retrieved evidence is sufficient for final operating advice.

## Experiment 6: Query Robustness Inspection

Question: Does retrieval behavior remain reasonably stable when the same query intent is expressed with small wording changes?

Evidence path:

- Analysis script: `eval/analyze_retail_query_robustness.py`
- Output CSV: `retail_ops/outputs/retrieval_query_robustness.csv`
- Threshold sweep: `retail_ops/outputs/retrieval_query_threshold_sweep.csv`
- Summary: `retail_ops/outputs/retrieval_query_robustness_summary.md`

Expected behavior:

Supported cases should generally retain expected evidence in top-k under small wording changes.

Unsupported, hard-negative, entity/period-mismatch, and ambiguous comparison cases should still require entity, period, slot, source-path, and interpretation-boundary checks.

Current result: Implemented as an offline small-corpus robustness inspection.

Boundary:

This check supports the current design choice that retrieval score is only one signal and should be paired with answer-boundary checks.

## Experiment 7: Future Gate Contract Check

Question: Can the project document a future pairwise comparability gate without accidentally exposing it as a finished current feature?

Evidence path:

- Design note: `retail_ops/COMPARABILITY_GATE_V0.md`
- Eval: `eval/eval_future_comparability_gate_contract.py`

Expected behavior:

The future gate may define planned factors such as transaction order volume, transaction amount, activity status, activity intensity, store type, region and market context, SKU structure, and repeated reporting windows.

It should not appear as a current implemented gate in Demo 2 outputs.

Current result: Contract documented as future work.

Checked by:

- `python3 eval/eval_future_comparability_gate_contract.py`

Result path:
- `eval/results/eval_future_comparability_gate_contract_result.txt`

## Experiment 8: Retail Data Contract Check

Question: Do the current retail data dictionary, source files, SQL outputs, and generated memory facts preserve the same Demo 1 / Demo 2 field contract?

Evidence path:

- Data-contract validator: `retail_ops/scripts/validate_retail_data_contract.py`

Expected behavior:

The data contract should keep canonical field names aligned across source CSV files, SQL outputs, generated memory facts, and dictionary boundary notes.

Current result: Implemented.

Checked by:

- `python3 retail_ops/scripts/validate_retail_data_contract.py`

## Repeated-Window Panel Extension Result

The repeated-window panel extension currently covers Store B through Store F across February, March, and April 2026.

| Result item | Current result |
|---|---|
| Stores covered | B, C, D, E, F |
| Months covered | 2026-02, 2026-03, 2026-04 |
| Coverage output | `retail_ops/outputs/store_period_panel_coverage_output.csv` |
| Coverage validator | `python3 retail_ops/scripts/validate_store_period_panel.py` |
| Coverage validator result | PASS |
| Coverage flag | `panel_ready_for_repeated_window_diagnostic` for B-F |
| Descriptive summary output | `retail_ops/outputs/repeated_window_panel_summary_output.csv` |
| Descriptive summary validator | `python3 retail_ops/scripts/validate_repeated_window_panel_summary.py` |
| Descriptive summary validator result | PASS |
| Descriptive summary flag | `summary_ready_for_descriptive_review` for B-F |
| Excluded fields | Unclear backend order-status fields. |
| Interpretation boundary | Repeated-window coverage and descriptive summary only; not a new numbered demo, pairwise comparability gate, endpoint behavior, generated memory facts, store ranking, operating recommendation, or causal analysis. |

The panel extension strengthens the data foundation after the current Demo 2 same-period diagnostic by adding repeated reporting windows and a descriptive SQL summary while preserving the existing field-boundary discipline.

## Repeated-Window Panel Validation Results

The post-Demo2 repeated-window panel extension is validated by two saved result files.

| Validator | Saved result | What it checks |
|---|---|---|
| `retail_ops/scripts/validate_store_period_panel.py` | `retail_ops/outputs/store_period_panel_validation_result.txt` | Stores B-F each have 2026-02, 2026-03, and 2026-04; panel uses canonical source fields; `store_type` values are canonical; schema stays within the documented source/output contract. |
| `retail_ops/scripts/validate_repeated_window_panel_summary.py` | `retail_ops/outputs/repeated_window_panel_summary_validation_result.txt` | Repeated-window summary contains Stores B-F, keeps three observed months per store, uses canonical `store_type` values, and remains descriptive rather than causal or ranking-oriented. |

These checks support panel readiness for descriptive review. They do not convert the panel into a pairwise comparability gate.
