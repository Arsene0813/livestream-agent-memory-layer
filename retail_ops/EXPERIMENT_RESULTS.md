# Retail Operations Experiment Map and Results

This file records the retail experiment map, validation results, and boundary checks for the current decision-support prototype.

Retrieval result records:

- `retail_ops/outputs/retrieval_threshold_summary.md` records the current score distributions, expected-hit results, corpus metadata, and exploratory reference threshold.
- `retail_ops/outputs/retrieval_query_robustness_summary.md` records the current wording-variation and threshold-sweep results.
- `python3 eval/check_retrieval_result_applicability.py` verifies whether both summaries still match their declared experiment inputs.

Retrieval counts, scores, and thresholds are read from these generated summaries because they can change when the corpus changes.

## What the Current Experiments Tested

This is the shortest reading path through the current results. The detailed
procedures, output paths, checks, and visible failure cases remain in the
sections below.

| Question | Evidence | Check | Observed result | Limit |
|---|---|---|---|---|
| Can one store be reviewed across repeated months without reducing the result to one cause? | Store A February-April 2026 backend metrics and nine top-SKU rows. | Month-over-month SQL diagnostic and field-level value-lineage validation. | The current check covers three source rows, three SQL output rows, nine top-SKU rows, 180 source, formula, month-over-month, rank-change, and tradeoff comparisons, and five generated facts. | One store and three months support descriptive diagnosis, not causal estimation. |
| Can Stores B-F be organized under one reporting window before stronger comparison is attempted? | Five March 2026 store-period rows with top-search and top-SKU evidence. | Same-period SQL diagnostic and a ±5 percentage-point guardrail sensitivity check. | Baseline notes reproduce for all five rows. Stores C-F change under the +5 scenario, no row changes under the -5 scenario, and Store B remains unchanged. | The thresholds are prototype diagnostic warnings, not validated peer-selection rules. |
| Does retrieval remain reliable when wording or requested scope changes? | Declared retrieval cases and deterministic query variants over the current local evidence corpus. | Top-k inspection, reference-threshold analysis, and wording-variation stress tests. | The applicable run and its exact metrics are recorded in the two generated retrieval summaries. | Semantic similarity can route related evidence without establishing that the requested entity, period, or decision scope is supported. |
| Is there enough repeated-window evidence to inspect movement for Stores B-F? | February-April 2026 records for five stores and 11 selected metrics. | Coverage SQL and a side-by-side repeated-window summary. | Every store has all three observed months, with the selected monthly values and February-to-April endpoint fields reproduced in the committed outputs. | The panel does not contain repeated top-SKU evidence or monthly recomputation of the current guardrail notes. |
| Can RAC expose evidence routing while keeping its coverage score separate from decision quality? | Four fixed RAC evaluation cases over structured records, boundary notes, and repository files. | Factor expansion, evidence routing, critique, claim checks, review-state updates, and report-contract validation. | Current per-case routes and counts are recorded in `rac/outputs/grounded_quality_summary.md`; the quality gate checks B-F record keys, canonical fields, and selected values against their source CSVs. | Routing coverage describes route resolution under the current rules; it is not evidence strength, causal validity, or business impact. |

## What the Results Support

The prototype preserves documented field meanings while moving selected
Meituan backend data through SQL diagnostics, structured evidence, retrieval,
and grounded review.

The main result is traceability. Selected source values can be followed into
derived outputs, generated facts, retrieval packets, and answer boundaries.

The experiments support descriptions of the observed store-period evidence
and documented calculations. They do not turn short-window backend data into
causal estimates, pairwise store decisions, retention measures, market-share
movement, or explanations of unavailable business context.

## Evidence Types

The project uses three evidence types. They answer different questions and
should not be combined into one accuracy benchmark.

| Evidence type | Question answered | Typical result |
|---|---|---|
| Contract and integrity checks | Does the implementation preserve documented names, formulas, paths, metadata, and answer boundaries? | Pass/fail results over declared fixtures and contracts. |
| Descriptive diagnostic analyses | What is visible in the selected store-period evidence under the stated reporting windows? | Observed levels, changes, coverage, and threshold sensitivity. |
| Retrieval behavior stress tests | How does semantic retrieval behave for supported, unsupported, mismatched, hard-negative, and ambiguous queries? | Score distributions, top-k retention, threshold sweeps, and visible failure modes. |

Answer-contract and endpoint-contract checks determine whether an answer
preserves the required entity, period, source, metric definition, and scope.
Retrieval scores answer the narrower question of how evidence is routed. A
high similarity score does not make a query answerable.

The retail data-contract validator is a static check over the selected
implemented fields listed in `REQUIRED_CANONICAL_FIELDS`, declared source and
output headers, registered aliases, generated-fact metadata, and required
dictionary boundary phrases. It does not semantically validate every
definition in `DATA_DICTIONARY.md`.

## Supporting Validation Map

| Check | What it verifies | Current role |
|---|---|---|
| Selected implemented-field contract check | Required implemented fields, declared headers, aliases, fact metadata, and documented boundary phrases remain aligned. | Protects the declared static contract scope; it is not a complete semantic parser of the dictionary. |
| Value-lineage validation | Source values, derived outputs, period metadata, formulas, and nested evidence agree at field level. | Makes Demo 1 and Demo 2 auditable from source rows to generated facts. |
| Answer and endpoint boundary checks | Responses preserve entity, period, source, metric-definition, and comparison limits. | Prevents retrieval matches from becoming unsupported operating conclusions. |
| Retrieval wording stress | Supported evidence remains retrievable while mismatch, ambiguity, and hard-negative behavior remains visible. | Separates semantic proximity from answerability. |
| RAC grounded-review checks | Evidence packets, source paths, limitations, report sections, and coverage inputs remain inspectable. | Keeps route resolution separate from evidence strength and decision quality. |
| Future comparability-gate scope validation | Checks the documented pairwise decision contract against the implemented repository scope. | Keeps specification validation separate from the current data experiments. |

## How to Read the Experiments

The experiments should be read as one staged evidence path rather than as
separate engineering features:

```text
selected Meituan backend metrics
-> canonical field dictionary
-> SQL diagnostic outputs
-> generated retail memory facts
-> answer-boundary checks
-> retrieval wording-variation stress test
-> repeated-window evidence review
-> RAC grounded review over local evidence
```

The early checks protect selected required implemented field names, declared headers, required boundary phrases, source paths, and
generated-fact structure. Demo 1 and Demo 2 then test whether selected
store-period evidence can be turned into diagnostic outputs without changing
backend metric definitions.

The answer-boundary, endpoint-boundary, retrieval, and robustness checks test
whether later answers preserve entity, period, source, metric-definition, and
comparison limits.

The repeated-window B-F panel extends the evidence base across February,
March, and April 2026. It verifies repeated store-period coverage and reports descriptive movement in selected metrics. Monthly guardrail stability is a separate future test because the current panel does not contain repeated top-SKU evidence or monthly recomputed `comparison_limit_notes`.

RAC applies factor expansion, evidence routing, critique, rule-based checks
for unsupported claims and definition conflicts, review-state updates,
limitations, source paths, and local evidence snippets to the structured
evidence produced by the earlier stages.

## Experiment 1: Store A Month-over-Month Diagnostic

| Item | Content |
|---|---|
| Question | Can selected Meituan backend metrics for one store be organized into a month-over-month diagnostic without changing backend metric meanings or reducing the result to one metric? |
| Input | Store A February, March, and April 2026 store-period metrics; Store A top-SKU evidence. |
| Transformation | `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql` derives month-over-month movement, ranking changes, traffic and conversion tradeoffs, and top-SKU concentration evidence. |
| Output | `retail_ops/outputs/store_a_demo1_sql_output.csv`; `retail_ops/outputs/generated_retail_memory_facts.json`. |
| Expected behavior | The output may describe observed month-over-month movement, but it should not attribute performance change to one metric alone. |
| Current result | The diagnostic output is generated. The data contract passes, and the value-lineage check passes across 3 source rows, 3 SQL output rows, 9 top-SKU rows, 180 source, formula, month-over-month, rank-change, and tradeoff comparisons, and 5 generated memory facts. |
| Checked by | `python3 retail_ops/scripts/validate_retail_data_contract.py`; `python3 retail_ops/scripts/validate_demo1_value_lineage.py` |
| Result path | `retail_ops/outputs/demo1_value_lineage_validation_result.txt` |

## Experiment 2: Demo 2 Same-Period Store Diagnostic

| Item | Content |
|---|---|
| Question | Can selected B-F store-period rows be placed under one March 2026 reporting window and one field contract before any stronger comparison is attempted? |
| Input | `retail_ops/data/demo2_store_period_metrics.csv`; top search-term evidence; top-SKU transaction-amount evidence. |
| Transformation | `retail_ops/sql/02_demo2_cross_store_comparability.sql` derives search-entry share/rate, activity-order share, top-3 SKU concentration, `comparison_scope_flag`, and `comparison_limit_notes`. |
| Output | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Expected behavior | The SQL output should include `comparison_scope_flag` and `comparison_limit_notes`, while staying at row-level same-period diagnostic scope. `same_period_diagnostic_ready` should certify only the fixed March window and the documented core readiness fields, not every output column or pairwise comparability. |
| Current result | Passed the current Demo 2 output validation and scope-boundary validation. The output validator recomputes the derived ratios and checks the narrow flag contract against the documented core fields. |
| Checked by | `python3 retail_ops/scripts/validate_demo2_comparability_output.py`; `python3 eval/eval_retail_demo2_scope_boundary.py` |
| Result path | `eval/retail_decision_support_eval_results/eval_retail_demo2_scope_boundary_result.txt` |
| Failure mode | Ranking stores globally, treating same-period diagnostic readiness as pairwise comparability, treating `activity_cost_ratio_pct` as ROI, or transferring a promotion, price, or SKU action without checking limits. |

## Experiment 2A: Demo 2 Guardrail Sensitivity Check

| Item | Content |
|---|---|
| Question | How do the current Demo 2 `comparison_limit_notes` change when the implemented thresholds are shifted by 5 percentage points in either direction? |
| Input | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`. |
| Transformation | `retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py` reads the implemented thresholds from the current SQL, reproduces every current-row `comparison_limit_notes` value, then lowers and raises those thresholds by 5 percentage points. |
| Output | `retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv`; `retail_ops/outputs/demo2_guardrail_sensitivity_result.txt`. |
| Expected behavior | The check should not optimize thresholds or turn them into peer-selection rules. It should only show whether the current threshold-based guardrail notes are sensitive to small threshold shifts. |
| Current result | Completed current guardrail sensitivity inspection. Baseline reproduction passes for all five rows. Four rows (C, D, E, and F) change under the harder-to-trigger +5 percentage-point scenario; the easier-to-trigger -5 percentage-point scenario produces no note changes, and Store B remains unchanged. |
| Interpretation | In this five-row fixture, the notes show one-sided sensitivity to stricter thresholds. The thresholds remain prototype diagnostic warnings rather than validated peer-comparison rules. |
| Checked by | `python3 retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py` |

## Experiment 3: Demo 2 Memory-Fact Generation

| Item | Content |
|---|---|
| Question | Can the Demo 2 diagnostic output be converted into retrieval-facing memory facts without losing source fields, observed values, source paths, or limitation notes? |
| Input | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/data/demo2_top_search_terms.csv`; `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`. |
| Transformation | `retail_ops/scripts/generate_demo2_retail_memory_facts.py` converts row-level diagnostics into slot-based retail memory facts. |
| Output | `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Expected behavior | Generated facts should preserve store entity, period, slot, observed values, `calculation` metadata, source fields, primary source path, supporting source paths, lineage path, confidence, limitations, active status, and period granularity. |
| Current result | Passed 25/25 entity-slot contracts across five stores and five implemented slots. The fact-contract evaluation checks exact coverage, required metadata, repository-backed paths, and slot-specific boundary terms. A separate value-lineage check passes across 5 source rows, 5 diagnostic output rows, 250 source-to-output and derived-value comparisons, and 320 fact metadata and observed-value comparisons, including nested top-search and top-SKU evidence. |
| Checked by | `python3 eval/eval_retail_demo2_facts.py`; `python3 retail_ops/scripts/validate_demo2_value_lineage.py`; `python3 retail_ops/scripts/validate_retail_data_contract.py` |
| Result paths | `eval/retail_decision_support_eval_results/eval_retail_demo2_facts_result.txt`; `retail_ops/outputs/demo2_value_lineage_validation_result.txt` |
| Failure mode | Mixing store-level and SKU-level fields, dropping source evidence, introducing undocumented fields, or letting top-search or top-SKU evidence appear without supporting source paths. |

Implemented Demo 2 evidence slots:

| Slot | Review role |
|---|---|
| `visibility_entry_profile` | Exposure, ranking, entry, and search-entry structure. |
| `activity_lever_profile` | Activity orders, activity cost, subsidy, and activity-cost ratio. |
| `transaction_conversion_profile` | Transaction scale, order conversion, payment, and average order value. |
| `top3_sku_product_mix_note` | Lightweight top-SKU concentration evidence from selected ranking views. |
| `single_metric_attribution_guard` | Guardrail against explaining performance from one metric alone. |

## Experiment 4: Demo 2 Answer-Contract Fixture Validation

| Item | Content |
|---|---|
| Question | Do six manually specified answer contracts preserve the documented Demo 2 metric and comparison boundaries? |
| Input | Six fixed answer-contract fixtures; Demo 2 SQL output and generated facts used for setup checks. |
| Transformation | `eval/eval_retail_demo2_answer_behavior.py` checks each fixed contract text for required and forbidden terms. It does not call `/chat_retail_ops_demo2_kb` and does not generate answers. |
| Output | `eval/retail_decision_support_eval_results/eval_retail_demo2_answer_behavior_result.txt`. |
| Expected behavior | Each fixture preserves the relevant metric definition and scope limit for activity-cost ratio, top-SKU evidence, search-entry comparison, strategy transfer, same-period readiness, or `region_type`. |
| Current result | Six of six manually specified answer contracts passed the current required/forbidden-term checks. |
| Checked by | `python3 eval/eval_retail_demo2_answer_behavior.py` |
| Result path | `eval/retail_decision_support_eval_results/eval_retail_demo2_answer_behavior_result.txt` |
| Failure mode | A contract omits required metric or scope language, or includes a forbidden ranking, causal, profit, or direct strategy-transfer claim. |

### Answer-contract rules checked here

| Boundary | Rule |
|---|---|
| Activity cost ratio | `activity_cost_ratio_pct` is not traditional ROI or profit margin. |
| Top-SKU share | `top3_sku_transaction_amount_share_pct` describes the listed top-SKU evidence rather than full product-category sales share. |
| Search-entry evidence | Search-entry evidence is one visibility-to-entry signal within the current store-period profile. |
| Activity evidence | Activity evidence describes operating-tool usage; it does not establish automatic promotion-transfer logic. |
| Same-period readiness | `same_period_diagnostic_ready` is not a finished pairwise comparability decision. |
| Region context | `region_type` is weak context only. |

## Experiment 4A: Demo 2 Endpoint-Boundary Contract Check

| Item | Content |
|---|---|
| Question | Does the implemented `/chat_retail_ops_demo2_kb` endpoint preserve the same evidence boundaries when answering from file-backed Demo 2 memory facts? |
| Input | `/chat_retail_ops_demo2_kb`; `api/main.py`; `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Transformation | Endpoint-level evaluation checks supported Store B-F questions, questions without a matched evidence slot, unsupported all-48-store questions, best-store ranking requests, final operating-recommendation requests, and out-of-Demo-2 entity questions. |
| Output | `eval/retail_decision_support_eval_results/eval_retail_demo2_endpoint_behavior_result.txt`. |
| Expected behavior | Store B-F questions with a matched evidence slot return file-backed facts; recognized cross-store questions stay at same-period diagnostic scope; questions without either match return `supported: false`. |
| Current result | The current repository-defined endpoint scenarios pass, including the unmatched evidence-class case that returns `supported: false`. |
| Checked by | `python3 eval/eval_retail_demo2_endpoint_behavior.py` |

## Experiment 5: Retrieval Threshold Inspection

| Item | Content |
|---|---|
| Question | What score distribution does the current file-backed corpus produce, and what exploratory reference threshold can be inspected from it? |
| Input | `eval/retrieval_threshold_cases.json`; generated Demo 1 and Demo 2 retail memory facts; selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`. |
| Transformation | `eval/analyze_retail_embedding_score_distribution.py` embeds each query and retrieval document with local Ollama `bge-m3`, retrieves top-k evidence, records scores, top-1/top-2 margins, expected matches, entity matches, slot matches, and period-scope checks. |
| Output | `retail_ops/outputs/retrieval_score_distribution.csv`; `retail_ops/outputs/retrieval_threshold_summary.md`; `retail_ops/outputs/retrieval_score_distribution.png`. |
| Expected behavior | The inspection records retrieval behavior across supported, unsupported, hard-negative, entity/period mismatch, and ambiguous comparison queries. |
| Current result | The applicable run is recorded in `retail_ops/outputs/retrieval_threshold_summary.md`, including corpus metadata, per-group score distributions, expected hit@5, top-1/top-2 margins, and the derived exploratory reference threshold. |
| Boundary | Offline reference-threshold inspection only. Retrieval scores make evidence routing inspectable, but final operating claims still require entity, period, source-path, metric-definition, and interpretation-boundary checks. |
| Failure mode | Treating a high retrieval score as sufficient evidence for an operating conclusion, or claiming a production-level threshold from the current small file-backed corpus. |

## Experiment 6: Query Wording-Variation Stress Test

| Item | Content |
|---|---|
| Question | How does retrieval behavior change when the same query intent is expressed with deterministic wording variations? |
| Input | Current retail retrieval-inspection cases and wording variants. |
| Transformation | `eval/analyze_retail_query_robustness.py` tests shortened, paraphrased, typo/noise, and keyword-order query variants. |
| Output | `retail_ops/outputs/retrieval_query_robustness.csv`; `retail_ops/outputs/retrieval_query_threshold_sweep.csv`; `retail_ops/outputs/retrieval_query_robustness_summary.md`. |
| Expected behavior | Supported cases should generally retain expected evidence in top-k under small wording changes. Unsupported, hard-negative, entity/period-mismatch, and ambiguous comparison cases should still require entity, period, slot, source-path, and interpretation-boundary checks. |
| Current result | The applicable run is recorded in `retail_ops/outputs/retrieval_query_robustness_summary.md`, including per-case-type expected-hit retention, reference-threshold crossings, top-1 changes, and the threshold sweep. |
| Boundary | Top-k retention or a score above the reference threshold does not make a query answerable; entity, period, slot, source-path, and answer-boundary checks remain required. |
| Boundary example | The generated summary reports reference-threshold crossings and expected-hit retention separately for entity/period-mismatch variants, keeping semantic proximity distinct from evidence applicability. |

## Experiment 7: Repeated-Window Coverage and Descriptive Movement Check

| Item | Content |
|---|---|
| Question | Do Stores B-F have complete February-April 2026 coverage, and what descriptive movement is visible in selected store-period metrics? |
| Input | B-F store-period records across 2026-02, 2026-03, and 2026-04. |
| Transformation | `retail_ops/sql/03_store_period_panel_coverage.sql` checks observed-month coverage and selected averages. `retail_ops/sql/04_repeated_window_panel_summary.sql` places February, March, and April values side by side for the selected metrics and retains February-to-April endpoint deltas and relative changes. |
| Output | `retail_ops/outputs/store_period_panel_coverage_output.csv`; `retail_ops/outputs/repeated_window_panel_summary_output.csv`. |
| Expected behavior | Confirm repeated-window coverage and report descriptive movement without converting short-window changes into store rankings, causal explanations, pairwise comparability decisions, or guardrail-stability claims. |
| Current result | Stores B-F each contain three observed months. The summary exposes all February, March, and April values for 11 selected transaction, traffic, conversion, search-entry, and activity-cost-ratio metrics. The committed CSV matches a fresh execution of the summary SQL, and the February-to-April endpoint fields remain reproducible. |
| Current use | Three-month descriptive review and evidence preparation for future question-specific comparison rules. The middle month is retained so endpoint changes are not read as the full observed path. |
| Checked by | `python3 retail_ops/scripts/validate_store_period_panel.py`; `python3 retail_ops/scripts/validate_repeated_window_panel_summary.py` |
| Failure mode | Treating the three-month descriptive panel as proof that March guardrail notes remain stable over time. |

Evidence required before a monthly guardrail-stability test:

| Missing evidence | Why it matters |
|---|---|
| Repeated top-SKU evidence for each store-month | The current March guardrail logic includes top-3 SKU concentration, so monthly notes cannot be reproduced without monthly SKU evidence. |
| Monthly recomputation of `comparison_limit_notes` under the implemented SQL contract | Stability must compare like-for-like monthly note sets rather than infer stability from unrelated store-level metrics. |
| Broader repeated store-period records beyond the current B-F three-month panel | More stores and months are needed before treating a local sensitivity pattern as reusable. |
| Activity calendar or campaign-condition evidence | Activity involvement should not be interpreted as full campaign status without operating context. |
| Local competition, price-pressure, fulfillment, or stockout evidence | Without these factors, the current results do not support promotion, pricing, or strategy-transfer conclusions. |

## Contract Validation: Future Gate Scope

| Item | Content |
|---|---|
| Question | Does the documented future pairwise comparability gate remain separated from the implemented Demo 2 scope? |
| Input | `retail_ops/COMPARABILITY_GATE_V0.md`. |
| Transformation | `eval/eval_future_comparability_gate_contract.py` checks that the planned input triple, output enum, and blocking-factor list are documented. |
| Output | `eval/retail_decision_support_eval_results/eval_future_comparability_gate_contract_result.txt`. |
| Expected behavior | The future gate may define planned factors such as transaction order volume, transaction amount, activity involvement, activity intensity, store type, region and market context, SKU structure, and repeated reporting windows. Current Demo 2 outputs remain within row-level same-period diagnostic scope. |
| Current result | The planned contract is documented, and the repository-defined future-gate scope validation passes. |
| Checked by | `python3 eval/eval_future_comparability_gate_contract.py` |
| Failure mode | Treating Demo 2 row-level diagnostic readiness as a pairwise comparability decision. |

## Method Notes: Demo 2 Guardrails

The Demo 2 thresholds are lightweight interpretation guardrails, not optimized business cutoffs.

The current sensitivity check exists because a future pairwise comparability gate should not inherit one-off threshold choices without repeated-window evidence.

| Item | Meaning |
|---|---|
| SQL baseline | Current SQL output values and current `comparison_limit_notes`. |
| Easier-to-trigger scenario | Recomputed notes after lowering each implemented threshold by 5 percentage points. |
| Harder-to-trigger scenario | Recomputed notes after raising each implemented threshold by 5 percentage points. |
| Threshold-sensitive row | A store row whose note set changes under at least one threshold scenario. |

Current result: four of five B-F rows are threshold-sensitive under the harder-to-trigger scenario, while Store B remains unchanged and the easier-to-trigger scenario changes no rows. This small-sample result supports treating the guardrails as diagnostic warnings rather than optimized pairwise-comparison rules.
