# Grounded RAC Report

## 1. Direct Answer

Stores B-F can be organized for same-period diagnostic review, but should not be treated as directly comparable without pairwise gates.

Deterministic local-file review; routing scores summarize route resolution under the current rules.

## 2. Question Type

- Question type: comparability_judgment
- Domain: retail_operations

## 3. Factor Weights

### 3a. How Factor Weights Are Generated

Factor weights are fixed review-priority buckets assigned by explicit rules in `rac/src/mock_pipeline.py`. They order review attention within the current evidence scope.

| Bucket | Weight | Rule | Factors in This Report |
|---|---:|---|---|
| high | 0.85 | Central to avoiding overconfident or misleading conclusions. | activity_intensity |
| medium | 0.72 | Important context but not sufficient on its own. | same_reporting_period, store_type, order_volume, transaction_amount |
| default | 0.60 | Potentially relevant but requires stronger evidence. | region_context, competition, sku_structure, repeated_reporting_windows |

Weighting boundary:

- Use these values only to order review attention within the current evidence scope.

### 3b. Factor Weights Used in This Report

| Decision Factor ID | Weight | Bucket | Evidence Status | Why It Matters |
|---|---:|---|---|---|
| same_reporting_period | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| store_type | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| order_volume | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| transaction_amount | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| activity_intensity | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| region_context | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| competition | 0.60 | default | missing | Potentially relevant but requires stronger evidence. |
| sku_structure | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| repeated_reporting_windows | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |

`partially_supported` indicates that a registered local evidence route was resolved for the factor. It does not necessarily mean that observed numeric evidence supports the decision.

## 4. Local Evidence Grounding

- Total evidence packets: 9
- Record matched packets: 8
- Keyword matched packets: 0
- Boundary matched packets: 1
- Fallback packets: 0
- Missing source files: 0

For CSV evidence, `Source Locator` shows the selected record scope and `Selected Values` shows values read from those records. For Markdown evidence, the locator remains a local line-range pointer.
`Decision Factor ID` is an internal RAC review identifier. The field column shows canonical project fields where available and labels unresolved requirements explicitly.

| Decision Factor ID | Source | Evidence Type | Status | Source Locator | Canonical Evidence Fields / Requirement | Selected Values |
|---|---|---|---|---|---|---|
| same_reporting_period | retail_ops/outputs/demo2_cross_store_comparability_output.csv | context_evidence | record_matched | records: store_id=B, C, D, E, F; period_month=2026-03; rows=5 | period_start, period_end, period_month | store_id=B, period_month=2026-03: period_start=2026-03-01, period_end=2026-03-31, period_month=2026-03<br>store_id=C, period_month=2026-03: period_start=2026-03-01, period_end=2026-03-31, period_month=2026-03<br>store_id=D, period_month=2026-03: period_start=2026-03-01, period_end=2026-03-31, period_month=2026-03<br>store_id=E, period_month=2026-03: period_start=2026-03-01, period_end=2026-03-31, period_month=2026-03<br>store_id=F, period_month=2026-03: period_start=2026-03-01, period_end=2026-03-31, period_month=2026-03 |
| store_type | retail_ops/outputs/demo2_cross_store_comparability_output.csv | context_evidence | record_matched | records: store_id=B, C, D, E, F; period_month=2026-03; rows=5 | store_type | store_id=B, period_month=2026-03: store_type=self-operated<br>store_id=C, period_month=2026-03: store_type=self-operated<br>store_id=D, period_month=2026-03: store_type=self-operated<br>store_id=E, period_month=2026-03: store_type=partner<br>store_id=F, period_month=2026-03: store_type=partner |
| order_volume | retail_ops/outputs/demo2_cross_store_comparability_output.csv | quantitative_evidence | record_matched | records: store_id=B, C, D, E, F; period_month=2026-03; rows=5 | transaction_orders | store_id=B, period_month=2026-03: transaction_orders=299<br>store_id=C, period_month=2026-03: transaction_orders=175<br>store_id=D, period_month=2026-03: transaction_orders=404<br>store_id=E, period_month=2026-03: transaction_orders=158<br>store_id=F, period_month=2026-03: transaction_orders=266 |
| transaction_amount | retail_ops/outputs/demo2_cross_store_comparability_output.csv | quantitative_evidence | record_matched | records: store_id=B, C, D, E, F; period_month=2026-03; rows=5 | transaction_amount | store_id=B, period_month=2026-03: transaction_amount=11665.5<br>store_id=C, period_month=2026-03: transaction_amount=7064.09<br>store_id=D, period_month=2026-03: transaction_amount=18078.7<br>store_id=E, period_month=2026-03: transaction_amount=5784.87<br>store_id=F, period_month=2026-03: transaction_amount=9301.8 |
| activity_intensity | retail_ops/outputs/demo2_cross_store_comparability_output.csv | quantitative_evidence | record_matched | records: store_id=B, C, D, E, F; period_month=2026-03; rows=5 | activity_orders, activity_order_share_pct, activity_cost, activity_cost_ratio_pct | store_id=B, period_month=2026-03: activity_orders=265, activity_order_share_pct=88.63, activity_cost=3361.3, activity_cost_ratio_pct=24.12<br>store_id=C, period_month=2026-03: activity_orders=124, activity_order_share_pct=70.86, activity_cost=490.21, activity_cost_ratio_pct=9.45<br>store_id=D, period_month=2026-03: activity_orders=337, activity_order_share_pct=83.42, activity_cost=2776.4, activity_cost_ratio_pct=14.92<br>store_id=E, period_month=2026-03: activity_orders=109, activity_order_share_pct=68.99, activity_cost=1576.26, activity_cost_ratio_pct=29.38<br>store_id=F, period_month=2026-03: activity_orders=217, activity_order_share_pct=81.58, activity_cost=1008.3, activity_cost_ratio_pct=12.16 |
| region_context | retail_ops/outputs/demo2_cross_store_comparability_output.csv | context_evidence | record_matched | records: store_id=B, C, D, E, F; period_month=2026-03; rows=5 | region_type | store_id=B, period_month=2026-03: region_type=Qingdao<br>store_id=C, period_month=2026-03: region_type=Qingdao<br>store_id=D, period_month=2026-03: region_type=Yantai<br>store_id=E, period_month=2026-03: region_type=Yantai<br>store_id=F, period_month=2026-03: region_type=Yantai |
| competition | retail_ops/COMPARABILITY_GATE_V0.md | boundary_evidence | boundary_matched | lines 101-103 | competition-context requirement in comparability contract | n/a |
| sku_structure | retail_ops/outputs/demo2_cross_store_comparability_output.csv | product_mix_evidence | record_matched | records: store_id=B, C, D, E, F; period_month=2026-03; rows=5 | top3_sku_transaction_amount, top3_sku_transaction_amount_share_pct | store_id=B, period_month=2026-03: top3_sku_transaction_amount=1300.9, top3_sku_transaction_amount_share_pct=11.15<br>store_id=C, period_month=2026-03: top3_sku_transaction_amount=2004.84, top3_sku_transaction_amount_share_pct=28.38<br>store_id=D, period_month=2026-03: top3_sku_transaction_amount=3055.78, top3_sku_transaction_amount_share_pct=16.9<br>store_id=E, period_month=2026-03: top3_sku_transaction_amount=726.25, top3_sku_transaction_amount_share_pct=12.55<br>store_id=F, period_month=2026-03: top3_sku_transaction_amount=1798.4, top3_sku_transaction_amount_share_pct=19.33 |
| repeated_reporting_windows | retail_ops/outputs/repeated_window_panel_summary_output.csv | quantitative_evidence | record_matched | records: store_id=B, C, D, E, F; rows=5 | observed_month_count, feb_transaction_amount, mar_transaction_amount, apr_transaction_amount, feb_transaction_orders, mar_transaction_orders, apr_transaction_orders | store_id=B: observed_month_count=3, feb_transaction_amount=10468.0, mar_transaction_amount=11665.5, apr_transaction_amount=11496.8, feb_transaction_orders=259.0, mar_transaction_orders=299.0, apr_transaction_orders=293.0<br>store_id=C: observed_month_count=3, feb_transaction_amount=9503.7, mar_transaction_amount=7064.09, apr_transaction_amount=6756.8, feb_transaction_orders=253.0, mar_transaction_orders=175.0, apr_transaction_orders=178.0<br>store_id=D: observed_month_count=3, feb_transaction_amount=20332.2, mar_transaction_amount=18078.7, apr_transaction_amount=14087.2, feb_transaction_orders=466.0, mar_transaction_orders=404.0, apr_transaction_orders=308.0<br>store_id=E: observed_month_count=3, feb_transaction_amount=6794.9, mar_transaction_amount=5784.87, apr_transaction_amount=11264.72, feb_transaction_orders=148.0, mar_transaction_orders=158.0, apr_transaction_orders=377.0<br>store_id=F: observed_month_count=3, feb_transaction_amount=12549.1, mar_transaction_amount=9301.8, apr_transaction_amount=14090.7, feb_transaction_orders=307.0, mar_transaction_orders=266.0, apr_transaction_orders=424.0 |


## 5. Competing Hypotheses

The `Scenario-Template Confidence` column records deterministic review labels assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`.

| Hypothesis | Scenario-Template Confidence | Status | Weakness |
|---|---:|---|---|
| Stores B-F can be organized in a same-period diagnostic table. | 0.78 | strong | Same-period diagnostic organization does not establish robust comparability. |
| Stores B-F should not be treated as directly comparable without pairwise gates. | 0.86 | strong | Pairwise quantitative thresholds are outside the current review contract. |

## 6. Critic Findings

- [medium] Current evidence scope is limited to committed local project files. Recommendation: Keep source paths and unresolved external requirements explicit.
- [critical] Same-period diagnostic organization must not be described as a completed pairwise comparability gate. Recommendation: Separate same-period diagnostic review from pairwise comparability.

## 7. Claim and Definition Check

- Status: pass
- Unsupported claims detected by current rules: none
- Definition conflicts detected by current rules: none

## 8. Final Judgment

Stores B-F can be organized for same-period diagnostic review, but should not be treated as directly comparable without pairwise gates.

The judgment is bounded by the cited local evidence and the unresolved requirements recorded above.

## 9. Evidence-Routing Coverage

Packet composition:

- Total packets: 9
- Record matched packets: 8
- Keyword matched packets: 0
- Boundary matched packets: 1
- Fallback packets: 0
- Missing source files: 0

- Routing coverage score: 0.95
- Read this value as route resolution under the current rules, not as evidence strength or decision quality.

How this score is calculated:

```text
routing_coverage_score =
  0.45 * record_or_keyword_route_rate
+ 0.25 * resolved_or_boundary_route_rate
+ 0.15 * no_missing_source_file_score
+ 0.15 * no_fallback_score
```

Weight rationale:

| Component | Weight | Why |
|---|---:|---|
| `record_or_keyword_route_rate` | 0.45 | Highest priority because record- or keyword-matched local routes should matter more than boundary-only evidence. |
| `resolved_or_boundary_route_rate` | 0.25 | Boundary evidence is valuable because it explicitly records missing requirements instead of hiding them. |
| `no_missing_source_file_score` | 0.15 | Source files must exist, but this is a basic traceability check rather than evidence strength. |
| `no_fallback_score` | 0.15 | Fallback packets indicate unresolved routing and reduce the current routing score. |

Score contract:

- Component weights are fixed prototype heuristics.
- The score summarizes route resolution under the current rules.
- Alternative weights are a formula sensitivity check; the report judgment is produced separately.

Score inputs (contract fields):

- total_packets = 9
- record_matched_packets = 8
- keyword_matched_packets = 0
- boundary_matched_packets = 1
- fallback_packets = 0
- missing_source_files = 0

Derived rates and checks:

- record_or_keyword_route_rate = (record_matched_packets + keyword_matched_packets) / total_packets = 0.89
- resolved_or_boundary_route_rate = (record_matched_packets + keyword_matched_packets + boundary_matched_packets) / total_packets = 1.00
- no_missing_source_file_score = 1.00
- no_fallback_score = 1.00

Reading the score:

- A higher value means more requested evidence routes were resolved or explicitly bounded.
- Boundary evidence contributes when it documents a missing requirement.
- Read the score as coverage rather than evidence strength, causal validity, decision quality, or business impact.

## 10. What Cannot Be Concluded

- Pairwise quantitative gates are not defined in the current contract.
- Region type remains weak context.
- Three monthly B-F reporting windows are available; they do not by themselves establish stable pairwise comparability.

## 11. Review-State Update

- review_state_id: stores_b_f_same_period_not_directly_comparable
- status: active
- validity_conditions:
  - Demo 2 March 2026 B-F context.
