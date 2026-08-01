# Grounded RAC Report

## 1. Direct Answer

Store A's April performance should not be attributed to search exposure alone.

Deterministic local-file review; routing scores summarize evidence coverage under the current rules.

## 2. Question Type

- Question type: causal_diagnostic
- Domain: retail_operations

## 3. Factor Weights

### 3a. How Factor Weights Are Generated

Factor weights are fixed review-priority buckets assigned by explicit rules in `rac/src/mock_pipeline.py`. They order review attention within the current evidence scope.

| Bucket | Weight | Rule | Factors in This Report |
|---|---:|---|---|
| high | 0.85 | Central to avoiding overconfident or misleading conclusions. | order_conversion, promotion_intensity |
| medium | 0.72 | Important context but not sufficient on its own. | search_exposure, entry_conversion, transaction_orders |
| default | 0.60 | Potentially relevant but requires stronger evidence. | none |

Weighting boundary:

- Use these values only to order review attention within the current evidence scope.

### 3b. Factor Weights Used in This Report

| Decision Factor ID | Weight | Bucket | Evidence Status | Why It Matters |
|---|---:|---|---|---|
| search_exposure | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| entry_conversion | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| order_conversion | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| promotion_intensity | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| transaction_orders | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |

## 4. Local Evidence Grounding

- Total evidence packets: 5
- Record matched packets: 5
- Keyword matched packets: 0
- Boundary matched packets: 0
- Fallback packets: 0
- Missing source files: 0

For CSV evidence, `Source Locator` shows the selected record scope and `Selected Values` shows values read from those records. For Markdown evidence, the locator remains a local line-range pointer.
`Decision Factor ID` is an internal RAC review identifier. The field column shows canonical project fields where available and labels unresolved requirements explicitly.

| Decision Factor ID | Source | Evidence Type | Status | Source Locator | Canonical Evidence Fields / Requirement | Selected Values |
|---|---|---|---|---|---|---|
| search_exposure | retail_ops/data/store_a_monthly_metrics.csv | quantitative_evidence | record_matched | records: store_id=A; period_month=2026-03, 2026-04; rows=2 | search_exposure_users, search_average_rank, search_entry_users | store_id=A, period_month=2026-03: search_exposure_users=4172, search_average_rank=20, search_entry_users=445<br>store_id=A, period_month=2026-04: search_exposure_users=7736, search_average_rank=18, search_entry_users=839 |
| entry_conversion | retail_ops/data/store_a_monthly_metrics.csv | quantitative_evidence | record_matched | records: store_id=A; period_month=2026-03, 2026-04; rows=2 | exposure_users, entry_users, entry_conversion_rate_pct | store_id=A, period_month=2026-03: exposure_users=4663, entry_users=522, entry_conversion_rate_pct=11.19<br>store_id=A, period_month=2026-04: exposure_users=8366, entry_users=906, entry_conversion_rate_pct=10.83 |
| order_conversion | retail_ops/data/store_a_monthly_metrics.csv | quantitative_evidence | record_matched | records: store_id=A; period_month=2026-03, 2026-04; rows=2 | entry_users, order_users, order_conversion_rate_pct | store_id=A, period_month=2026-03: entry_users=522, order_users=221, order_conversion_rate_pct=42.34<br>store_id=A, period_month=2026-04: entry_users=906, order_users=339, order_conversion_rate_pct=37.42 |
| promotion_intensity | retail_ops/data/store_a_monthly_metrics.csv | quantitative_evidence | record_matched | records: store_id=A; period_month=2026-03, 2026-04; rows=2 | activity_original_transaction_amount, activity_orders, activity_cost, merchant_subsidy_amount, platform_subsidy_amount, activity_cost_ratio_pct | store_id=A, period_month=2026-03: activity_original_transaction_amount=10035.00, activity_orders=201, activity_cost=3868.16, merchant_subsidy_amount=3727.16, platform_subsidy_amount=141.00, activity_cost_ratio_pct=38.55<br>store_id=A, period_month=2026-04: activity_original_transaction_amount=15006.90, activity_orders=329, activity_cost=6105.86, merchant_subsidy_amount=5947.36, platform_subsidy_amount=158.50, activity_cost_ratio_pct=40.69 |
| transaction_orders | retail_ops/data/store_a_monthly_metrics.csv | quantitative_evidence | record_matched | records: store_id=A; period_month=2026-03, 2026-04; rows=2 | transaction_orders | store_id=A, period_month=2026-03: transaction_orders=207<br>store_id=A, period_month=2026-04: transaction_orders=337 |


## 5. Competing Hypotheses

The `Scenario-Template Confidence` column records deterministic review labels assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`.

| Hypothesis | Scenario-Template Confidence | Status | Weakness |
|---|---:|---|---|
| Search exposure may have contributed to Store A's April performance, but it is not sufficient as a single explanation. | 0.52 | plausible | Does not isolate promotion effects; Does not prove source-field improvement. |
| Store A's April performance should be reviewed as a multi-factor movement across search exposure, entry conversion, order conversion, promotion intensity, and transaction orders. | 0.74 | strong | Observational evidence cannot establish strict causality. |
| The available evidence is insufficient for single-cause attribution. | 0.82 | strong | Conservative rather than complete causal explanation. |

## 6. Critic Findings

- [high] Observational evidence supports bounded association claims only. Recommendation: Keep attribution language conditional and record unresolved alternatives.
- [medium] Current evidence scope is limited to committed local project files. Recommendation: Keep source paths and unresolved external requirements explicit.

## 7. Claim and Definition Check

- Status: pass
- Unsupported claims detected by current rules: none
- Definition conflicts detected by current rules: none

## 8. Final Judgment

Store A's April performance should not be attributed to search exposure alone.

The judgment is bounded by the cited local evidence and the unresolved requirements recorded above.

## 9. Evidence-Coverage Score

1.00

How this score is calculated:

```text
evidence_coverage_score =
  0.45 * direct_evidence_rate
+ 0.25 * supported_or_boundary_rate
+ 0.15 * no_missing_source_file_score
+ 0.15 * no_fallback_score
```

Weight rationale:

| Component | Weight | Why |
|---|---:|---|
| `direct_evidence_rate` | 0.45 | Highest priority because actual local evidence should matter more than boundary-only evidence. |
| `supported_or_boundary_rate` | 0.25 | Boundary evidence is valuable because it explicitly records missing requirements instead of hiding them. |
| `no_missing_source_file_score` | 0.15 | Source files must exist, but this is a basic traceability check rather than evidence strength. |
| `no_fallback_score` | 0.15 | Fallback packets indicate unresolved routing and reduce the current coverage score. |

Score contract:

- Component weights are fixed prototype heuristics.
- The score summarizes evidence-routing coverage under the current rules.
- Alternative weights are a formula sensitivity check; the report judgment is produced separately.

Current report inputs:

- total_packets = 5
- record_matched_packets = 5
- keyword_matched_packets = 0
- boundary_matched_packets = 0
- fallback_packets = 0
- missing_source_files = 0
- direct_evidence_rate = (record_matched_packets + keyword_matched_packets) / total_packets = 1.00
- supported_or_boundary_rate = (record_matched_packets + keyword_matched_packets + boundary_matched_packets) / total_packets = 1.00
- no_missing_source_file_score = 1.00
- no_fallback_score = 1.00

Reading the score:

- A higher value means more requested evidence routes were resolved or explicitly bounded.
- Boundary evidence contributes when it documents a missing requirement.
- Read the score as coverage rather than evidence strength, causal validity, decision quality, or business impact.

## 10. What Cannot Be Concluded

- No randomized experiment.
- Evidence scope is limited to committed local project files.
- No complete competitor-side evidence.

## 11. Review-State Update

- review_state_id: store_a_april_growth_not_search_only
- status: active
- validity_conditions:
  - Store A Demo 1 context.
  - Available month-over-month evidence only.
