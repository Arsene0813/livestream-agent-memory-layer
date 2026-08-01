# Grounded RAC Report

## 1. Direct Answer

Stores B-F can be organized for same-period diagnostic review, but should not be treated as directly comparable without pairwise gates.

Deterministic local-file review; routing scores summarize evidence coverage under the current rules.

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

## 4. Local Evidence Grounding

- Total evidence packets: 9
- Record matched packets: 0
- Keyword matched packets: 8
- Boundary matched packets: 1
- Fallback packets: 0
- Missing source files: 0

For CSV evidence, `Source Locator` shows the selected record scope and `Selected Values` shows values read from those records. For Markdown evidence, the locator remains a local line-range pointer.
`Decision Factor ID` is an internal RAC review identifier. The field column shows canonical project fields where available and labels unresolved requirements explicitly.

| Decision Factor ID | Source | Evidence Type | Status | Source Locator | Canonical Evidence Fields / Requirement | Selected Values |
|---|---|---|---|---|---|---|
| same_reporting_period | retail_ops/data/demo2_source_notes.md | context_evidence | keyword_matched | lines 6-8 | period_start, period_end, period_month | n/a |
| store_type | retail_ops/outputs/demo2_cross_store_comparability_output.csv | context_evidence | keyword_matched | lines 1-2 | store_type | n/a |
| order_volume | retail_ops/outputs/demo2_cross_store_comparability_output.csv | quantitative_evidence | keyword_matched | lines 1-2 | transaction_orders | n/a |
| transaction_amount | retail_ops/outputs/demo2_cross_store_comparability_output.csv | quantitative_evidence | keyword_matched | lines 1-2 | transaction_amount | n/a |
| activity_intensity | retail_ops/outputs/demo2_cross_store_comparability_output.csv | quantitative_evidence | keyword_matched | lines 1-2 | activity_orders, activity_order_share_pct, activity_cost, activity_cost_ratio_pct | n/a |
| region_context | retail_ops/data/DATA_DICTIONARY.md | definition_evidence | keyword_matched | lines 74-76 | region_type | n/a |
| competition | retail_ops/COMPARABILITY_GATE_V0.md | boundary_evidence | boundary_matched | lines 101-103 | competition-context requirement in comparability contract | n/a |
| sku_structure | retail_ops/outputs/demo2_cross_store_comparability_output.csv | product_mix_evidence | keyword_matched | lines 1-2 | top3_sku_transaction_amount, top3_sku_transaction_amount_share_pct, SKU source tables | n/a |
| repeated_reporting_windows | retail_ops/outputs/repeated_window_panel_summary_output.csv | quantitative_evidence | keyword_matched | lines 1-2 | repeated period_month records in store_period_panel_metrics | n/a |


## 5. Competing Hypotheses

The `Scenario-Template Confidence` column records deterministic review labels assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`.

| Hypothesis | Scenario-Template Confidence | Status | Weakness |
|---|---:|---|---|
| Stores B-F can be organized in a same-period diagnostic table. | 0.78 | strong | Same-period diagnostic organization does not establish robust comparability. |
| Stores B-F should not be treated as directly comparable without pairwise gates. | 0.86 | strong | Pairwise quantitative thresholds are outside the current review contract. |

## 6. Critic Findings

- [high] Observational evidence supports bounded association claims only. Recommendation: Keep attribution language conditional and record unresolved alternatives.
- [medium] Current evidence scope is limited to committed local project files. Recommendation: Keep source paths and unresolved external requirements explicit.
- [critical] Same-period diagnostic organization must not be described as a completed pairwise comparability gate. Recommendation: Separate same-period diagnostic review from pairwise comparability.

## 7. Claim and Definition Check

- Status: pass
- Unsupported claims detected by current rules: none
- Definition conflicts detected by current rules: none

## 8. Final Judgment

Stores B-F can be organized for same-period diagnostic review, but should not be treated as directly comparable without pairwise gates.

The judgment is bounded by the cited local evidence and the unresolved requirements recorded above.

## 9. Evidence-Coverage Score

0.95

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

- total_packets = 9
- record_matched_packets = 0
- keyword_matched_packets = 8
- boundary_matched_packets = 1
- fallback_packets = 0
- missing_source_files = 0
- direct_evidence_rate = (record_matched_packets + keyword_matched_packets) / total_packets = 0.89
- supported_or_boundary_rate = (record_matched_packets + keyword_matched_packets + boundary_matched_packets) / total_packets = 1.00
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
