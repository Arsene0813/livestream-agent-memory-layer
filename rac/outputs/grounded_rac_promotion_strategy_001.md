# Grounded RAC Report

## 1. Direct Answer

The current evidence supports a bounded promotion-review checklist, not an automatic promotion change.

Deterministic local-file review; routing scores summarize route resolution under the current rules.

## 2. Question Type

- Question type: strategic_recommendation
- Domain: retail_operations

## 3. Factor Weights

### 3a. How Factor Weights Are Generated

Factor weights are fixed review-priority buckets assigned by explicit rules in `rac/src/mock_pipeline.py`. They order review attention within the current evidence scope.

| Bucket | Weight | Rule | Factors in This Report |
|---|---:|---|---|
| high | 0.85 | Central to avoiding overconfident or misleading conclusions. | order_conversion, sku_margin_structure |
| medium | 0.72 | Important context but not sufficient on its own. | payment_conversion |
| default | 0.60 | Potentially relevant but requires stronger evidence. | activity_orders, activity_cost, merchant_subsidy, platform_subsidy, competitor_context |

Weighting boundary:

- Use these values only to order review attention within the current evidence scope.

### 3b. Factor Weights Used in This Report

| Decision Factor ID | Weight | Bucket | Evidence Status | Why It Matters |
|---|---:|---|---|---|
| activity_orders | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| activity_cost | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| merchant_subsidy | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| platform_subsidy | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| order_conversion | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| payment_conversion | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| sku_margin_structure | 0.85 | high | missing | Central to avoiding overconfident or misleading conclusions. |
| competitor_context | 0.60 | default | missing | Potentially relevant but requires stronger evidence. |

`partially_supported` indicates that a registered local evidence route was resolved for the factor. It does not necessarily mean that observed numeric evidence supports the decision.

## 4. Local Evidence Grounding

- Total evidence packets: 8
- Record matched packets: 0
- Keyword matched packets: 6
- Boundary matched packets: 2
- Fallback packets: 0
- Missing source files: 0

For CSV evidence, `Source Locator` shows the selected record scope and `Selected Values` shows values read from those records. For Markdown evidence, the locator remains a local line-range pointer.
`Decision Factor ID` is an internal RAC review identifier. The field column shows canonical project fields where available and labels unresolved requirements explicitly.

| Decision Factor ID | Source | Evidence Type | Status | Source Locator | Canonical Evidence Fields / Requirement | Selected Values |
|---|---|---|---|---|---|---|
| activity_orders | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | lines 659-661 | activity_orders | n/a |
| activity_cost | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | lines 665-667 | activity_cost | n/a |
| merchant_subsidy | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | lines 671-673 | merchant_subsidy_amount | n/a |
| platform_subsidy | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | lines 677-679 | platform_subsidy_amount | n/a |
| order_conversion | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | lines 523-525 | order_conversion_rate_pct, order_users, entry_users | n/a |
| payment_conversion | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | lines 575-577 | payment_conversion_rate_pct, payment_users, order_users | n/a |
| sku_margin_structure | retail_ops/COMPARABILITY_GATE_V0.md | boundary_evidence | boundary_matched | lines 137-139 | required SKU margin context; unavailable in current evidence | n/a |
| competitor_context | retail_ops/COMPARABILITY_GATE_V0.md | boundary_evidence | boundary_matched | lines 71-73 | required competitor context; unavailable in current evidence | n/a |


## 5. Competing Hypotheses

The `Scenario-Template Confidence` column records deterministic review labels assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`.

| Hypothesis | Scenario-Template Confidence | Status | Weakness |
|---|---:|---|---|
| A bounded promotion review should cover activity cost, merchant and platform subsidy, and order and payment conversion. | 0.84 | strong | The available evidence defines review dimensions but does not establish a promotion outcome. |
| The current evidence can support a bounded promotion review checklist, but not an automatic promotion change. | 0.68 | plausible | Repeated-period cost evidence is required for trend interpretation; SKU margin and competitor context remain unresolved for final action. |

## 6. Critic Findings

- [medium] Current evidence scope is limited to committed local project files. Recommendation: Keep source paths and unresolved external requirements explicit.
- [high] SKU margin and competitor context remain unresolved for final promotion action. Recommendation: Keep the output at bounded review-checklist level.

## 7. Claim and Definition Check

- Status: pass
- Unsupported claims detected by current rules: none
- Definition conflicts detected by current rules: none

## 8. Final Judgment

The current evidence supports a bounded promotion-review checklist, not an automatic promotion change.

The judgment is bounded by the cited local evidence and the unresolved requirements recorded above.

## 9. Evidence-Routing Coverage

Packet composition:

- Total packets: 8
- Record matched packets: 0
- Keyword matched packets: 6
- Boundary matched packets: 2
- Fallback packets: 0
- Missing source files: 0

- Routing coverage score: 0.89
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

- total_packets = 8
- record_matched_packets = 0
- keyword_matched_packets = 6
- boundary_matched_packets = 2
- fallback_packets = 0
- missing_source_files = 0

Derived rates and checks:

- record_or_keyword_route_rate = (record_matched_packets + keyword_matched_packets) / total_packets = 0.75
- resolved_or_boundary_route_rate = (record_matched_packets + keyword_matched_packets + boundary_matched_packets) / total_packets = 1.00
- no_missing_source_file_score = 1.00
- no_fallback_score = 1.00

Reading the score:

- A higher value means more requested evidence routes were resolved or explicitly bounded.
- Boundary evidence contributes when it documents a missing requirement.
- Read the score as coverage rather than evidence strength, causal validity, decision quality, or business impact.

## 10. What Cannot Be Concluded

- Margin fields are absent from the current evidence.
- Competitor data may be incomplete.
- One reporting window is insufficient for robust action attribution.

## 11. Review-State Update

- review_state_id: promotion_changes_require_multi_factor_check
- status: active
- validity_conditions:
  - Retail operations decision-support questions.
