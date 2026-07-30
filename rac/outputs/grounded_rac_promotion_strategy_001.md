# Grounded RAC Report

## 1. Direct Answer

Promotion changes should be checked against cost, conversion, SKU structure, margin, and competitor context.

Deterministic local-file review; routing scores summarize evidence coverage under the current rules.

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

| Factor | Weight | Bucket | Evidence Status | Why It Matters |
|---|---:|---|---|---|
| activity_orders | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| activity_cost | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| merchant_subsidy | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| platform_subsidy | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |
| order_conversion | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| payment_conversion | 0.72 | medium | partially_supported | Important context but not sufficient on its own. |
| sku_margin_structure | 0.85 | high | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| competitor_context | 0.60 | default | partially_supported | Potentially relevant but requires stronger evidence. |

## 4. Local Evidence Grounding

- Total evidence packets: 8
- Keyword matched packets: 6
- Boundary matched packets: 2
- Fallback packets: 0
- Missing source files: 0

The `Source Lines` column is an audit pointer to the local source-file line range used for each evidence row. It is not a business metric. The `Evidence Fields` column lists the canonical fields or documented evidence concepts used for review; raw matched keywords are intentionally not shown.

| Factor | Source | Evidence Type | Status | Source Lines | Evidence Fields |
|---|---|---|---|---|---|
| activity_orders | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | 618-620 | ### `activity_orders` |
| activity_cost | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | 624-626 | ### `activity_cost` |
| merchant_subsidy | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | 630-632 | ### `merchant_subsidy_amount` |
| platform_subsidy | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | 636-638 | ### `platform_subsidy_amount` |
| order_conversion | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | 482-484 | ### `order_conversion_rate_pct` |
| payment_conversion | retail_ops/data/DATA_DICTIONARY.md | default_evidence | keyword_matched | 534-536 | ### `payment_conversion_rate_pct` |
| sku_margin_structure | retail_ops/COMPARABILITY_GATE_V0.md | boundary_evidence | boundary_matched | 137-139 | margin-aware structure |
| competitor_context | retail_ops/COMPARABILITY_GATE_V0.md | boundary_evidence | boundary_matched | 71-73 | competitor reaction |


## 5. Competing Hypotheses

The `Scenario-Template Confidence` column records deterministic review labels assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`.

| Hypothesis | Scenario-Template Confidence | Status | Weakness |
|---|---:|---|---|
| Promotion decisions should be checked against cost, conversion, SKU structure, margin, and competitor context. | 0.84 | strong | Final action still requires real margin and competitor evidence. |
| The current evidence can support a bounded promotion review checklist, but not an automatic promotion change. | 0.68 | plausible | The current grounded RAC pipeline does not calculate a real cost trend. |

## 6. Critic Findings

- [high] The grounded RAC pipeline must not claim causal proof from observational evidence. Recommendation: Use cautious language and state that attribution is not proven.
- [medium] The grounded RAC pipeline uses deterministic local file evidence resolution rather than live backend or vector retrieval. Recommendation: State this limitation clearly.
- [high] Promotion recommendations require margin and competitor context. Recommendation: Avoid final action recommendations without these checks.

## 7. Claim and Definition Check

- Status: pass
- Unsupported claims detected by current rules: none
- Definition conflicts detected by current rules: none

## 8. Final Judgment

Promotion changes should be checked against cost, conversion, SKU structure, margin, and competitor context.

The judgment is bounded by the cited local evidence and the unresolved requirements recorded above.

## 9. Evidence-Coverage Score

0.89

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

Formula limitation:

- These weights are fixed prototype heuristics.
- They are not learned parameters, optimized thresholds, calibrated probabilities, or business-performance predictors.
- The score is intended to make evidence-routing coverage inspectable, not to prove that a business conclusion is correct.

Future sensitivity check:

- A future sensitivity check should report how the numeric score changes under alternative weight settings.
- Weight sensitivity indicates score instability only; it must not be used to change the report judgment.

Current report inputs:

- total_packets = 8
- keyword_matched_packets = 6
- boundary_matched_packets = 2
- fallback_packets = 0
- missing_source_files = 0
- direct_evidence_rate = keyword_matched_packets / total_packets = 0.75
- supported_or_boundary_rate = (keyword_matched_packets + boundary_matched_packets) / total_packets = 1.00
- no_missing_source_file_score = 1.00
- no_fallback_score = 1.00

Interpretation boundary:

- This is a deterministic evidence-routing coverage score for the current local report.
- It is not a learned probability, Bayesian posterior, causal confidence score, or business-success probability.
- Boundary-matched evidence can increase coverage because it explicitly records missing requirements instead of hiding them.
- A high score means more requested evidence routes were locally resolved or explicitly bounded under the current rules.
- It does not measure evidence strength, conclusion correctness, decision quality, or business impact, and it is not used to select the final judgment.

## 10. What Cannot Be Concluded

- The current grounded RAC pipeline does not compute real margins.
- Competitor data may be incomplete.
- One reporting window is insufficient for robust action attribution.

## 11. Review-State Update

- review_state_id: promotion_changes_require_multi_factor_check
- status: active
- validity_conditions:
  - Retail operations decision-support questions.
