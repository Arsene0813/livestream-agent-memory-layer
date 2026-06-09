# Answer

## 1. Direct Answer

Promotion changes should be checked against cost, conversion, SKU structure, margin, and competitor context.

This is a deterministic mock result. It proves the workflow can run end-to-end, but it does not claim live retrieval or autonomous world modeling.

## 2. Question Type

- Question type: strategic_recommendation
- Domain: retail_operations

## 3. Relevant Factors Considered

| Factor | Weight | Evidence Status | Why It Matters |
|---|---:|---|---|
| activity_orders | 0.60 | partially_supported | Potentially relevant but requires stronger evidence. |
| activity_cost | 0.60 | partially_supported | Potentially relevant but requires stronger evidence. |
| merchant_subsidy | 0.60 | partially_supported | Potentially relevant but requires stronger evidence. |
| platform_subsidy | 0.60 | partially_supported | Potentially relevant but requires stronger evidence. |
| order_conversion | 0.85 | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| payment_conversion | 0.72 | partially_supported | Important context but not sufficient on its own. |
| sku_margin_structure | 0.85 | partially_supported | Central to avoiding overconfident or misleading conclusions. |
| competitor_context | 0.60 | partially_supported | Potentially relevant but requires stronger evidence. |

## 4. Evidence Used

| Evidence | Source | Supports | Limitations |
|---|---|---|---|
| evidence_activity_orders | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: activity_orders. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |
| evidence_activity_cost | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: activity_cost. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |
| evidence_merchant_subsidy | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: merchant_subsidy. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |
| evidence_platform_subsidy | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: platform_subsidy. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |
| evidence_order_conversion | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: order_conversion. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |
| evidence_payment_conversion | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: payment_conversion. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |
| evidence_sku_margin_structure | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: sku_margin_structure. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |
| evidence_competitor_context | retail_ops/data/DATA_DICTIONARY.md | Provides context needed to evaluate factor: competitor_context. | Deterministic mock pipeline only.; No live retrieval is performed in this step. |

## 5. Competing Hypotheses

Hypothesis confidence values are deterministic scenario-template values assigned by `generate_hypotheses(question_type)` in `rac/src/mock_pipeline.py`. They are not learned probabilities, calibrated likelihoods, or direct calculations from observed Meituan metric tables.

| Hypothesis | Confidence | Status | Weakness |
|---|---:|---|---|
| Promotion decisions should be checked against cost, conversion, SKU structure, margin, and competitor context. | 0.84 | strong | Final action still requires real margin and competitor evidence. |

## 6. Critic Findings

- [high] The mock pipeline must not claim causal proof from observational evidence. Recommendation: Use cautious language and state that attribution is not proven.
- [medium] The mock pipeline uses structured placeholder evidence rather than live retrieval. Recommendation: State this limitation clearly.
- [high] Promotion recommendations require margin and competitor context. Recommendation: Avoid final action recommendations without these checks.

## 7. Final Judgment

Promotion changes should be checked against cost, conversion, SKU structure, margin, and competitor context.

The conclusion is conservative because this mock pipeline uses structured placeholder evidence and does not perform live retrieval.

## 8. Scenario-Template Confidence

0.80

How this value is assigned:

- Source: `build_belief_update(question_type)` in `rac/src/mock_pipeline.py`.
- Rule: deterministic case-template assignment by question type.
- It is not calculated from evidence-packet counts, factor weights, or observed metric tables.
- It is not learned from historical data.
- It is not a calibrated probability or Bayesian posterior.
- It is not a causal confidence score or business-success probability.
- It is kept only to show how the deterministic mock scaffold carries a review-state value.
- Grounded reports use a formula-based `Evidence-Coverage Score` instead.

## 9. What Cannot Be Concluded

- The mock pipeline does not compute real margins.
- Competitor data may be incomplete.
- One reporting window is insufficient for robust action attribution.

## 10. Review-State Update

- review_state_id: promotion_changes_require_multi_factor_check
- status: active
- validity_conditions:
 - Retail operations decision-support questions.
