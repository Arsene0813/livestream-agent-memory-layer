# Future Work: Pairwise Comparability Gate

This note records the next planned stage of the retail operations prototype.

The current implemented retail scope stops at:

1. Demo 1: Store A month-over-month diagnostic.
2. Demo 2: Stores B-F same-period diagnostic structure.

The next stage is a pairwise comparability gate. It is documented here as future work.

## Why This Gate Matters

The Meituan merchant backend provides rich store-level data, but the workflow is mainly designed for reviewing one store at a time.

For a 48-store operation, the harder problem is deciding:

- which store-period records are comparable, not just which stores look similar;
- under what operating conditions the comparison is valid;
- what kind of operating question the comparison can support;
- what limitation should stop the system from giving unsupported advice.

The gate should answer one narrower question:

- Can these two store-period records be compared for this specific operating question?

## Current Demo 2 Output vs Future Pairwise Gate

### Current Demo 2 Output

Current Demo 2 provides:

- row-level same-period diagnostic readiness;
- current implemented fields including `comparison_scope_flag` and `comparison_limit_notes`;
- evidence about whether each store-period row is inside the current Demo 2 scope;
- interpretation limits that should be preserved in later retrieval or analysis.

The current `comparison_limit_notes` field covers only a limited subset of future gate factors.

It does not yet encode:

- transaction-volume bands;
- transaction-scale bands;
- competition context;
- repeated-window stability;
- data-supported market-area classification;
- pairwise operating-transfer evidence.

### Future Pairwise Comparability Gate

A future pairwise comparability gate should provide:

- a pair-level decision for a specific comparison question;
- input that identifies a reference store, a candidate store, and the operating question being asked;
- output that explains whether the selected records can be compared;
- supporting evidence and limiting factors.

The current `comparison_scope_flag` should not be treated as a pairwise comparability decision.

## What Store Comparability Should Depend On

A reliable gate should judge a selected pair of store-period records for a specific operating question.


The future gate should compare selected store-period records for a specific operating question instead of creating a store label from one threshold.

## Operating Question Evidence Matrix

This matrix separates what the current evidence can support from what the future pairwise gate would still need. It keeps the future gate tied to operating questions instead of turning the current Demo 2 fields into a general store label.

| Operating question | Current usable evidence | Current missing evidence | Current allowed answer |
|---|---|---|---|
| Search-entry structure comparison | `search_exposure_users`, `search_average_rank`, `search_entry_users`, `search_entry_rate_pct`, `search_entry_share_pct`, `entry_conversion_rate_pct`, `order_conversion_rate_pct` | repeated windows, competition context, stronger market context | cautious same-period diagnostic comparison |
| Promotion or subsidy comparison | `activity_orders`, `activity_order_share_pct`, `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | campaign calendar, campaign mechanism, competitor reaction, repeated activity windows | activity involvement and activity intensity only |
| SKU-mix comparison | top-SKU transaction-amount evidence, top-SKU sales-volume evidence, `top3_sku_transaction_amount_share_pct` | full product-category classification, broader catalog structure, category-level share | lightweight product-mix constraint |
| Market-context comparison | `region_type` | local consumption level, delivery-radius context, competitor density, broader store coverage | weak context only |
| Strategy-transfer readiness | current diagnostic fields and `comparison_limit_notes` | repeated windows, pairwise decision output, activity mechanism, competition, market context, fulfillment and stockout evidence | boundary-preserving answer until pairwise evidence is available |


## Activity Evidence Boundary

Activity evidence should be separated into:

- explicit activity status, only if campaign-calendar or backend status evidence exists;
- activity involvement, currently proxied by `activity_orders` and `activity_order_share_pct`;
- activity intensity, currently proxied by `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, and `platform_subsidy_amount`.

A future `activity_status` field should be added only if campaign-calendar or explicit backend activity-status evidence is available and documented in `retail_ops/data/DATA_DICTIONARY.md` and the lineage section of `retail_ops/TECHNICAL_APPENDIX.md`.

## Activity Factor Boundary

The future gate should not collapse activity evidence into a single yes/no label.

| Activity factor | Current evidence | Current boundary |
|---|---|---|
| Activity status | Not directly implemented as a source field. | A future gate should not infer full campaign status only from `activity_orders > 0`. |
| Activity involvement | `activity_orders`, `activity_order_share_pct` | Shows how much of the transaction-order structure involved activity orders; it does not prove activity caused demand. |
| Activity intensity | `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | Shows subsidy and cost structure, but not full campaign mechanism, competitor response, or activity calendar. |

## Candidate Gate Factors

| Future factor | Current evidence available | Current limitation | Future evidence needed |
|---|---|---|---|
| Reporting-window alignment | `period_start`, `period_end`, `period_month` | Demo 2 uses one March 2026 window only. | Repeated windows across more stores. |
| Order volume | `transaction_orders` | One-period volume may be unstable. | Repeated order-volume bands. |
| Transaction scale | `transaction_amount`, `average_order_value`; `estimated_income_proxy` as weak supplementary backend context only | `estimated_income_proxy` lacks a full calculation breakdown in the current demo data and should not be used as a primary gate factor. | Clearer gross-revenue, cost, and settlement evidence if available. |
| Activity status | Not directly implemented as a current source field. | Full campaign status should not be inferred from activity orders alone. | Campaign-calendar or explicit backend activity-status evidence. |
| Activity involvement | `activity_orders`, `activity_order_share_pct` | Shows activity participation in the order structure, not causal lift. | Repeated windows and campaign mechanism evidence. |
| Activity intensity | `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | Current thresholds are diagnostic guardrails, not transfer rules. | Repeated activity windows and stronger evidence on activity mechanism. |
| Store type | `store_type` | Store type alone does not prove comparability. | Broader sample by store type. |
| Region and market context | `region_type` | Current demo sample is too small for reliable regional classification. | More store data, local consumption-level evidence, delivery-radius context, and competition-context evidence. |
| Competition context | Not currently structured. | Local competitor density and price pressure are not included. | Competitor and local market evidence. |
| SKU structure | Top-SKU transaction-amount and sales-volume evidence. | Top-SKU evidence is not full category-share analysis. | Broader SKU classification or category mapping. |
| Fulfillment or stockout context | Not currently structured. | Delivery condition, stockout, and fulfillment-cause evidence are not included. | Fulfillment status, stockout history, delivery-radius or delivery-capacity evidence if available. |
| Data completeness | `comparison_scope_flag`, `comparison_limit_notes` | Current notes are diagnostic guardrails, not a gate decision. | Explicit pairwise decision output after broader data. |

Top-SKU evidence is not full category-share analysis, and missing amount or volume fields are not back-calculated because store-level SKU prices may vary by operating stage and local competitive context.

## Region and Market-Context Boundary

`region_type` follows the central definition in `retail_ops/data/DATA_DICTIONARY.md`: weak region or market-context metadata for the current demo.

Future market-area classification should be added as new documented fields only after broader store coverage, repeated windows, local consumption-level evidence, competition evidence, delivery-radius evidence, and activity-condition evidence are available.

The future gate should not classify stores by subjective experience, intuition, or habitual labels.

## Question-Specific Comparability

A store pair may be comparable for search-entry structure while still requiring additional evidence for promotion transfer, pricing pressure, SKU strategy, refund interpretation, or fulfillment interpretation.

The gate should return a decision for the selected operating question and explain the supporting evidence, limiting factors, and blocking factors.

## Minimum Evidence Before Implementation

A stronger pairwise comparability gate should be implemented after Demo 2 evidence is expanded beyond one selected month, or after missing evidence is explicitly represented as a blocking or limiting factor for the selected operating question. The gate should wait until the evidence below is available, or until the missing evidence is explicitly marked as a blocking or limiting factor for the selected operating question.

| Gate factor | Current Demo 2 support | Missing before gate | Future source |
|---|---|---|---|
| Reporting-window alignment | `period_start`, `period_end`, `period_month` | Repeated windows beyond one selected month | Backend exports across more store-period records |
| Order volume | `transaction_orders` | Volume bands and repeated-window stability | Store-period metrics |
| Transaction scale | `transaction_amount`, `average_order_value`, `estimated_income_proxy` | Scale bands and clearer income/cost evidence where available | Store-period metrics and backend financial views |
| Activity status | Not directly implemented | Campaign calendar or explicit backend activity-status evidence | Backend activity records or manually documented campaign calendar |
| Activity involvement | `activity_orders`, `activity_order_share_pct` | Repeated activity windows and campaign mechanism evidence | Activity-order metrics over more periods |
| Activity intensity | `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | Subsidy-intensity stability and context for why the activity was used | Activity-cost and subsidy records |
| Store type | `store_type` | Broader sample by operating model | Store metadata |
| Region and market context | `region_type` as weak context only | Data-supported market-area classification, local consumption evidence, delivery-radius context, and competition context | More store data plus external or manually reviewed market evidence |
| Competition context | Not currently structured | Competitor density, price pressure, ranking pressure, and local market reaction | Manual collection, platform observation, or future structured competitor table |
| SKU structure | Top-SKU transaction-amount and sales-volume evidence | Broader SKU classification, category share, and margin-aware structure where available | SKU export and future category mapping |
| Fulfillment or stockout context | Not currently structured | Delivery-condition, stockout, fulfillment-capacity, or delivery-radius evidence | Future fulfillment or inventory records |
| Data completeness | `comparison_scope_flag`, `comparison_limit_notes` | Pairwise decision output and question-specific evidence coverage | Future gate output contract |

Before then, Demo 2 remains a same-period diagnostic stage for preserving field definitions, diagnostic signals, and interpretation limits before pairwise gate implementation.

## Future Decision Flow

A future gate should be implemented as a question-specific decision flow, not as a global store score.

1. Expand store-period coverage across more of the 48-store operation.
2. Check whether diagnostic signals remain stable across repeated reporting windows.
3. For a selected operating question, check whether the candidate store-period records have the required evidence.
4. Check reporting-window alignment.
5. Check whether transaction order volume and transaction amount are within a reasonable comparison band.
6. Check activity involvement and activity intensity, without inferring full activity status unless campaign-calendar evidence exists.
7. Check store type and market-context evidence, while keeping `region_type` as weak context only.
9. Return a question-specific `comparison_decision` with supporting fields, limiting factors, allowed interpretation, and unsupported interpretation.

The gate should separate at least these comparison questions:

- search-entry comparability;
- promotion or subsidy comparability;
- SKU-mix comparability;
- data-completeness comparability;
- fulfillment or stockout comparability;
- strategy-transfer readiness.

The future flow should return a question-specific decision for the selected record pair, with supporting evidence and blocking factors.

## Future Gate Input Triple

A future comparability gate should start from a narrow input triple:

- `reference_store_id`
- `candidate_store_id`
- `comparison_question_type`

These names describe a proposed future gate contract. They are not current implemented data-contract fields.

Before any of them is used in CSV outputs, generated memory facts, or evaluation cases, the field must be documented in:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/TECHNICAL_APPENDIX.md`

## Future Gate Output Contract

A future gate should return a small decision object with these fields:

- `comparison_decision`
- `supporting_fields`
- `blocking_or_limiting_factors`
- `allowed_interpretation`
- `unsupported_interpretation`

Candidate decision values:

- `comparable`
- `comparable_with_limits`
- `not_comparable`
- `insufficient_evidence`

These names are future contract terms only. They are not current Demo 2 output columns.

## Contract-Only Examples

The following examples describe intended future gate behavior. They are not current Demo 2 outputs.

| Reference store | Candidate store | Comparison question | Likely future decision type | Reasoning boundary |
|---|---|---|---|---|
| B | D | Compare search-entry structure | `comparable_with_limits` | Same reporting window and both records contain search-entry evidence, but activity involvement, region context, store type, and repeated-window stability would still limit interpretation. |
| C | E | Transfer promotion strategy | `insufficient_evidence` | Current Demo 2 has activity involvement and cost-ratio evidence, but not campaign mechanism, activity calendar, competitor reaction, or repeated-window stability. |
| B | F | Compare market-area performance | `insufficient_evidence` | `region_type` is weak context only. Market-area classification needs broader store data and supporting local-consumption and competition evidence. |

## Concrete Future Gate Example

This section is a contract example only. It describes intended future behavior.
It is not a current Demo 2 output and does not change the canonical field
definitions in `retail_ops/data/DATA_DICTIONARY.md`.

The future gate should answer a narrow comparison question, not produce a
question-specific comparability decision.

| Future gate item | Example content |
|---|---|
| Reference record | Store B, March 2026 |
| Candidate record | Store F, March 2026 |
| Operating question | Can search-entry structure be compared between these records? |
| Possible decision | Comparable with limits. |
| Why limited | Same-period search-entry evidence may support a cautious diagnostic comparison, but current evidence is not enough to approve promotion transfer, pricing strategy, market-area classification, or general store ranking. |

A different operating question would require a different gate decision. For
example, the same two records might be usable for search-entry diagnosis but
still insufficient for deciding whether one store's subsidy strategy should be
transferred to another store.

## Implementation Boundary

This document freezes the future contract without claiming that a pairwise gate has been implemented.

Current implemented artifacts remain:

- Demo 1 SQL diagnostics and generated Store A memory facts;
- Demo 2 same-period B-F diagnostic output;
- Demo 2 generated retail memory facts;
- answer-boundary evaluations for the implemented evidence path.

<!-- COMPARABILITY_GATE_FUTURE_PLAN_START -->
## Future Implementation Plan: Question-Specific Pairwise Gate

The future gate should remain question-specific. It should not decide whether two stores are generally comparable. It should decide whether two store-period records can support one selected comparison question.

| Stage | Purpose | Conservative boundary |
|---|---|---|
| Pairwise gate | Return a future decision such as `comparable`, `comparable_with_limits`, or `insufficient_evidence` for one question type | The decision should not approve strategy transfer by itself |
| Factor-aware review | Expand the question into relevant factors, assign factor weights, retrieve evidence by factor, and generate competing hypotheses | Factor weights should be inspectable and evidence-bounded |
| Critique and confidence update | Check unsupported assumptions, missing evidence, and contradiction risks before writing the final answer | The output should use confidence-and-limitation update, not a final causal claim |

This plan does not add current Demo 2 output columns. Demo 2 remains a same-period diagnostic evidence layer. The pairwise gate becomes valid only after more store-period windows and stronger market-context evidence are added.
<!-- COMPARABILITY_GATE_FUTURE_PLAN_END -->

## Why the Same Store Pair May Receive Different Decisions

The future gate should not decide whether two stores are generally comparable. It should decide whether the records are comparable for one selected question.

| Pair | Question | Possible future decision | Reason |
|---|---|---|---|
| B vs D | Compare search-entry structure | `comparable_with_limits` | Both records have same-period search-entry evidence, but activity and region limits still apply. |
| B vs D | Transfer activity strategy | `insufficient_evidence` | Current evidence lacks campaign calendar, competitor response, and repeated activity windows. |
| B vs D | Compare market-area performance | `insufficient_evidence` | `region_type` is weak context and not a market-area classification. |
