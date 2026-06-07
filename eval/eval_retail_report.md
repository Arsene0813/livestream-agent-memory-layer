# Retail Retrieval Evaluation Report

## Scope

This evaluation checks the Store A retail memory retrieval path after the retail extension was kept within the documented Store A Demo 1 boundary.

The current retail evaluation is intentionally narrow. It only covers Store A and the February-April 2026 demo period.

It does not evaluate:

- cross-store comparison;
- store-stage diagnosis;
- causal attribution;
- margin-aware recommendations;
- full SKU category-share analysis;
- automated Meituan backend ingestion.

## Evaluated Behavior

| Evaluation Area | Case Name |
|---|---|
| Visibility and entry profile retrieval | `store_a_visibility_entry_profile` |
| Activity-lever profile retrieval | `store_a_activity_lever_profile` |
| Transaction and conversion profile retrieval | `store_a_transaction_conversion_profile` |
| Refund and invalid-order pressure retrieval | `store_a_order_quality_pressure_profile` |
| Single-metric attribution guard | `single_metric_attribution_guard` |
| Top-SKU evidence limitation | `top3_sku_mix_limited` |
| Unknown-store stage refusal | `unknown_store_stage_refusal` |
| Unsupported cross-store comparison refusal | `cross_store_comparability_not_implemented` |

## Expected System Behavior

The retail retrieval endpoint should:

- retrieve Store A evidence when the requested slot is supported;
- describe observed metrics with caveats;
- avoid one-factor explanations of growth or decline;
- avoid treating activity metrics as standalone causal proof;
- avoid treating top-SKU evidence as full category-share analysis;
- refuse unknown-store stage claims;
- refuse unsupported cross-store strategy recommendations.

## Result

The current result is:

```text
8 / 8 retail evaluation cases passing
```

After modifying retail memory slots or generated facts, rerun:

```bash
python eval/eval_retail.py
```

Then update this report and `eval/results/eval_retail_result.txt` if the case names or outputs change.

## Limitation

This is a scenario-based consistency evaluation. It is not a statistical benchmark and does not prove that the system can make reliable multi-store operating decisions.
