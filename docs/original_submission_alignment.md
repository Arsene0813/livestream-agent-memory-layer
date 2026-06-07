# Submitted Material Mapping

This archival note maps the repository to the supplementary material that described the project as an evolution from an earlier LLM-powered livestream system into a lifecycle-aware agent memory layer.

It is not the main admissions-facing project narrative. The current admissions-facing summary is `PROJECT_SUMMARY_FOR_ADMISSIONS.md`. The current retail implementation details are under `retail_ops/`.

## Mapping

| Submitted material theme | Repository component |
|---|---|
| Structured memory | Typed fact storage and fact-policy registry. |
| Retrieval control | Retrieval gating, fallback, and refusal behavior. |
| Fact update handling | Slot-based overwrite and soft deactivation. |
| Memory freshness | Timestamps, freshness windows, and active-state flags. |
| Traceability | Supporting memory items returned with answers. |
| Livestream commerce use case | Product price, promotions, stock status, shipping policy, and product features. |
| Retail operations extension | Meituan instant-retail diagnostic evidence, SQL outputs, generated memory facts, and answer-boundary checks. |

## Current Retail Extension Boundary

The retail extension should be read as a staged decision-support prototype based on real store-operation problems.

It is not a finished automated operating system, a complete 48-store deployment, or a finished pairwise comparability gate.

The repository keeps the original lifecycle-aware memory direction while extending the same evidence-control design into Meituan instant-retail operations.
