# Retrieval Flow Visualization

This note explains how a query moves through the current retrieval path in the local prototype.

It is not a production monitoring dashboard and does not claim live Meituan backend integration. The retail examples are based on generated retail memory facts and field-contract notes from the current Demo 1 and Demo 2 evidence files.

## Flow

```text
User query
  ↓
Embedding with local bge-m3
  ↓
Top-k retail memory facts and field-contract notes
  ↓
Similarity threshold check
  ↓
Top-1 / top-2 margin check
  ↓
Entity / period / slot check
  ↓
Accepted context or strict refusal / qualified answer
```

## What the Flow Is Meant to Prevent

The retrieval system should not treat semantic similarity as enough evidence for an operating conclusion.

A high-scoring retrieved fact may still be unsafe to use when:

- the query asks for a metric that is not in the current evidence;
- the query mixes stores, months, or demo scopes;
- the query asks for a strategy-transfer decision;
- the retrieved metric has a documented interpretation boundary;
- several candidate facts have similar scores and the comparison question is too broad.

## Case Pattern 1: Supported Query Accepted

Example query:

```text
What does Store B's March 2026 visibility and search-entry profile show?
```

Expected retrieval behavior:

- retrieve Store B's `visibility_entry_profile`;
- preserve March 2026 scope;
- use `search_entry_rate_pct` and `search_entry_share_pct` as diagnostic evidence;
- avoid turning search-entry strength into a full store-performance conclusion.

## Case Pattern 2: Related Fact Retrieved but Answer Must Be Qualified

Example query:

```text
Does activity_cost_ratio_pct show activity ROI?
```

Expected retrieval behavior:

- retrieve an activity-related fact or field definition;
- recognize that `activity_cost_ratio_pct` is not ROI;
- answer with the metric boundary instead of accepting the user's wording.

## Case Pattern 3: Entity or Period Mismatch

Example query:

```text
What was Store B's April 2026 order_conversion_rate_pct?
```

Expected retrieval behavior:

- avoid answering from Store B March 2026 evidence;
- avoid borrowing Store A April 2026 evidence;
- return an insufficient-evidence or scope-limited answer.

## Case Pattern 4: Ambiguous Comparison

Example query:

```text
Which store has stronger search performance?
```

Expected retrieval behavior:

- retrieve multiple related visibility/search-entry facts;
- detect that the question needs a specific metric and operating purpose;
- avoid ranking stores globally from one retrieved metric.

## Relationship to Retrieval Threshold Calibration

The file `retail_ops/outputs/retrieval_threshold_summary.md` summarizes the score distribution across supported, unsupported, hard-negative, entity/period mismatch, and ambiguous comparison queries.

The score distribution helps inspect threshold behavior. It does not replace answer-boundary checks.
