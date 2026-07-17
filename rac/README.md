# Factor-Aware Grounded Review (RAC)

RAC is the factor-aware grounded review layer of the current
decision-support prototype.

It takes a decision question and structured local evidence, makes the
review path explicit, and records where the available evidence supports
a statement, where only a boundary can be established, and where a
stronger conclusion should remain unresolved.

The runnable implementation is deterministic and file-grounded. It uses
local project files and explicit review rules so that the reasoning
contract can be inspected and evaluated independently of model behavior.

For the reviewer-facing cases and generated reports, start with
[`DEMO_INDEX.md`](DEMO_INDEX.md).

## Implemented Review Path

```text
question
-> question analysis
-> factor expansion
-> factor weighting
-> source-aware local evidence routing
-> boundary evidence for unavailable requirements
-> competing hypotheses
-> critique
-> fact checking
-> evidence-coverage and limitation update
-> grounded final report
-> quality gate
```

The pipeline separates factor selection, evidence retrieval, hypothesis
formation, critique, and report generation. A final answer therefore
retains an inspectable path back to the evidence packets and limitations
used during review.

## Why This Layer Exists

A plausible answer is not necessarily a supported decision.

RAC adds explicit checks for:

- which factors are relevant to the question;
- which source files were used;
- which assumptions remain unsupported;
- which competing explanations remain possible;
- which requirements are absent from the current evidence;
- which conclusion should be qualified or withheld;
- whether the final report preserves the evidence boundary.

This layer complements the retail field dictionary, SQL diagnostics,
generated memory facts, lineage validation, and answer-boundary
evaluation.

## Current Implementation

The implemented RAC scaffold includes:

- a shared review-state contract;
- deterministic question analysis;
- question-specific factor expansion;
- explicit heuristic factor weights;
- local file-based evidence routing;
- source-path existence checks;
- factor-specific keyword matching;
- local evidence snippets and line ranges;
- boundary evidence for unavailable requirements;
- competing-hypothesis templates;
- critique and fact-check stages;
- evidence-coverage scoring;
- explicit limitation updates;
- grounded Markdown and JSON reports;
- a deterministic grounded quality gate.

The current runnable path uses local files and deterministic rules.
Graph orchestration, model calls, vector retrieval, and live
merchant-backend access are future integration options rather than
requirements for the present evidence contract.

## Evidence Types

RAC distinguishes three evidence-routing outcomes.

| Evidence type | Meaning | Review treatment |
|---|---|---|
| Direct evidence | A local source contains factor-relevant evidence for the current question. | The report may use the evidence within its documented scope. |
| Boundary evidence | A local source explicitly records that a required field, gate, or condition is not implemented or not available. | The report records the missing requirement instead of inventing supporting evidence. |
| Fallback evidence | No sufficiently specific direct or boundary source was resolved. | Evidence coverage is reduced and the limitation remains visible. |

For the cross-store comparability case, requirements such as competition
context and repeated reporting windows can be routed to
`retail_ops/COMPARABILITY_GATE_V0.md` as boundary evidence. This allows
the report to distinguish a documented missing requirement from a
missing source-resolution result.

## Factor Weight Generation

Factor weights are generated in `rac/src/mock_pipeline.py` using explicit
factor-ID buckets.

```text
high-priority review factors   -> 0.85
medium-priority review factors -> 0.72
default relevant factors       -> 0.60
```

### High-priority factors

```text
promotion_intensity
activity_intensity
order_conversion
sku_margin_structure
evidence_packets
belief_records
retrieval_trace
```

### Medium-priority factors

```text
search_exposure
entry_conversion
same_reporting_period
store_type
order_volume
transaction_amount
transaction_orders
payment_conversion
typed_memory
hypotheses
confidence
limitations
active_state_filtering
```

Any relevant factor not listed in the high- or medium-priority buckets
receives the default value.

These values are review-priority weights. They are fixed prototype
heuristics, not learned parameters, metric-derived estimates,
probabilities, business thresholds, or causal-effect estimates.

The grounded reports expose the bucket, factor membership, assigned
weight, and interpretation limit so the weighting rule can be reviewed
directly.

## Evidence-Coverage Score

Grounded RAC reports use a formula-based Evidence-Coverage Score:

```text
evidence_coverage_score
= 0.45 * direct_evidence_rate
+ 0.25 * supported_or_boundary_rate
+ 0.15 * no_missing_source_file_score
+ 0.15 * no_fallback_score
```

| Component | Weight | Rationale |
|---|---:|---|
| `direct_evidence_rate` | 0.45 | Direct local evidence receives the highest weight. |
| `supported_or_boundary_rate` | 0.25 | Explicit boundary evidence is preferable to an unsupported inference. |
| `no_missing_source_file_score` | 0.15 | Existing source paths are required for traceability. |
| `no_fallback_score` | 0.15 | Unresolved fallback packets reduce evidence coverage. |

The score summarizes local evidence-routing coverage. It is not a
Bayesian posterior, calibrated probability, causal-confidence estimate,
or predicted business-success rate.

Mock reports separately carry Scenario-Template Confidence. That value
is assigned by deterministic question-type templates and is not merged
with the grounded Evidence-Coverage Score.

## Competing Hypotheses, Critique, and Fact Checks

RAC does not route evidence directly into a single preferred narrative.

For each review case, the pipeline can retain multiple plausible
explanations, then check:

- whether each hypothesis has relevant evidence;
- whether the hypothesis exceeds the evidence scope;
- whether another explanation remains plausible;
- whether a causal or strategy-transfer statement is unsupported;
- whether source paths and snippets are present;
- whether the final report carries unresolved limitations forward.

This structure is especially important when observational store metrics
move in different directions or when same-period records lack the
question-specific context required for comparison.

## Runnable Layers

| Layer | Run command | Validation command | Main output |
|---|---|---|---|
| Deterministic review scaffold | `python3 rac/scripts/run_mock_pipeline.py --all-eval` | `python3 rac/scripts/validate_mock_pipeline.py` | Mock review-state and report outputs in `rac/outputs/` |
| Local evidence resolver | `python3 rac/scripts/run_local_evidence_resolver.py --all-eval` | `python3 rac/scripts/validate_local_evidence_resolver.py` | Source-grounded evidence packets |
| Grounded deterministic pipeline | `python3 rac/scripts/run_grounded_pipeline.py --all-eval` | `python3 rac/scripts/validate_grounded_pipeline.py` | `rac/outputs/grounded_*.json` and `rac/outputs/grounded_*.md` |
| Grounded quality gate | Generated with the grounded pipeline | `python3 rac/scripts/validate_grounded_quality_gate.py` | `grounded_quality_summary.json` and `grounded_quality_summary.md` |

The mock pipeline establishes the review-state contract. The local
resolver connects factors to repository evidence. The grounded pipeline
combines both layers into inspectable reports. The quality gate checks
whether those reports preserve the required evidence structure.

## Grounded Report Contract

The grounded quality gate checks that every generated report contains:

- the reviewed question;
- expanded factors and factor weights;
- competing hypotheses;
- critic findings;
- fact-check output;
- local source paths;
- source line ranges;
- local evidence snippets;
- explicit limitations;
- evidence-routing status;
- no missing source files;
- no forbidden positive overclaims.

This contract prevents the RAC layer from becoming a loose prompt
wrapper. A report must show what evidence it used, where that evidence
came from, and what remains unresolved.

## Reviewer Cases

The reviewer-facing cases are indexed in
[`DEMO_INDEX.md`](DEMO_INDEX.md).

They cover questions such as:

### Store A attribution review

Can Store A's April change be attributed to search exposure?

The review should consider search exposure, entry conversion, order
conversion, activity involvement, and competing explanations rather than
assigning the movement to one metric.

### Cross-store comparability review

Are Stores B-F directly comparable in March 2026?

The review should preserve the common reporting period and metric
definitions while identifying the additional question-specific evidence
required for pairwise comparability.

### Promotion-strategy review

What should be checked before changing a store's promotion strategy?

The review should identify relevant factors, retrieve available evidence,
record missing requirements, and distinguish diagnostic checks from a
strategy recommendation.

## Interpretation Boundary

The current RAC implementation supports an inspectable, deterministic
review over structured local evidence. It can expose relevant factors,
source use, competing explanations, evidence gaps, and report
limitations.

Causal inference, learned factor weights, live merchant-backend access,
automated pairwise comparability decisions, and autonomous operating
actions require additional evidence or implementation beyond the
current review contract.

## Future Extensions

The existing contracts can later be connected to a shared-state graph
workflow, model-assisted factor expansion, vector retrieval, or adaptive
factor weighting.

Those extensions should retain the current invariants:

- typed review state;
- factor-first evidence retrieval;
- source references for evidence claims;
- separation of direct and boundary evidence;
- explicit competing hypotheses;
- critique before final reporting;
- evidence-based confidence and limitation updates;
- refusal to fill missing evidence with unsupported claims.

A useful next experiment is a small sensitivity analysis over alternative
factor-weight and Evidence-Coverage Score settings, using the same
principle as the retail guardrail sensitivity analysis.
