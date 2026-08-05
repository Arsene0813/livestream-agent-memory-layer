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

## Quick Start

Generate the reviewer-facing grounded reports:

```bash
python3 rac/scripts/run_grounded_pipeline.py --all-eval
```

Validate the generated reports and their evidence boundaries:

```bash
python3 rac/scripts/validate_grounded_pipeline.py
python3 rac/scripts/validate_grounded_quality_gate.py
```

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
-> rule-based claim and definition check
-> review-state update
-> grounded final report
-> report-contract quality gate
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
- critique and rule-based claim and definition-check stages;
- routing coverage scoring;
- explicit limitation updates;
- grounded Markdown and JSON reports;
- a deterministic report-contract quality gate.

The current runnable path uses local files and deterministic rules.
Graph orchestration, model calls, vector retrieval, and live
merchant-backend access are future integration options rather than
requirements for the present evidence contract.

Reviewer-facing terminology is deliberately narrow: the claim and definition check applies deterministic rules to selected unsupported claims and definition conflicts, while the report-contract quality gate validates report structure and selected evidence boundaries. Neither mechanism establishes that an operating conclusion is correct.

## Evidence Types

RAC distinguishes three evidence-routing outcomes.

| Evidence type | Meaning | Review treatment |
|---|---|---|
| Direct evidence | A local source contains factor-relevant evidence for the current question. | The report may use the evidence within its documented scope. |
| Boundary evidence | A local source explicitly records that a required field, gate, or condition is not implemented or not available. | The report records the missing requirement instead of inventing supporting evidence. |
| Fallback evidence | No sufficiently specific direct or boundary source was resolved. | Evidence coverage is reduced and the limitation remains visible. |

For the cross-store comparability case, the implemented February-April
2026 B-F panel provides direct repeated-window evidence for descriptive
review. Competition context and the requirements for a future pairwise gate remain
boundary evidence in `retail_ops/COMPARABILITY_GATE_V0.md`. This keeps
the available multi-period evidence separate from the stronger evidence
still required for stable pairwise comparability.

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

## Routing Coverage Score

Grounded RAC reports use a formula-based routing coverage score:

```text
evidence_coverage_score
= 0.45 * direct_evidence_rate
+ 0.25 * supported_or_boundary_rate
+ 0.15 * no_missing_source_file_score
+ 0.15 * no_fallback_score
```

| Component | Weight | Rationale |
|---|---:|---|
| `direct_evidence_rate` | 0.45 | Record- or keyword-matched local routes receive the highest weight. |
| `supported_or_boundary_rate` | 0.25 | Explicit boundary evidence is preferable to an unsupported inference. |
| `no_missing_source_file_score` | 0.15 | Existing source paths are required for traceability. |
| `no_fallback_score` | 0.15 | Unresolved fallback packets reduce evidence coverage. |

The score summarizes whether the current deterministic resolver found local
evidence or recorded an explicit boundary for requested evidence routes. It
does not measure evidence strength, conclusion correctness, business impact,
decision quality, or model confidence. It is not used to select or rank the
final judgment.

Mock reports separately carry Scenario-Template Confidence. That value
is assigned by deterministic question-type templates and is not merged
with the grounded routing coverage score.

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
| Report-contract quality gate | Generated with the grounded pipeline | `python3 rac/scripts/validate_grounded_quality_gate.py` | `grounded_quality_summary.json` and `grounded_quality_summary.md` |

The deterministic review pipeline establishes the review-state contract. The local
resolver connects factors to repository evidence. The grounded pipeline
combines both layers into inspectable reports. The quality gate checks
whether those reports preserve the required evidence structure.

## Grounded Report Contract

The report-contract quality gate checks that every generated report contains:

- the reviewed question;
- expanded factors and factor weights;
- competing hypotheses;
- critic findings;
- rule-based claim and definition-check output;
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

## Implementation Roadmap

The deterministic RAC path remains the reference baseline for later integrations. Future implementations should preserve the typed review state, evidence-source references, explicit limitations, critique stage, and report contract already used by the current pipeline.

### Embedding-Based or Hybrid Retrieval

The current retrieval experiments expose behavior under wording variation, hard-negative cases, entity and reporting-period mismatch, and ambiguous questions. A later experiment can compare the existing file-backed routing baseline with embedding-based or hybrid retrieval over the same evaluation cases.

The comparison should record expected-evidence recall, false routing, entity and period preservation, unsupported-question behavior, threshold sensitivity, and source traceability. Vector retrieval would remain an evidence-routing mechanism rather than a source of operating conclusions.

### Model-Assisted Review

Model calls can be evaluated for factor expansion, competing-hypothesis generation, critique drafting, unresolved-evidence identification, and grounded report synthesis.

Canonical field definitions, SQL-derived values, source paths, entity and period constraints, and deterministic positive-claim boundaries should remain externally controlled. Model-assisted outputs should be compared with the deterministic baseline for source use, unsupported claims, stability, preservation of limitations, and reviewer effort.

### LangGraph-Style Orchestration

The current review path can remain sequential while its state transitions are limited and directly inspectable. LangGraph-style orchestration becomes relevant when the workflow requires conditional re-retrieval, evidence-insufficiency branches, repeated critique, question-specific paths, recoverable failures, or human-review checkpoints.

A later graph implementation should preserve the existing typed review state and report contract. Its value should be evaluated through clearer state transitions and recoverable failure handling rather than framework adoption alone.

### Adaptive Factor Weighting

The current factor weights are explicit prototype heuristics. Later experiments can compare the fixed buckets with alternative rule-based settings or learned weighting methods.

Any adaptive approach should retain the factor set, the source of each weight, sensitivity results, and the distinction between review priority and business effect. Adaptive weights should not be interpreted as causal effects, business thresholds, or calibrated probabilities without separate evidence.

### Live Merchant-Backend Integration

Live merchant-backend access is a later data-integration path. It should be introduced after source-field mapping, reporting-period handling, schema drift, access control, anonymization, and lineage capture can be checked against the existing data dictionary and contract tests.

The live path should preserve the canonical field names and Chinese definitions governed by `retail_ops/data/DATA_DICTIONARY.md`.

### Next RAC Experiment

The next experiment is a controlled sensitivity analysis over alternative factor-weight and routing coverage score settings.

It should keep the same reviewer cases and evidence packets, vary one heuristic setting at a time, record changes in factor priority and final review state, and identify which outputs remain stable. The current deterministic settings should remain the reference condition.

All roadmap implementations should continue to preserve:

- typed review state;
- factor-first evidence routing;
- source references for evidence claims;
- separation of direct, boundary, and fallback evidence;
- explicit competing hypotheses;
- critique before final reporting;
- visible routing coverage and limitation updates;
- withholding of conclusions that exceed the available evidence.
