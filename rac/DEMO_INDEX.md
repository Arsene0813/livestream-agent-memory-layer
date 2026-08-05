# Factor-Aware Grounded Review Demo Index

This page is the reviewer-facing entry point for the `rac/`
implementation. RAC is an important factor-aware grounded review layer
over the structured retail evidence.

It makes factor selection, source routing, unavailable evidence,
competing explanations, critique, rule-based claim and definition checks, and evidence-routing coverage
inspectable before a report is accepted.

The implemented workflow is:

```text
question
-> question analysis
-> factor expansion
-> factor weighting
-> source-aware local evidence grounding
-> boundary evidence for unavailable requirements
-> hypothesis generation
-> critique
-> rule-based claim and definition check
-> review-state update
-> grounded final report
-> report-contract quality gate
```

## 30-Second Summary

RAC turns a retail decision question into an inspectable review path.
Each case records which factors were considered, where the supporting
evidence came from, which requirements remain unavailable, and how the
final judgment was bounded.

The implemented review process:

1. Identify relevant factors.
2. Assign interpretable factor weights.
3. Route each factor to local evidence.
4. Distinguish quantitative evidence from boundary evidence.
5. Record source paths, structured-record locators for CSV evidence, source-line pointers for text evidence, and canonical evidence fields.
6. Generate competing hypotheses.
7. Critique weak claims.
8. Check unsupported claims and definition conflicts.
9. Produce a grounded report with a routing coverage score,
   scenario-template confidence labels, and explicit limitations.
10. Validate the report through a report-contract quality gate.

The current implementation is deterministic and source-aware. It uses
local project files, explicit boundary evidence, a shared review-state
contract, and fixed evaluation cases.

## Current Implementation Status

| Component | Status | Current role |
|---|---|---|
| Structured review contracts | Implemented | `rac/prompts/`, `rac/schemas/`, and `rac/eval/` define the review inputs, shared state, and fixed cases. |
| Deterministic review baseline | Implemented | `rac/src/mock_pipeline.py` establishes the typed review-state and report contracts. |
| Local evidence resolver | Implemented | `rac/src/local_evidence_resolver.py` routes factors to structured CSV records, local text evidence, or explicit boundary evidence. |
| Grounded RAC pipeline | Implemented | `rac/src/grounded_pipeline.py` generates source-aware JSON and Markdown reports. |
| Report-contract quality gate | Implemented | `rac/scripts/validate_grounded_quality_gate.py` checks report structure, evidence references, factor evidence status, limitations, and selected claim boundaries. |
| Pairwise comparability gate | Deferred evidence contract | Requires stronger repeated-window, operating-condition, and market-context evidence before a question-specific pairwise decision can be supported. |

Files beginning with `grounded_rac_` are the reviewer-facing evidence
reports. Files beginning with `rac_` are deterministic mock fixtures used
to verify the shared review-state and report contracts.

## How To Run

Run all grounded RAC demo cases:

  python3 rac/scripts/run_grounded_pipeline.py --all-eval

Validate the grounded reports:

  python3 rac/scripts/validate_grounded_quality_gate.py

Expected quality-gate result:

  [OK] RAC report-contract quality gate passed
  [OK] Cases checked: 4
  [OK] Total grounded packets: 30
  [OK] Record matched packets: 5
  [OK] Keyword matched packets: 22
  [OK] Boundary matched packets: 3
  [OK] Fallback packets: 0
  [OK] Missing source files: 0

## Demo Cases

| Case | Question | What It Demonstrates | Grounded Report |
|---|---|---|---|
| `rac_store_a_attribution_001` | Can Store A's March-to-April increases in transaction amount and transaction orders be attributed to search exposure alone? | Grounds search exposure, entry conversion, order conversion, promotion intensity, and transaction orders to the March-April Store A CSV records while rejecting single-cause attribution. | [Store A attribution report](outputs/grounded_rac_store_a_attribution_001.md) |
| `rac_cross_store_comparability_001` | Are Stores B-F directly comparable in March 2026? | Routes available quantitative evidence to Demo 2 sources and unavailable pairwise requirements to explicit boundary evidence. | [Cross-store boundary report](outputs/grounded_rac_cross_store_comparability_001.md) |
| `rac_promotion_strategy_001` | What should be checked before changing promotions for a store? | Routes available transaction, cost, and conversion evidence while retaining required SKU-level margin and competitor context as unresolved requirements. | [Promotion-strategy report](outputs/grounded_rac_promotion_strategy_001.md) |
| `rac_system_design_001` | How should RAC connect to the existing memory layer? | Shows how typed memory records feed a factor-aware grounded review path. | [System-design report](outputs/grounded_rac_system_design_001.md) |

## Cross-Store Grounding Hardening

The cross-store case uses factor-specific source requirements for quantitative, descriptive, and boundary evidence.

Current result for rac_cross_store_comparability_001:

- Total evidence packets: 9
- Keyword matched packets: 8
- Boundary matched packets: 1
- Fallback packets: 0
- Missing source files: 0

Required source routing:

| Factor | Required Source | Grounding Role |
|---|---|---|
| order_volume | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` | quantitative_evidence |
| transaction_amount | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` | quantitative_evidence |
| repeated_reporting_windows | `retail_ops/outputs/repeated_window_panel_summary_output.csv` | quantitative_evidence |
| competition | `retail_ops/COMPARABILITY_GATE_V0.md` | boundary_evidence |

The implemented scope is a factor-routing audit. Current B-F repeated-window panel evidence covers 2026-02 to 2026-04. Broader stability across additional stores, months, activity conditions, and market contexts remains required for a future pairwise comparability gate.

## Recommended Review Order

For a quick review, read these files in order:

1. [RAC demo index](DEMO_INDEX.md)
2. [Store A attribution-boundary report](outputs/grounded_rac_store_a_attribution_001.md)
3. [Cross-store comparability-boundary report](outputs/grounded_rac_cross_store_comparability_001.md)
4. [Promotion-review report](outputs/grounded_rac_promotion_strategy_001.md)
5. [Report-contract quality summary](outputs/grounded_quality_summary.md)
6. [RAC implementation guide](README.md)

For code review, inspect:

1. [`mock_pipeline.py`](src/mock_pipeline.py)
2. [`local_evidence_resolver.py`](src/local_evidence_resolver.py)
3. [`store_a_csv_grounding.py`](src/store_a_csv_grounding.py)
4. [`grounded_pipeline.py`](src/grounded_pipeline.py)
5. [`validate_grounded_quality_gate.py`](scripts/validate_grounded_quality_gate.py)

## What The Grounded Reports Show

Each grounded report includes:

- a direct answer;
- the detected question type;
- factor weights and their interpretation limits;
- local evidence source paths and grounding roles;
- structured-record locators for CSV evidence and source-line audit pointers for text evidence;
- canonical evidence fields;
- competing hypotheses with scenario-template confidence labels;
- critic findings;
- claim and definition-check status;
- a bounded final judgment;
- a routing coverage score and its calculation inputs;
- explicit limitations;
- a review-state update.

This structure keeps the evidence path, unresolved requirements, and
interpretation boundary visible in the final report.

## Review Contract

A reviewer can assess each grounded output at three linked levels:

1. **Factor coverage:** whether the question was decomposed into the factors needed for the current decision.
2. **Evidence routing:** whether each factor was linked to an appropriate local source or recorded as explicit boundary evidence.
3. **Judgment boundary:** whether the final judgment remains within the evidence, definitions, entity scope, and reporting period available to the case.

Each evidence record retains its source path, grounding role, and canonical evidence fields. Structured CSV evidence records the selected row keys and values; text evidence records a local line pointer. Missing requirements remain visible in the review trace rather than being replaced by an unsupported conclusion.

## Current Evidence and Integration Boundary

The current implemented scope includes:

- deterministic question classification and factor expansion;
- fixed and interpretable factor-weight buckets;
- factor-specific local evidence routing;
- explicit boundary evidence for unavailable requirements;
- competing hypothesis templates;
- critique and rule-based claim and definition-check records;
- JSON Schema and cross-record state validation;
- fixed evaluation cases and generated quality summaries.

The local evidence resolver uses deterministic matching over project
files. Factor weights and scenario-template confidence labels are
review-state values rather than learned probabilities or estimated
business outcomes.

For pairwise cross-store decisions, the next evidence contract requires
defined thresholds, repeated reporting windows, and additional operating
and market-context evidence. Pairwise comparability remains future work.

## Possible Follow-up Experiments

These are planned extensions rather than part of the current
reviewer-facing implementation. Each one is tied to a concrete
evaluation question.

1. **Evidence robustness:** test paraphrases, hard negatives, entity and
   period preservation, missing-source behavior, cross-store period
   mismatches, and sensitivity to alternative routing coverage weights.
2. **Model-assisted review:** compare model-assisted factor expansion,
   competing hypotheses, critique, and report synthesis against the
   deterministic baseline.
3. **Embedding-based or hybrid retrieval:** evaluate whether retrieval
   improves wording coverage without weakening source traceability,
   entity scope, period scope, or hard-negative behavior.
4. **LangGraph-style orchestration:** introduce conditional
   re-retrieval, repeated critique, recoverable branches, or
   human-review checkpoints only when those branches are needed and can
   be tested against the sequential baseline.
5. **Live Meituan backend integration:** treat live integration as a
   separate data-access and lineage task requiring source-field mapping,
   period handling, schema-drift checks, access control, anonymization,
   and lineage capture.

## Score Explainability Note

Grounded reports use Routing coverage score. It summarizes record- or keyword-matched local routes, explicit boundary routes, missing-source status, and fallback status.

Mock reports use Scenario-Template Confidence. This is a deterministic template value, not a calculated routing coverage score.

The grounded score weights are fixed prototype heuristics:

- `0.45` for record- or keyword-matched local routes;
- `0.25` for supported-or-boundary coverage;
- `0.15` for source-file traceability;
- `0.15` for fallback avoidance.

They are not learned, calibrated, or presented as optimized decision weights. Sensitivity testing remains a planned robustness check.
