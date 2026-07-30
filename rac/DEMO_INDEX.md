# Factor-Aware Grounded Review Demo Index

This page is the reviewer-facing entry point for the `rac/`
implementation. RAC is an important factor-aware grounded review layer
over the structured retail evidence.

It makes factor selection, source routing, unavailable evidence,
competing explanations, critique, rule-based claim and definition checks, and evidence coverage
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
5. Record source paths, source-line audit pointers, and evidence fields.
6. Generate competing hypotheses.
7. Critique weak claims.
8. Check unsupported claims and definition conflicts.
9. Produce a grounded report with an evidence-coverage score,
   scenario-template confidence labels, and explicit limitations.
10. Validate the report through a report-contract quality gate.

The current implementation is deterministic and source-aware. It uses
local project files, explicit boundary evidence, a shared review-state
contract, and fixed evaluation cases.

## Current Implementation Status

| Component | Current Status | Evidence or Next Validation |
|---|---|---|
| Structured review contracts | Implemented | `rac/prompts/`, `rac/schemas/`, and `rac/eval/` define the current review inputs, state, and fixed cases. |
| Deterministic review pipeline | Implemented | `rac/src/mock_pipeline.py` establishes the typed review-state and report contracts. |
| Local evidence resolver | Implemented | `rac/src/local_evidence_resolver.py` routes factors to local source snippets or explicit boundary evidence. |
| Grounded RAC pipeline | Implemented | `rac/src/grounded_pipeline.py` generates source-aware JSON and Markdown reports. |
| Report-contract quality gate | Implemented | `rac/scripts/validate_grounded_quality_gate.py` validates report structure, evidence references, limitations, and selected claim boundaries. |
| LangGraph-style orchestration | Planned experiment | Evaluate when conditional re-retrieval, repeated critique, recoverable branches, or human-review checkpoints make the sequential path difficult to inspect. |
| Model-assisted review | Planned experiment | Compare model-assisted factor expansion, competing hypotheses, critique, and report synthesis against the deterministic baseline. |
| Embedding-based or hybrid retrieval | Planned experiment | Test wording variation, hard-negative behavior, entity and period preservation, threshold sensitivity, and source traceability. |
| Pairwise comparability gate | Planned evidence contract | Requires defined thresholds, broader repeated-window evidence, and additional market and operating context. |
| Live Meituan backend integration | Future data integration | Requires source-field mapping, period handling, schema-drift checks, access control, anonymization, and lineage capture. |

## How To Run

Run all grounded RAC demo cases:

  python3 rac/scripts/run_grounded_pipeline.py --all-eval

Validate the grounded reports:

  python3 rac/scripts/validate_grounded_quality_gate.py

Expected quality-gate result:

  [OK] RAC report-contract quality gate passed
  [OK] Cases checked: 4
  [OK] Total grounded packets: 30
  [OK] Keyword matched packets: 27
  [OK] Boundary matched packets: 3
  [OK] Fallback packets: 0
  [OK] Missing source files: 0

## Demo Cases

| Case | Question | What It Demonstrates | Grounded Report |
|---|---|---|---|
| rac_store_a_attribution_001 | Can Store A's April growth be attributed to search exposure? | Avoids single-cause attribution and considers traffic, conversion, promotion, SKU context, and evidence limits. | rac/outputs/grounded_rac_store_a_attribution_001.md |
| rac_cross_store_comparability_001 | Are Stores B-F directly comparable in March 2026? | Routes quantitative factors to Demo 2 output evidence and routes unavailable comparability requirements to explicit boundary evidence. It does not implement a pairwise comparability gate. | rac/outputs/grounded_rac_cross_store_comparability_001.md |
| rac_promotion_strategy_001 | What should be checked before changing promotions for a store? | Routes available transaction, cost, and conversion evidence while retaining margin and competitor context as unresolved requirements. | rac/outputs/grounded_rac_promotion_strategy_001.md |
| rac_system_design_001 | How should the RAC system be connected to the existing memory layer? | Shows how RAC uses typed memory records as inputs to a factor-aware grounded review path. | rac/outputs/grounded_rac_system_design_001.md |

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

1. rac/DEMO_INDEX.md
2. rac/outputs/grounded_rac_store_a_attribution_001.md
3. rac/outputs/grounded_rac_cross_store_comparability_001.md
4. rac/outputs/grounded_quality_summary.md
5. rac/README.md

For code review, inspect:

1. rac/src/mock_pipeline.py
2. rac/src/local_evidence_resolver.py
3. rac/src/grounded_pipeline.py
4. rac/scripts/validate_grounded_quality_gate.py

## What The Grounded Reports Show

Each grounded report includes:

- a direct answer;
- the detected question type;
- factor weights and their interpretation limits;
- local evidence source paths and grounding roles;
- source-line audit pointers;
- canonical evidence fields;
- competing hypotheses with scenario-template confidence labels;
- critic findings;
- claim and definition-check status;
- a bounded final judgment;
- an evidence-coverage score and its calculation inputs;
- explicit limitations;
- a review-state update.

This structure keeps the evidence path, unresolved requirements, and
interpretation boundary visible in the final report.

## Review Contract

A reviewer can assess each grounded output at three linked levels:

1. **Factor coverage:** whether the question was decomposed into the factors needed for the current decision.
2. **Evidence routing:** whether each factor was linked to an appropriate local source or recorded as explicit boundary evidence.
3. **Judgment boundary:** whether the final judgment remains within the evidence, definitions, entity scope, and reporting period available to the case.

Each evidence record retains its source path, grounding role, source-line audit pointer, and evidence fields. Missing requirements remain visible in the review trace rather than being replaced by an unsupported conclusion.

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

## Next Experiments

The next experiments should test the stability of the implemented review
logic:

1. Run sensitivity checks over alternative evidence-coverage weights.
2. Test factor routing against paraphrased versions of each evaluation
   question.
3. Remove selected evidence sources and verify that missing requirements
   appear as boundary evidence.
4. Add cross-store stress cases covering period mismatch, missing fields,
   metadata drift, and incomplete repeated windows.
5. Test whether small changes in factor templates alter the final
   judgment or only the review trace.

## Score Explainability Note

Grounded reports use Evidence-Coverage Score. This is calculated from evidence-routing coverage, boundary coverage, missing-source status, and fallback status.

Mock reports use Scenario-Template Confidence. This is a deterministic template value, not a calculated evidence-coverage score.

The grounded score weights are fixed prototype heuristics:

- `0.45` for direct local evidence;
- `0.25` for supported-or-boundary coverage;
- `0.15` for source-file traceability;
- `0.15` for fallback avoidance.

These weights are not learned or optimized. A future version should run sensitivity checks over alternative weight settings.
