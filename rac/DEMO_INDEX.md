# Structured Reasoning Scaffold Demo Index

This page is the reviewer-facing entry point for the `rac/` implementation folder.

The module is a structured reasoning scaffold for question decomposition, evidence routing, critique, and evidence-coverage update. It is not a free-form multi-agent chatroom and should not be presented as autonomous cognition.

It forces an answer to pass through:

```text
question
-> question analysis
-> factor expansion
-> factor weighting
-> source-aware local evidence grounding
-> boundary evidence for unavailable requirements
-> hypothesis generation
-> critique
-> fact checking
-> evidence-coverage update
-> grounded final report
-> quality gate
```

## 30-Second Summary

Normal LLM answers can be fluent but weakly grounded. They may skip relevant factors, over-attribute causality, ignore missing evidence, or hide uncertainty.

This scaffold turns answer generation into a staged review process:

1. Identify relevant factors.
2. Assign interpretable factor weights.
3. Route each factor to local evidence.
4. Distinguish quantitative evidence from boundary evidence.
5. Extract local evidence snippets.
6. Generate competing hypotheses.
7. Critique weak claims.
8. Check unsupported claims.
9. Output a grounded report with confidence and limitations.
10. Validate the report through a quality gate.

The current implementation is deterministic. It does not call an LLM, Qdrant, Ollama, OpenAI, or a live Meituan backend service.

## Current Implementation Status

| Stage | Status | Main Files |
|---|---|---|
| Structured reasoning scaffold | Done | rac/prompts/, rac/schemas/, rac/eval/ |
| Deterministic mock pipeline | Done | rac/src/mock_pipeline.py |
| Local evidence resolver | Done | rac/src/local_evidence_resolver.py |
| Source-aware cross-store grounding | Done | rac/src/local_evidence_resolver.py |
| Grounded RAC pipeline | Done | rac/src/grounded_pipeline.py |
| Grounded quality gate | Done | rac/scripts/validate_grounded_quality_gate.py |
| LangGraph orchestration | Future work | Not implemented |
| LLM-based factor generation | Future work | Not implemented |
| Qdrant / vector retrieval integration | Future work | Not implemented |
| Pairwise comparability gate | Future work | Not implemented |
| Live Meituan backend integration | Not claimed | Not implemented |

## How To Run

Run all grounded RAC demo cases:

  python3 rac/scripts/run_grounded_pipeline.py --all-eval

Validate the grounded reports:

  python3 rac/scripts/validate_grounded_quality_gate.py

Expected quality-gate result:

  [OK] RAC grounded quality gate passed
  [OK] Cases checked: 4
  [OK] Total grounded packets: 29
  [OK] Keyword matched packets: 27
  [OK] Boundary matched packets: 2
  [OK] Fallback packets: 0
  [OK] Missing source files: 0

## Demo Cases

| Case | Question | What It Demonstrates | Grounded Report |
|---|---|---|---|
| rac_store_a_attribution_001 | Can Store A's April growth be attributed to search exposure? | Avoids single-cause attribution and considers traffic, conversion, promotion, SKU context, and evidence limits. | rac/outputs/grounded_rac_store_a_attribution_001.md |
| rac_cross_store_comparability_001 | Are Stores B-F directly comparable in March 2026? | Routes quantitative factors to Demo 2 output evidence and routes unavailable comparability requirements to explicit boundary evidence. It does not implement a pairwise comparability gate. | rac/outputs/grounded_rac_cross_store_comparability_001.md |
| rac_promotion_strategy_001 | What should be checked before changing promotions for a store? | Prevents action recommendations based only on transaction amount and forces cost, conversion, margin, and competitor checks. | rac/outputs/grounded_rac_promotion_strategy_001.md |
| rac_system_design_001 | How should the RAC system be connected to the existing memory layer? | Shows how the scaffold can sit above typed memory as a reasoning layer instead of replacing existing endpoints. | rac/outputs/grounded_rac_system_design_001.md |

## Cross-Store Grounding Hardening

The cross-store structured-reasoning case was hardened to avoid weak generic fallback evidence.

Current result for rac_cross_store_comparability_001:

- Total evidence packets: 9
- Keyword matched packets: 7
- Boundary matched packets: 2
- Fallback packets: 0
- Missing source files: 0

Required source routing:

| Factor | Required Source | Grounding Role |
|---|---|---|
| order_volume | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` | quantitative_evidence |
| transaction_amount | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` | quantitative_evidence |
| current_b_f_repeated_window_panel | `retail_ops/outputs/store_period_panel_coverage_output.csv` and `retail_ops/outputs/repeated_window_panel_summary_output.csv` | descriptive_evidence |
| competition | `retail_ops/COMPARABILITY_GATE_V0.md` | boundary_evidence |
| broader_repeated_window_stability | `retail_ops/COMPARABILITY_GATE_V0.md` | boundary_evidence |

This is not a completed comparability engine. The grounded report should be read as a factor-routing audit: current B-F repeated-window panel evidence exists for 2026-02 to 2026-04, while broader repeated-window stability across more stores, months, activity conditions, and market contexts remains future evidence for a pairwise comparability gate.

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

- direct answer,
- question type,
- factor weights,
- local evidence source paths,
- grounding roles,
- line ranges,
- matched terms,
- local evidence snippets,
- competing hypotheses,
- critic findings,
- fact-check status,
- final judgment,
- confidence,
- limitations,
- review-state update.

This structure is designed to make the reasoning trace visible instead of hiding it inside a single fluent answer.

## Why This Is Different From Ordinary RAG

Ordinary RAG often works like this:

  question -> retrieve similar chunks -> generate answer

This scaffold works like this:

  question -> factors -> factor-specific evidence -> boundary evidence where evidence is unavailable -> hypotheses -> critique -> fact check -> grounded report

The difference is that retrieval is not treated as a generic context dump. Evidence is routed by factor and then carried into the final report with source paths, grounding roles, line ranges, and snippets.

## What This Module Does Not Claim

This module does not claim:

- live Meituan backend access,
- true autonomous world modeling,
- neural-network weight updates,
- true Bayesian posterior estimation,
- completed LangGraph orchestration,
- completed Qdrant integration,
- completed pairwise comparability gate,
- causal proof from observational store metrics,
- final operational recommendations without margin and competitor checks.

The correct claim is narrower:

The scaffold makes LLM-assisted decision support more traceable by decomposing a question into relevant factors, grounding each factor in local evidence or explicit boundary evidence, generating competing hypotheses, critiquing weak assumptions, and producing grounded reports with a formula-based evidence-coverage score and mock reports with scenario-template confidence.

## Current Limitations

The current implementation is intentionally conservative.

Limitations:

- factor generation is deterministic,
- factor weights are interpretable relevance estimates, not learned probabilities,
- evidence grounding uses local keyword matching,
- boundary evidence is used when structured evidence is not currently available,
- no semantic embedding retrieval is used yet,
- no LLM calls are used yet,
- no live backend data is retrieved,
- no causal inference engine is implemented,
- pairwise comparability remains future work.

## Next Possible Steps

Recommended next steps:

1. Convert deterministic nodes into a LangGraph workflow.
2. Add LLM-based factor expansion and hypothesis generation behind strict schemas.
3. Replace local keyword matching with Qdrant-backed factor-specific retrieval.
4. Add stronger eval cases for overclaim prevention and evidence coverage.
5. Implement pairwise comparability gates only after defining thresholds, repeated windows, and required external/contextual evidence.


## Score Explainability Note

Grounded reports use Evidence-Coverage Score. This is calculated from evidence-routing coverage, boundary coverage, missing-source status, and fallback status.

Mock reports use Scenario-Template Confidence. This is a deterministic template value, not a calculated evidence-coverage score.

The grounded score weights are fixed prototype heuristics:

- `0.45` for direct local evidence;
- `0.25` for supported-or-boundary coverage;
- `0.15` for source-file traceability;
- `0.15` for fallback avoidance.

These weights are not learned or optimized. A future version should run sensitivity checks over alternative weight settings.
