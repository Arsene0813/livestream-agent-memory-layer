# Structured Reasoning Scaffold

This module is a structured reasoning scaffold for question decomposition, evidence routing, critique, and evidence-coverage and limitation update.

The `rac/` directory name is retained for path stability. In reviewer-facing wording, this should be read as a deterministic, source-grounded reasoning scaffold with explicit evidence and boundary checks.

Workflow:

question -> question analysis -> factor expansion -> factor weighting -> evidence routing -> hypothesis generation -> critique / contradiction -> fact checking -> evidence-coverage and limitation update -> final report

## Why this module exists

A normal LLM can produce a plausible answer without explicitly checking:

  * which factors are relevant,
  * which assumptions are unsupported,
  * which evidence was actually used,
  * which competing explanations remain possible,
  * which conclusion should be downgraded because evidence is incomplete.

This module turns answer generation into an evidence-grounded review workflow.

## Current deterministic review scaffold

The current implementation is intentionally narrow and conservative. It defines:

* a shared review-state schema,
* deterministic question analysis and factor expansion,
* deterministic factor-weight buckets,
* local file-based evidence routing,
* competing-hypothesis templates,
* critique and fact-check stages,
* evidence-coverage scoring,
* explicit limitation reporting.

The current implementation does not claim to implement autonomous world modeling, live Meituan backend access, neural-network weight updates, completed pairwise comparability, or automated operating decisions.

## Future graph-based extension

A future implementation could connect these contracts to a graph workflow, such as a LangGraph-style shared-state node design. That would be an extension of the current deterministic scaffold, not a capability claimed by the current implementation.

The future version should still keep the current evidence-boundary discipline:

* carry typed state between stages,
* require source references for evidence claims,
* keep boundary evidence separate from direct metric evidence,
* report missing evidence instead of filling gaps with unsupported claims,
* avoid causal or operating-decision claims when the evidence does not support them.

<!-- RAC_FUTURE_FACTOR_LOOP_START -->
## Future extension: dynamic factor weighting

The current RAC implementation is deterministic and file-grounded. A future version could make the reasoning loop more adaptive by assigning different weights to factors depending on the question type and available evidence.

For example, a search-entry comparison may prioritize search exposure, search entry rate, and ranking context. A promotion-transfer question should give more weight to activity involvement, activity-cost ratio, campaign timing evidence, competitor context, and repeated-window evidence.

Any future loop should preserve the current boundary discipline:

* expand factors before retrieving evidence,
* retrieve evidence by factor instead of searching for a ready-made answer,
* generate competing hypotheses when multiple explanations are plausible,
* critique unsupported assumptions before writing the final answer,
* update confidence and limitations only when the evidence supports the change.

This planned extension should keep the same evidence-boundary discipline as the current implementation: strategy decisions and causal claims still require stronger evidence than the current scaffold provides.
<!-- RAC_FUTURE_FACTOR_LOOP_END -->

## Factor Weight Generation

The current RAC scaffold uses deterministic heuristic factor weights.

The weights are generated in `rac/src/mock_pipeline.py` by fixed factor-id buckets:

```text
high-priority review factors -> 0.85
medium-priority review factors -> 0.72
default relevant factors -> 0.60
```

The bucket assignment is explicit:

```text
high:
promotion_intensity
activity_intensity
order_conversion
refund_pressure
sku_margin_structure
evidence_packets
belief_records
retrieval_trace

medium:
search_exposure
entry_conversion
same_reporting_period
store_type
order_volume
transaction_amount
payment_conversion
typed_memory
hypotheses
confidence
limitations
active_state_filtering

default:
any relevant factor not listed in high or medium
```

These weights are review-priority weights. They are not learned from data, not calculated directly from observed metric tables, not probabilities, and not optimized business thresholds.

Their purpose is to make the review path explicit: higher-weight factors are treated as more central when the scaffold tries to prevent overconfident causal, comparability, or strategy-transfer claims.

The generated Grounded RAC Reports include a `How Factor Weights Are Generated` subsection so a reviewer can see the bucket rule, weight value, factor membership, and limitations directly in the report.

## Design boundary

This module should not claim:

  * that the system has a true Bayesian posterior,
  * that factor weights are mathematically learned probabilities,
  * that it has live access to Meituan backend data,
  * that Demo 2 already implements a pairwise comparability gate,
  * that the system can fully infer causality from observational store metrics.

The correct claim is narrower:

This scaffold makes LLM-assisted reasoning more traceable by decomposing a question into relevant factors, retrieving evidence by factor, generating competing hypotheses, checking unsupported assumptions, and producing grounded reports with a formula-based evidence-coverage score and mock reports with scenario-template confidence.
## Example use cases

### Retail operations diagnostic

Question:

Can Store A's April growth be attributed to search exposure?

Expected behavior:

- consider search exposure,
- consider entry conversion,
- consider order conversion,
- consider promotion intensity,
- consider refund pressure,
- avoid attributing growth to search alone.

### Cross-store comparability judgment

Question:

Are Stores B-F directly comparable in March 2026?

Expected behavior:

- recognize the same reporting period,
- preserve backend metric definitions,
- identify missing pairwise comparability gates,
- avoid claiming robust cross-store causal comparison.

### Strategic recommendation

Question:

What should be checked before changing promotions for a store?

Expected behavior:

- identify relevant operating factors,
- retrieve available evidence,
- generate competing explanations,
- state missing evidence,
- recommend checks rather than overconfident actions.

## Deterministic mock pipeline

The deterministic mock pipeline is the current runnable implementation of the structured reasoning workflow.

It does not call an LLM, Qdrant, Ollama, OpenAI, or any live backend service.

Its purpose is to prove that the system contract can run end-to-end:

question
-> question analysis
-> factor expansion
-> factor weighting
-> evidence routing
-> hypothesis generation
-> critique
-> fact checking
-> review-state update
-> final report

Run all mock evaluation cases:

python3 rac/scripts/run_mock_pipeline.py --all-eval

Validate the deterministic pipeline:

python3 rac/scripts/validate_mock_pipeline.py

The generated reports are written to:

rac/outputs/

This stage is intentionally conservative. The mock pipeline should not be presented as real LLM reasoning or real retrieval. It is a stable execution scaffold before any future LangGraph-style or retrieval-backed implementation.

## Local evidence resolver

The local evidence resolver is the first grounding layer for the RAC workflow.

It upgrades placeholder evidence packets into local, source-grounded evidence packets by reading project files and matching factor-specific keywords.

The resolver does not call an LLM, Qdrant, Ollama, OpenAI, or any live backend service.

It performs:

- source path existence checks,
- factor-specific keyword matching,
- local evidence snippet extraction,
- resolver limitation reporting.

Run the resolver for all evaluation cases:

python3 rac/scripts/run_local_evidence_resolver.py --all-eval

Validate the resolver:

python3 rac/scripts/validate_local_evidence_resolver.py

Generated outputs:

rac/outputs/local_evidence_resolver_all_cases.json
rac/outputs/local_evidence_resolver_sample.json

This stage is still deterministic. It should not be presented as semantic retrieval or true reasoning. Its purpose is to prove that the scaffold can ground factor-specific evidence in local project files before adding LangGraph, LLM calls, or vector retrieval.

## Grounded deterministic pipeline

The grounded deterministic pipeline connects the mock RAC workflow to the local evidence resolver.

It runs:

question
-> mock RAC pipeline
-> local evidence resolver
-> grounded evidence rows
-> final report with local evidence snippets

Run all grounded evaluation cases:

python3 rac/scripts/run_grounded_pipeline.py --all-eval

Validate the grounded pipeline:

python3 rac/scripts/validate_grounded_pipeline.py

Generated outputs are written to:

rac/outputs/grounded_*.json
rac/outputs/grounded_*.md

This stage still does not call an LLM, vector database, or live backend service.

The purpose is to prove that the scaffold can produce factor-aware reports grounded in local project evidence before adding LangGraph, model calls, or Qdrant retrieval.

## Grounded quality gate

The grounded quality gate validates the generated grounded RAC reports.

It checks that each report includes:

- required report sections,
- factor weights,
- competing hypotheses,
- critic findings,
- fact check output,
- local source paths,
- line ranges,
- local evidence snippets,
- explicit limitations,
- zero missing source files,
- no forbidden positive overclaims.

Run the quality gate:

python3 rac/scripts/validate_grounded_quality_gate.py

Generated outputs:

rac/outputs/grounded_quality_summary.json
rac/outputs/grounded_quality_summary.md

This quality gate is intentionally strict. It is designed to prevent the structured reasoning scaffold from becoming a loose prompt wrapper. A grounded report must show what evidence it used, where the evidence came from, what the system cannot conclude, and whether any source-grounding problem exists.

## Reviewer demo index

For a reviewer-facing overview of the RAC module, start here:

rac/DEMO_INDEX.md

This page explains what the module does, how to run the grounded demos, which outputs to inspect, and what the current implementation does not claim.

## Source-aware and boundary-aware grounding

The RAC grounding layer has been hardened so that the cross-store comparability demo no longer relies on generic fallback snippets for key factors.

Current quality-gate result:

- Total grounded packets: 32
- Keyword matched packets: 29
- Boundary matched packets: 2
- Fallback packets: 1
- Missing source files: 0

For rac_cross_store_comparability_001:

- order_volume, transaction_amount, refund_pressure, and sku_structure are routed to retail_ops/outputs/demo2_cross_store_comparability_output.csv.
- competition and repeated_reporting_windows are routed to retail_ops/COMPARABILITY_GATE_V0.md as boundary_evidence.
- The report does not claim that pairwise comparability is implemented.
- Pairwise comparability remains future work.

This distinction matters because RAC should not pretend that missing evidence exists. When a factor is required but not currently structured, the system should ground that factor in an explicit boundary source rather than using a generic fallback snippet.


## Score Explainability

Grounded RAC reports use a formula-based Evidence-Coverage Score.

Formula: evidence_coverage_score = 0.45 * direct_evidence_rate + 0.25 * supported_or_boundary_rate + 0.15 * no_missing_source_file_score + 0.15 * no_fallback_score.

This score is calculated from local evidence-routing status. It is not a learned probability, Bayesian posterior, causal confidence score, or business-success probability.

Weight rationale for the grounded score:

| Component | Weight | Why |
|---|---:|---|
| `direct_evidence_rate` | 0.45 | Highest priority because actual local evidence should matter more than boundary-only evidence. |
| `supported_or_boundary_rate` | 0.25 | Boundary evidence is valuable because it explicitly records missing requirements instead of hiding them. |
| `no_missing_source_file_score` | 0.15 | Source files must exist, but this is a basic traceability check rather than evidence strength. |
| `no_fallback_score` | 0.15 | Fallback evidence is acceptable as a warning, but should reduce confidence in coverage. |

These weights are fixed prototype heuristics, not learned parameters, optimized thresholds, calibrated probabilities, or business-performance predictors.

Future work should run a small sensitivity check over alternative weight settings, similar to the Demo 2 guardrail sensitivity check.

Mock RAC reports use Scenario-Template Confidence instead. That value is assigned by deterministic question-type templates and is kept only to show how the mock scaffold carries a review-state value.
