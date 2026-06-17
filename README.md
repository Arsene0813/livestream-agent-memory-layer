# Lifecycle-Aware AI Memory Layer for Retail Decision Support

Repository: `livestream-agent-memory-layer`

A local evidence-bounded decision-support prototype for multi-store Meituan instant-retail operations, built with SQL diagnostics, metric-boundary preservation, and lifecycle-aware retrieval.

The repository name is historical. The project began as a lifecycle-aware memory layer for commerce interaction, and the current implemented extension applies the same evidence-boundary design to Meituan instant-retail decision support.

## Core Research Question

This project grew from a real Meituan instant-retail operating problem.

The Meituan merchant backend provides detailed single-store metrics, but it is mainly designed for reviewing one store at a time. As store count increased, the harder problem became cross-store decision support: which store-period records can be compared, under what operating conditions, and what kind of operating judgment the available evidence can support.

For standardized instant-retail products, store competition is organized around one operating chain:

```text
being seen -> being entered -> being ordered -> being selected again / maintaining share
```

Promotion, subsidy, pricing, SKU arrangement, ranking position, and fulfillment stability are operating levers inside this chain. Their meaning depends on store state, local competition, activity intensity, product mix, and reporting-window alignment.


The goal is to build a more reliable evidence-based framework for multi-store operational analysis, so that future operating decisions can be made with clearer data boundaries as the business expands.

## Current Prototype Workflow

The current prototype follows one evidence path: Meituan backend metric evidence -> canonical field dictionary -> SQL diagnostic output -> generated retail memory facts -> boundary-preserving answer checks.

In practice, this means the project first preserves backend metric definitions, then uses SQL to structure selected store-period data, then converts diagnostic evidence into memory facts with source fields, observed values, source paths, and limitations. The final check is whether later answers stay inside the available evidence boundary.

The single source of truth for retail field names and metric meanings is:

- `retail_ops/data/DATA_DICTIONARY.md`

## Current Implemented Scope

The current retail decision-support path has three implemented evidence layers: Demo 1 for Store A month-over-month diagnosis, Demo 2 for selected Stores B-F under one March 2026 reporting window, and the repeated-window B-F panel across 2026-02, 2026-03, and 2026-04.

A future pairwise comparability gate would judge whether two store-period records can be compared for a specific operating question. Demo 2 keeps the historical `cross_store_comparability` path wording for reference stability, but the implemented output remains same-period B-F diagnostic evidence rather than a pairwise gate decision.

| Area | Current implementation | Current boundary |
| --- | --- | --- |
| Livestream memory layer | Typed product facts, overwrite control, soft deactivation, active-state retrieval, fallback/refusal, scenario evaluation. | Local prototype for lifecycle-aware memory behavior. |
| Data dictionary | Preserves Meituan-style backend metric meanings and canonical field names. | Manual normalization of selected backend evidence; field meanings follow `retail_ops/data/DATA_DICTIONARY.md`. |
| Retail Demo 1 | Store A month-over-month diagnostic across February, March, and April 2026. | Multi-metric interpretation rather than single-cause monthly explanation. |
| Retail Demo 2 | Same-period B-F diagnostic for March 2026. | Same-period diagnostic evidence with explicit interpretation limits before pairwise comparability rules. |
| Repeated-window panel | B-F coverage and descriptive summary across 2026-02, 2026-03, and 2026-04. | Descriptive repeated-window evidence before future pairwise comparability rules. |
| Memory facts | Converts diagnostic outputs into source-bounded facts with observed values, source fields, source paths, confidence labels, and limitations. | File-backed evidence records for the implemented demos; not a replacement for raw backend evidence. |
| Answer-boundary checks | Tests whether answers stay within entity, period, metric-definition, source, and interpretation boundaries. | Scenario-based checks tied to the current evidence path; not broad LLM robustness tests. |
| RAC scaffold | Provides deterministic factor expansion, evidence routing, critique, fact checking, evidence-coverage update, and grounded report generation over local project evidence. | Local-evidence scaffold for reviewability; not live backend ingestion, autonomous cognition, or operating-decision automation. |

## Key Design Principles

This prototype emphasizes:

- preserving Meituan backend metric semantics and reporting-window grain;
- structuring store-period observations before pairwise comparison;
- converting diagnostics into retrieval-facing evidence records with source fields and observed values;
- carrying source paths, confidence labels, and limitations into memory facts;
- checking whether generated answers remain tied to entity, period, metric definitions, and documented evidence boundaries;
- returning boundary-preserving answers when the evidence does not support an operating conclusion.

## Reviewer Orientation

Use the `Admissions Review Path` table below as the first pass. It keeps the business problem, field dictionary, implemented diagnostics, repeated-window evidence, experiment results, and future comparability-gate contract in one stable order. Technical appendices remain available after the first pass.

## Admissions Review Path

For admissions review, use this path first. It keeps the retail decision-support story in one order: business problem, field contract, implemented diagnostics, repeated-window evidence, experiment results, future comparability-gate design, and optional grounded-review depth.

| Step | File | What to check |
|---:|---|---|
| 1 | `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Business origin, 48-store decision-support problem, staged prototype scope, and repeated-window evidence path. |
| 2 | `retail_ops/data/DATA_DICTIONARY.md` | Canonical Meituan metric meanings, implemented field names, and naming boundaries. |
| 3 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Store A month-over-month diagnostic path across February, March, and April 2026. |
| 4 | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | Same-period B-F diagnostic reading under one reporting window and one field contract. |
| 5 | `retail_ops/outputs/store_period_panel_coverage_output.csv` and `retail_ops/outputs/repeated_window_panel_summary_output.csv` | Repeated-window B-F evidence coverage and descriptive summary for 2026-02 to 2026-04. This is the preparation layer before future pairwise comparability rules. |
| 6 | `retail_ops/EXPERIMENT_RESULTS.md` | Experiment questions, implemented checks, validation outcomes, failure modes, and evidence-boundary behavior. |
| 7 | `retail_ops/COMPARABILITY_GATE_V0.md` | Future question-specific pairwise comparability-gate contract. |
| 8 | `rac/DEMO_INDEX.md` | Optional deterministic grounded-review scaffold for factor routing, critique, fact checking, and evidence-coverage reporting. |

Technical appendix material is consolidated under `retail_ops/TECHNICAL_APPENDIX.md`, but it is not required for the first admissions reading path.

## Architecture

The prototype has two connected layers.

| Layer | Purpose | Main files |
| --- | --- | --- |
| Memory-layer prototype | Store and retrieve typed facts while handling updates, stale knowledge, and unsupported questions. | `api/`, `scripts/`, `eval/` |
| Retail operations extension | Structure Meituan-style backend metrics and preserve diagnostic evidence for cautious comparison. | `retail_ops/` |

Basic flow:

```text
backend metrics / operator input
-> metric dictionary and data contract
-> SQL diagnostic output
-> generated memory facts
-> retrieval with source fields and limitations
-> qualified answer or refusal
```

The important design choice is that memory facts are not just summaries. They carry source fields, observed values, calculation notes, source paths, supporting source paths, confidence labels, and limitations.

## Implemented API and Retrieval Scope

The local FastAPI prototype includes general chat, memory, and retrieval endpoints. The key implemented paths are:

- `/health`
- `/chat`
- `/chat_mem`
- `/chat_livestream_kb`
- `/chat_retail_ops_kb`
- `/chat_retail_ops_demo2_kb`

The retail endpoints are local prototype endpoints over file-backed generated retail memory facts. Demo 2 endpoint checks are used to inspect evidence behavior before any live Meituan integration.

Retrieval-score inspection is kept as a separate offline analysis and is not the runtime selection logic of the Demo 2 endpoint.

### Retrieval Mode Boundary

| Endpoint | Current evidence mode | How to read it |
| --- | --- | --- |
| `/chat_livestream_kb` | Qdrant-backed lifecycle-aware memory retrieval. | Original memory-layer prototype for typed product facts, freshness, overwrite behavior, and fallback/refusal. |
| `/chat_retail_ops_kb` | Retail memory retrieval over implemented Store A facts. | Retail extension path for source-bounded Store A diagnostic facts. |
| `/chat_retail_ops_demo2_kb` | File-backed generated Demo 2 retail memory facts. | Boundary test for B-F same-period diagnostic facts; not retrieval-score evaluation and not a pairwise comparability gate. |

## Retail Evidence Files

Retail demo details are kept under `retail_ops/`.

| Evidence layer | Primary file |
|---|---|
| Store A month-over-month diagnostic | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` |
| B-F same-period diagnostic | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` |
| Retail experiment map and validation outcomes | `retail_ops/EXPERIMENT_RESULTS.md` |
| Future comparability-gate contract | `retail_ops/COMPARABILITY_GATE_V0.md` |

## Appendix Ownership

The first review path should stay short. These files are retained for technical audit, but they are not separate entry points.

| Appendix | Owns | Should not repeat |
|---|---|---|
| `retail_ops/TECHNICAL_APPENDIX.md` | Consolidated architecture, source-to-claim lineage, and field-usage review. | Admissions summary, field dictionary, future gate rationale, or experiment results. |

## Evaluation Snapshot

For retail field names and metric meanings, `retail_ops/data/DATA_DICTIONARY.md` is authoritative. For generated retail diagnostic values and retail evaluation outcomes, use `retail_ops/outputs/` and `eval/retail_decision_support_eval_results/`.

The evaluation layer checks whether the current prototype preserves metric definitions, source boundaries, entity/period scope, and comparison limits when selected retail evidence is retrieved.

| Check | Scope | Result |
| --- | --- | --- |
| Livestream memory evaluation | Fact retrieval, overwrite behavior, entity separation, fallback/refusal, and non-fact filtering. | Current implemented cases pass. |
| Retail retrieval evaluation | Store A retail-memory retrieval and unsupported-scope handling. | 8/8 passed. |
| Retail Demo 2 facts evaluation | Store B-F generated fact coverage across diagnostic slots. | 5/5 passed. |
| Retail Demo 2 scope-boundary evaluation | Demo 2 remains a row-level same-period diagnostic and does not expose future pairwise-gate schema. | 5/5 passed. |
| Retail Demo 2 answer-boundary evaluation | Activity-cost ratio, top-SKU share, search-entry comparison, promotion-transfer limits, same-period boundary, and `region_type` weak-context boundary. | 6/6 passed. |
| Retail Demo 2 endpoint behavior evaluation | File-backed Demo 2 endpoint behavior for supported Store B-F questions, unsupported-scope handling, and pairwise strategy-transfer boundary. | 7/7 passed. |
| Retail data-contract validation | Dictionary phrases, source/output headers, forbidden aliases, and generated fact structure. | Passed. |
| Retrieval score distribution inspection | Inspects score distributions over file-backed Demo 1 and Demo 2 retail evidence across supported, unsupported, hard-negative, entity/period mismatch, and ambiguous comparison queries. | Completed as offline inspection. |
| Query robustness under wording variation | Tests retrieval behavior under shortened, paraphrased, typo/noise, and keyword-order query variants. | Completed. |

Demo 2 guardrail sensitivity is also inspected. In the current B-F sample, all five store rows have `comparison_limit_notes` that change under at least one plus or minus 5 percentage-point threshold scenario. These rows remain useful as diagnostic evidence, while the threshold-sensitive notes should be reviewed again when broader repeated-window evidence is added.

## Optional Local Run

The repository can be reviewed through the Markdown documents, SQL files, generated outputs, and evaluation results without running the local API.

For local reproduction, the prototype uses FastAPI, Ollama, Qdrant, and Docker Compose. The local setup is defined by `docker-compose.yml`; the current validation commands are listed below.

## Reproduce Key Checks

Run the current implemented checks from the repository root:

```bash
python3 retail_ops/scripts/validate_retail_data_contract.py
python3 retail_ops/scripts/validate_demo2_comparability_output.py
python3 retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py
python3 retail_ops/scripts/validate_store_period_panel.py
python3 retail_ops/scripts/validate_repeated_window_panel_summary.py
python3 eval/eval_retail_demo2_facts.py
python3 eval/eval_retail_demo2_scope_boundary.py
python3 eval/eval_retail_demo2_answer_behavior.py
python3 eval/eval_retail_demo2_endpoint_behavior.py
python3 eval/eval_future_comparability_gate_contract.py
python3 scripts/validate_demo2_retail_endpoint_boundary.py
python3 scripts/validate_markdown_readability.py
python3 retail_ops/scripts/validate_csv_physical_rows.py
```

The endpoint behavior eval imports `api.main`, so run it inside the project virtual environment after dependencies are installed.

Optional offline retrieval-inspection checks:

```bash
python3 eval/analyze_retail_embedding_score_distribution.py
python3 eval/analyze_retail_query_robustness.py
```

These retrieval checks inspect score distribution and wording-variation behavior over the current file-backed retail evidence corpus.

## Key Evidence Files

Use the Admissions Review Path above for the first-pass file order.

Detailed retail file ownership is kept in `retail_ops/README.md`.

## Review Takeaway and Next Step

This repository demonstrates a staged decision-support prototype for a real Meituan multi-store operating problem. Selected backend metrics are preserved under documented definitions, structured with SQL, converted into retrieval-facing memory facts, and checked against evidence boundaries before later answers make operating claims.

| Area | Current role | Next use |
| --- | --- | --- |
| Metric dictionary | Preserves selected Meituan backend metric meanings and canonical field names. | Keeps later SQL, memory facts, and reviewer-facing answers aligned. |
| Demo 1 | Structures Store A February-April 2026 movement as a multi-metric operating profile. | Shows single-store temporal diagnosis. |
| Demo 2 | Structures selected B-F March 2026 store-period records under one reporting window and one field contract. | Provides same-period diagnostic evidence before stronger pairwise comparison rules. |
| Repeated-window panel | Adds B-F coverage and descriptive summary across 2026-02, 2026-03, and 2026-04. | Prepares the evidence base for future question-specific comparability checks. |
| Evaluation layer | Checks answer-boundary behavior, field meanings, entity scope, period scope, and comparison limits. | Keeps retrieval-backed answers tied to the current evidence path. |
| Future comparability gate | Design contract is documented in `retail_ops/COMPARABILITY_GATE_V0.md`. | Should judge whether two store-period records can be compared for one selected operating question. |

The next development step is to add more repeated store-period evidence and test whether the current diagnostic guardrails remain stable across more stores, months, activity conditions, and market contexts.

## Editing and Scope Guardrails

Retail field names and metric meanings must follow `retail_ops/data/DATA_DICTIONARY.md`.

Retail experiment wording and validation claims should stay aligned with `retail_ops/EXPERIMENT_RESULTS.md`.

Future pairwise comparability-gate wording must follow `retail_ops/COMPARABILITY_GATE_V0.md`.

## Structured Reasoning Scaffold: Factor-Aware Grounded Review

The `rac/` module adds a deterministic, source-aware review scaffold above the retail evidence path. It runs after the field dictionary, SQL diagnostics, and generated memory facts have already structured the available evidence.

Its role is to make the review path inspectable before a grounded report is accepted. It is secondary to the retail evidence path and does not replace SQL diagnostics, the field dictionary, or future pairwise comparability rules.

The scaffold covers:

- question decomposition
- factor expansion
- factor weighting
- local evidence grounding
- competing hypotheses
- critique
- fact check
- confidence and limitations

The scaffold is deterministic and local-evidence-based. It uses already structured project files to make factor coverage, missing evidence, critique, fact checks, and confidence updates inspectable before a grounded report is accepted.


Current RAC execution boundary:

- deterministic only
- no LLM calls
- no Qdrant retrieval
- no live Meituan backend access
- no completed pairwise comparability gate
- no causal proof from observational store metrics

Current RAC evidence and scripts:

- `rac/DEMO_INDEX.md`
- `rac/outputs/grounded_rac_store_a_attribution_001.md`
- `rac/outputs/grounded_rac_cross_store_comparability_001.md`
- `rac/outputs/grounded_quality_summary.md`
- `rac/scripts/run_grounded_pipeline.py`
- `rac/scripts/validate_grounded_quality_gate.py`

Current implementation scope: deterministic review over local project evidence.

<!-- RAC_EXTENSION_END -->
