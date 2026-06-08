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

Promotion, subsidy, pricing, SKU arrangement, ranking position, and fulfillment stability are operating levers inside this chain. Their meaning depends on store state, local competition, activity intensity, order quality, product mix, and reporting-window alignment.

The goal is to build a more reliable evidence-based framework for multi-store operational analysis, so that future operating decisions can be made with clearer data boundaries as the business expands.

## Current Prototype Workflow

The current prototype follows one evidence path: Meituan backend metric evidence -> canonical field dictionary -> SQL diagnostic output -> generated retail memory facts -> boundary-preserving answer checks.

In practice, this means the project first preserves backend metric definitions, then uses SQL to structure selected store-period data, then converts diagnostic evidence into memory facts with source fields, observed values, source paths, and limitations. The final check is whether later answers stay inside the available evidence boundary.

The single source of truth for retail field names and metric meanings is:

- `retail_ops/data/DATA_DICTIONARY.md`

## Current Implemented Scope

Retail Demo 2 is the current same-period diagnostic stage. It structures selected Stores B-F records under a shared March 2026 reporting window, the `DATA_DICTIONARY.md` field contract, and diagnostic guardrails before any future pairwise comparability gate is attempted.

A future pairwise comparability gate would judge whether two store-period records can be compared for a specific operating question. Naming note: existing Demo 2 paths keep `cross_store_comparability` for reference stability. In the current implementation, this means same-period diagnostic evidence and interpretation guardrails, not a completed pairwise comparability gate.

| Area | Current implementation | Current boundary |
| --- | --- | --- |
| Livestream memory layer | Typed product facts, overwrite control, soft deactivation, active-state retrieval, fallback/refusal, scenario evaluation. | Local prototype for lifecycle-aware memory behavior. |
| Data dictionary | Preserves Meituan-style backend metric meanings and canonical field names. | Manual normalization of selected backend evidence; field meanings follow `retail_ops/data/DATA_DICTIONARY.md`. |
| Retail Demo 1 | Store A month-over-month diagnostic across February, March, and April 2026. | Multi-metric interpretation rather than single-cause monthly explanation. |
| Retail Demo 2 | Same-period B-F diagnostic for March 2026. | Same-period diagnostic evidence before pairwise comparability gating; not peer selection, store ranking, or strategy-transfer approval. |
| SQL diagnostics | Derives limited diagnostic fields such as search-entry structure, activity involvement, refund pressure, invalid-order pressure, and top-SKU concentration. | Diagnostic structuring only; current derived fields are not optimized business cutoffs or final decision rules. |
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

## Fast Reading Path

For admissions review, use the `Admissions Review Path` table below. It keeps the business problem, current Demo 2 boundary, field dictionary, experiment results, and future comparability-gate contract in one stable order.

For technical audit, use the same table first. Architecture, lineage, field-change review, and future-gate details are kept as appendices under `retail_ops/` so that the main reviewer path stays short.

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

The retail endpoints are local prototype endpoints. Demo 2 currently uses file-backed generated retail memory facts; it is not a production Meituan API integration.

Retrieval-score inspection is kept as a separate offline analysis and is not the runtime selection logic of the Demo 2 endpoint.

### Retrieval Mode Boundary

| Endpoint | Current evidence mode | How to read it |
| --- | --- | --- |
| `/chat_livestream_kb` | Qdrant-backed lifecycle-aware memory retrieval. | Original memory-layer prototype for typed product facts, freshness, overwrite behavior, and fallback/refusal. |
| `/chat_retail_ops_kb` | Retail memory retrieval over implemented Store A facts. | Retail extension path for source-bounded Store A diagnostic facts. |
| `/chat_retail_ops_demo2_kb` | File-backed generated Demo 2 retail memory facts. | Boundary test for B-F same-period diagnostic facts; not retrieval-score evaluation and not a pairwise comparability gate. |

## Retail Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes one self-operated Qingdao store across February, March, and April 2026.

- Main file: `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`
- Role in the project: preserve a careful month-over-month operating profile before any cross-store interpretation is attempted.

## Retail Demo 2: Same-Period B-F Diagnostic

Demo 2 is the current cross-store diagnostic stage. It uses selected Stores B-F under the same March 2026 reporting window and the same `retail_ops/data/DATA_DICTIONARY.md` field contract.

The section exposes store-period operating profiles and points to the canonical future-gate contract instead of repeating the full boundary in this README.

- Main file: `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`
- Future gate contract: `retail_ops/COMPARABILITY_GATE_V0.md`
- Current boundary: same-period diagnostic evidence only; peer selection, store ranking, and strategy-transfer approval belong to the future gate stage.

## Admissions Review Path

For admissions review, use this path first. It keeps the retail decision-support story in one order and avoids reading implementation appendices before the business problem is clear.

| Step | File | What to check |
|---:|---|---|
| 1 | `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Business origin, staged prototype scope, and current Demo 2 boundary. |
| 2 | `retail_ops/data/DATA_DICTIONARY.md` | Canonical Meituan metric meanings and implemented field names. |
| 3 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Store A month-over-month diagnostic path. |
| 4 | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | Same-period B-F diagnostic reading and comparison limits. |
| 5 | `retail_ops/EXPERIMENTS.md` | What each current analytical check is designed to test. |
| 6 | `retail_ops/EXPERIMENT_RESULTS.md` | Implemented checks and validation outcomes. |
| 7 | `retail_ops/COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate contract. |

Technical appendices remain available under `retail_ops/ARCHITECTURE.md`, `retail_ops/LINEAGE.md`, and `retail_ops/FIELD_USAGE_REVIEW.md`, but they are not required for the first admissions reading path.

## Appendix Ownership

The first review path should stay short. These files are retained for technical audit, but they are not separate entry points.

| Appendix | Owns | Should not repeat |
|---|---|---|
| `retail_ops/ARCHITECTURE.md` | System structure, evidence path, retrieval mode, and endpoint responsibility. | Admissions summary, field dictionary, or future gate rationale. |
| `retail_ops/LINEAGE.md` | Source-to-SQL-to-memory lineage and claim traceability. | Full architecture explanation or repeated business narrative. |
| `retail_ops/FIELD_USAGE_REVIEW.md` | Field-name and semantic-change review before future expansion. | General project summary or experiment results. |

## Evaluation Snapshot

For retail field names and metric meanings, `retail_ops/data/DATA_DICTIONARY.md` is authoritative. For generated diagnostic values and evaluation outcomes, use the saved files under `retail_ops/outputs/` and `eval/results/` if this summary table is later updated.

The evaluations are intentionally narrow scenario-based behavior checks. They do not prove business correctness, causal effects, or general model performance. Their value is checking whether the current prototype preserves metric definitions, source boundaries, entity/period scope, and comparison limits when limited retail evidence is retrieved.

Demo 2 guardrail sensitivity is also inspected. In the current B-F sample, all five store rows have `comparison_limit_notes` that change under at least one plus or minus 5 percentage-point threshold scenario. This means the current threshold notes should be read as diagnostic warnings, not stable peer-selection rules, strategy-transfer approvals, or optimized business cutoffs.

| Check | Scope | Result |
| --- | --- | --- |
| Livestream memory evaluation | Fact retrieval, overwrite behavior, entity separation, fallback/refusal, non-fact filtering. | Current implemented cases pass. |
| Retail retrieval evaluation | Store A retail-memory retrieval and unsupported-scope refusal. | 8/8 passed. |
| Retail Demo 2 facts evaluation | Store B-F generated fact coverage across diagnostic slots. | 6/6 passed. |
| Retail Demo 2 scope-boundary evaluation | Demo 2 remains a row-level same-period diagnostic and does not expose future pairwise-gate schema. | 5/5 passed. |
| Retail Demo 2 answer-boundary evaluation | Activity-cost ratio, top-SKU share, search-entry comparison, promotion-transfer limits, same-period boundary, and `region_type` weak-context boundary. | 6/6 passed. |
| Retail Demo 2 endpoint behavior evaluation | File-backed Demo 2 endpoint behavior for supported Store B-F questions, unsupported-scope refusal, and pairwise strategy-transfer refusal. | 7/7 passed. |
| Retail data-contract validation | Dictionary phrases, source/output headers, forbidden aliases, generated fact structure. | Passed. |
| Retrieval score distribution inspection | Inspects score distributions over file-backed Demo 1 and Demo 2 retail evidence across supported, unsupported, hard-negative, entity/period mismatch, and ambiguous comparison queries. | Completed as offline inspection; not used as production or Demo 2 runtime threshold. |
| Query robustness under wording variation | Tests whether retrieval behavior remains stable under shortened, paraphrased, typo/noise, and keyword-order query variants. | Completed. |
| Project consistency validation | Current-scope files, Demo 2 boundary wording, stale future-work artifacts, endpoint claims. | Passed. |

Offline retrieval inspections make score behavior inspectable, but answer safety still depends on entity, period, slot, source path, and interpretation-boundary checks rather than retrieval score alone.

## Optional Local Run

The repository can be reviewed through the Markdown documents, SQL files, generated outputs, and evaluation results without running the local API.

For local reproduction, the prototype uses FastAPI, Ollama, Qdrant, and Docker Compose. The local setup is defined by `docker-compose.yml`; the current validation commands are listed below.

## Reproduce Key Checks

Run the current implemented checks from the repository root:

```bash
python3 retail_ops/scripts/validate_retail_data_contract.py
python3 retail_ops/scripts/validate_demo2_comparability_output.py
python3 retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py
python3 eval/eval_retail_demo2_facts.py
python3 eval/eval_retail_demo2_scope_boundary.py
python3 eval/eval_retail_demo2_answer_behavior.py
python3 eval/eval_retail_demo2_endpoint_behavior.py

The endpoint behavior eval imports `api.main`, so run it inside the project virtual environment after dependencies are installed.
python3 eval/eval_future_comparability_gate_contract.py
python3 scripts/validate_demo2_retail_endpoint_boundary.py
python3 scripts/validate_project_consistency.py
python3 scripts/validate_markdown_readability.py
python3 retail_ops/scripts/validate_csv_physical_rows.py
```

Optional offline retrieval-inspection checks:

```bash
python3 eval/analyze_retail_embedding_score_distribution.py
python3 eval/analyze_retail_query_robustness.py
```

These retrieval checks inspect score distribution and wording-variation behavior over the current file-backed retail evidence corpus. They support the boundary-check design, but they are not production retrieval benchmarks.

## Key Evidence Files

| File | Why it matters |
| --- | --- |
| `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Admissions-facing summary of the real business problem and prototype scope. |
| `retail_ops/data/DATA_DICTIONARY.md` | Canonical backend metric definitions and naming boundaries. |
| `retail_ops/LINEAGE.md` | How source fields support SQL diagnostics and memory facts. |
| `retail_ops/FIELD_USAGE_REVIEW.md` | Field-name review before future comparability-gate expansion. |
| `retail_ops/EXPERIMENTS.md` | Experiment questions, inputs, transformations, pass conditions, and failure modes. |
| `retail_ops/COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate design note. |
| `retail_ops/sql/` | SQL transformations for Demo 1 and Demo 2. |
| `retail_ops/outputs/` | Generated diagnostic outputs and generated memory facts. |
| `eval/` | Scenario-based evaluation scripts and reports. |

## What This Demonstrates

This project demonstrates:

- turning a real multi-store Meituan instant-retail operating problem into a structured data problem;
- preserving backend metric definitions instead of flattening them into generic business metrics;
- using SQL to place selected store-period records under a shared diagnostic structure before stronger comparability claims are made;
- converting diagnostic outputs into retrieval-facing memory facts with source fields, observed values, confidence labels, and limitations;
- testing whether later answers preserve metric boundaries, source scope, and comparison limits;
- adding a deterministic review scaffold that makes factor selection, evidence routing, critique, and confidence limits visible.

The core contribution is the evidence boundary: selected backend metrics are kept under documented definitions, SQL diagnostics expose interpretation limits, and generated memory facts carry source fields and limitations before later answers are allowed to make operating claims.

## Editing and Scope Guardrails

Retail field names and metric meanings must follow `retail_ops/data/DATA_DICTIONARY.md`.

Future field-name or semantic changes must follow `retail_ops/FIELD_USAGE_REVIEW.md` before source CSVs, SQL outputs, generated facts, README/admissions wording, or evaluation cases are changed.

Future pairwise comparability-gate wording must follow `retail_ops/COMPARABILITY_GATE_V0.md`.

## Current Boundary and Next Development

The current retail implementation is a local evidence-bounded prototype. Demo 1 covers Store A month-over-month analysis, and Demo 2 covers selected Stores B-F under the same March 2026 reporting window. Demo 2 is a diagnostic evidence layer under one field contract before future pairwise comparability rules are added.

| Area | Current status | Boundary |
| --- | --- | --- |
| Backend data | Selected Meituan backend evidence is manually structured into source files. | Not production Meituan backend ingestion. |
| Demo 2 | Same-period B-F diagnostic evidence. | Not a peer-selection rule, store ranking, or strategy-transfer approval. |
| Retrieval behavior | File-backed facts and offline retrieval inspections are available. | Retrieval score alone is not treated as sufficient evidence for operating conclusions. |
| `region_type` | Weak region or market-context evidence. | Not a mature market-area classification, store-stage label, or hard peer-grouping rule. |
| Top-SKU evidence | Lightweight product-mix evidence from selected top-SKU rows. | Not full product-category sales share. |
| Future comparability gate | Design contract is documented in `retail_ops/COMPARABILITY_GATE_V0.md`. | Not implemented in the current prototype. |

The next development step is to add more repeated store-period evidence and test whether the current diagnostic guardrails remain stable across more stores, months, activity conditions, and market contexts. Future pairwise comparison should be question-specific: a store pair may be comparable for search-entry structure but not comparable for promotion transfer, pricing pressure, SKU strategy, refund interpretation, or fulfillment interpretation.

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

Current RAC evidence and scripts:

- `rac/DEMO_INDEX.md`
- `rac/outputs/grounded_rac_store_a_attribution_001.md`
- `rac/outputs/grounded_rac_cross_store_comparability_001.md`
- `rac/outputs/grounded_quality_summary.md`
- `rac/scripts/run_grounded_pipeline.py`
- `rac/scripts/validate_grounded_quality_gate.py`

Current implementation boundary: deterministic and local-evidence-based. It should be read as a review scaffold over the current project evidence, not as live backend ingestion or operating-decision automation.

## Current Retail Decision-Support Path

The retail decision-support path now has three implemented evidence layers:

1. Store A month-over-month diagnostic for one-store temporal interpretation.
2. B-F same-period diagnostic for one-window cross-store review under a shared field contract.
3. B-F repeated-window panel coverage and descriptive summary for 2026-02 to 2026-04.

The third layer is not a pairwise comparability gate. It is a preparation step: before deciding which stores can be compared under which conditions, the project first checks whether repeated monthly evidence exists and whether the fields remain aligned with `retail_ops/data/DATA_DICTIONARY.md`.
