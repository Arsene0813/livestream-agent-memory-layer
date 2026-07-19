# Lifecycle-Aware AI Memory Layer for Retail Decision Support

Repository: `livestream-agent-memory-layer`

A local evidence-bounded decision-support prototype for multi-store Meituan instant-retail operations, built with SQL diagnostics, metric-boundary preservation, lifecycle-aware retrieval, and deterministic grounded review.

The current implemented retail path applies the repository's lifecycle-aware evidence-boundary design to Meituan backend metrics, SQL diagnostic outputs, generated retail memory facts, answer-boundary checks, and RAC grounded review.

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

The current prototype follows one evidence path: selected merchant-backend metrics -> canonical field dictionary -> SQL diagnostic output -> generated retail memory facts -> boundary-preserving answer checks.

Current source tables were manually transcribed and anonymized at store level; this repository does not implement automated backend ingestion.

In practice, the project first preserves the documented metric definitions, then uses SQL to structure selected manually transcribed store-period records, and then converts diagnostic evidence into memory facts with source fields, observed values, source paths, and limitations. The final check is whether later answers stay inside the available evidence boundary.

The single source of truth for retail field names and metric meanings is:

- `retail_ops/data/DATA_DICTIONARY.md`

## Current Implemented Scope

The current retail decision-support path has three implemented evidence layers: Demo 1 for Store A month-over-month diagnosis, Demo 2 for selected Stores B-F under one March 2026 reporting window, and the repeated-window B-F panel across 2026-02, 2026-03, and 2026-04.

The operating problem came from a 48-store business. The evidence committed to this repository covers six anonymized stores: Store A and selected Stores B-F under the reporting windows listed below. The 48-store figure describes the operating context; the committed analytical evidence covers the selected six stores.

Demo 2 uses the `cross_store_comparability` file-path term for reference stability, but in the current implementation it means same-period B-F diagnostic evidence with interpretation guardrails. The future pairwise comparability gate remains question-specific: it should judge whether two store-period records can be compared for one selected operating question.

| Area | Current implementation | Current boundary |
| --- | --- | --- |
| Livestream memory layer | Typed product facts, overwrite control, soft deactivation, active-state retrieval, fallback/refusal, scenario evaluation. | Local prototype for lifecycle-aware memory behavior. |
| Data dictionary | Preserves Meituan-style backend metric meanings and canonical field names. | Manual normalization of selected backend evidence; field meanings follow `retail_ops/data/DATA_DICTIONARY.md`. |
| Retail Demo 1 | Store A month-over-month diagnostic across February, March, and April 2026. | Multi-metric interpretation rather than single-cause monthly explanation. |
| Retail Demo 2 | Same-period B-F diagnostic for March 2026. | Same-period diagnostic evidence with explicit interpretation limits before pairwise comparability rules. |
| Repeated-window panel | B-F coverage and descriptive summary across 2026-02, 2026-03, and 2026-04. | Descriptive repeated-window evidence before future pairwise comparability rules. |
| Memory facts | Converts diagnostic outputs into source-bounded facts with observed values, source fields, source paths, confidence labels, and limitations. | File-backed evidence records derived from manually structured source tables. |
| Answer-boundary checks | Tests whether answers stay within entity, period, metric-definition, source, and interpretation boundaries. | Scenario-based checks tied to the current evidence path; not broad LLM robustness tests. |
| Factor-aware grounded review layer (RAC) | Provides deterministic factor expansion, evidence routing, critique, rule-based claim and definition checks, review-state updates, and grounded report generation over local project evidence. | File-grounded review with explicit source paths, competing hypotheses, and limitations. |

## Key Design Principles

This prototype emphasizes:

- preserving Meituan backend metric semantics and reporting-window grain;
- structuring store-period observations before pairwise comparison;
- converting diagnostics into retrieval-facing evidence records with source fields and observed values;
- carrying source paths, confidence labels, and limitations into memory facts;
- checking whether generated answers remain tied to entity, period, metric definitions, and documented evidence boundaries;
- returning boundary-preserving answers when the evidence does not support an operating conclusion.

## Admissions Review Path

`PROJECT_SUMMARY_FOR_ADMISSIONS.md` is the single application-facing
starting point. The first pass follows the business problem, implemented
evidence, experiments, and RAC in one order.

| Step | File | Review purpose |
|---:|---|---|
| 1 | `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Understand the business problem, evidence coverage, implemented scope, and decision boundary. |
| 2 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Inspect the Store A repeated-window diagnostic and its multi-metric interpretation. |
| 3 | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | Inspect the same-period B-F diagnostic and its interpretation guardrails. |
| 4 | `retail_ops/outputs/store_period_panel_coverage_output.csv` and `retail_ops/outputs/repeated_window_panel_summary_output.csv` | Check repeated-window B-F coverage and descriptive movement. |
| 5 | `retail_ops/EXPERIMENT_RESULTS.md` | Review validation questions, procedures, results, and failure boundaries. |
| 6 | `rac/DEMO_INDEX.md` | Inspect the factor-aware grounded review pipeline, reports, and quality gate. |

After the first pass:

- `case_studies/from_livestream_to_retail_decision_support.md`
  preserves the complete system evolution from livestream product memory
  to retail decision support.
- `retail_ops/data/DATA_DICTIONARY.md` and
  `retail_ops/TECHNICAL_APPENDIX.md` are the field-contract and technical
  audit references.
- `retail_ops/COMPARABILITY_GATE_V0.md` records the future
  question-specific pairwise comparability contract.

## Architecture

The repository has three connected layers. Each layer owns a different
part of the decision-support problem.

| Layer | Responsibility | Main files |
|---|---|---|
| Lifecycle-aware memory layer | Stores typed product facts, controls updates and active state, retrieves current evidence, and falls back when knowledge is unsupported. | `api/`, `scripts/`, `eval/` |
| Retail evidence layer | Preserves merchant-backend metric definitions, structures store-period evidence with SQL, generates source-bounded memory facts, and validates lineage and interpretation limits. | `retail_ops/` |
| Factor-aware grounded review layer (RAC) | Expands decision factors, routes local evidence, records competing hypotheses, applies critique and rule-based claim and definition checks, and reports evidence coverage and limitations. | `rac/` |

The reviewer-oriented evidence flow is:

```text
selected merchant-backend metrics
-> canonical field contract
-> SQL diagnostic outputs
-> source-bounded memory facts
-> boundary evaluation and RAC grounded review
-> grounded report, qualified answer, or refusal
```

The retail evidence layer establishes what the available data supports.
RAC makes the multi-factor review path inspectable over that structured
evidence.

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

For retail field names and metric meanings, `retail_ops/data/DATA_DICTIONARY.md` is authoritative. Generated diagnostic values and evaluation outputs are recorded under `retail_ops/outputs/` and `eval/retail_decision_support_eval_results/`.

The pass counts below come from repository-defined contract, fixture, lineage, and endpoint checks. They show whether the current evidence path satisfies its declared checks; they are not predictive-accuracy or general-model-performance measures. Retrieval stress tests are reported separately because their main outputs are score distributions and failure cases rather than one combined pass rate.

| Check | Scope | Current result |
|---|---|---|
| Livestream memory evaluation | Fact retrieval, overwrite behavior, entity separation, fallback/refusal, and non-fact filtering. | Current implemented cases pass. |
| Store A retail retrieval evaluation | Store A retail-memory retrieval and unsupported-scope handling. | 8/8 repository-defined cases passed. |
| Retail Demo 2 fact-contract coverage | Exact Store B-F coverage across five implemented evidence slots, required metadata, repository-backed paths, and slot-specific boundary terms. | 25/25 entity-slot contracts passed: five stores multiplied by five implemented slots. |
| Retail Demo 2 scope-boundary evaluation | Required Demo 2 output fields, expected store IDs, row-level scope flag, and absence of future pairwise-gate schema. | 5/5 declared scope checks passed. |
| Retail Demo 2 answer-contract fixture validation | Six manually specified contracts covering activity-cost ratio, top-SKU evidence, search-entry comparison, strategy transfer, same-period readiness, and `region_type`. | 6/6 fixtures passed. This check evaluates required and forbidden wording; it does not call the endpoint or generate answers. |
| Retail Demo 2 endpoint-boundary evaluation | Supported Store B-F questions, unsupported all-48-store scope, ranking and final-recommendation requests, pairwise strategy transfer, and out-of-scope entities. | 7 endpoint scenarios passed. |
| Retail data-contract and value-lineage validation | Declared canonical fields, headers, aliases, formulas, source-to-output values, generated-fact values, paths, and metadata. | Current static-contract and Demo 1/Demo 2 lineage checks pass. |
| Retrieval threshold inspection | 29 cases over 282 file-backed retrieval documents, including supported, unsupported, hard-negative, mismatched, and ambiguous queries. | Supported cases retained expected evidence at top-5 in 8/8 cases; unsupported cases had 0/6 expected hits. The `0.5767` threshold remains an exploratory reference. |
| Query wording-variation stress test | 131 deterministic shortened, paraphrased, typo/noise, and keyword-order variants. | Supported variants retained expected evidence and crossed the reference threshold in 34/34 cases; unsupported variants had 0/30 expected hits and 0/30 threshold crossings. Hard-negative, entity/period-mismatch, and ambiguous variants still crossed the threshold in 23/33, 12/18, and 5/16 cases. |
| Demo 2 guardrail sensitivity | Baseline notes compared with thresholds lowered and raised by five percentage points. | Four of five store rows changed under the harder-to-trigger plus-five-point scenario; Store B remained unchanged, and the easier-to-trigger scenario changed no rows. |

The current evaluation supports contract consistency, value traceability, and inspectable evidence routing. The threshold-crossing failure cases show why semantic similarity remains one routing signal rather than a sufficient basis for an operating conclusion.

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

This repository demonstrates a staged decision-support prototype for a real Meituan multi-store operating problem. Selected merchant-backend metrics are manually transcribed into source tables under documented definitions, transformed with SQL, converted into retrieval-facing memory facts, and checked against evidence boundaries before later answers make operating claims.

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

## Factor-Aware Grounded Review (RAC)

The `rac/` module is an important technical component of the current
prototype. It operates over the structured retail evidence and makes the
reasoning path visible before a grounded report is accepted.

Its implemented workflow covers:

- question analysis and factor expansion;
- interpretable factor weighting;
- source-aware local evidence routing;
- explicit boundary evidence for unavailable requirements;
- competing hypotheses;
- critique and rule-based claim and definition checks;
- review-state updates that record evidence coverage and limitations;
- grounded report generation;
- a deterministic report-contract quality gate.

Start with `rac/DEMO_INDEX.md` for the reviewer-facing cases, generated
reports, execution commands, and quality-gate results.

The current implementation is deterministic and file-grounded. It
complements the field dictionary, SQL diagnostics, generated facts, and
answer-boundary evaluations rather than replacing them.
