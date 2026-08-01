# Lifecycle-Aware AI Memory Layer for Retail Decision Support

Repository: `livestream-agent-memory-layer`

A local evidence-bounded decision-support prototype that connects selected Meituan backend metrics, reproducible SQL diagnostics, source-bounded memory facts, retrieval and boundary tests, and deterministic grounded review.

## Core Research Question

This project grew from a real Meituan instant-retail operating problem.

The Meituan merchant backend provides detailed single-store metrics, but it is mainly designed for reviewing one store at a time. As store count increased, the harder problem became cross-store decision support: which store-period records can be compared, under what operating conditions, and what kind of operating judgment the available evidence can support.

For standardized instant-retail products, store competition is organized around one operating chain:

```text
being seen -> being entered -> being ordered -> being selected again / maintaining share
```

This chain is the business framing for the decision problem. Monthly transaction records provide the continuous sales outcome across reporting periods, while `maintaining share` is not treated as a separately measured result.

Promotion, subsidy, pricing, SKU arrangement, ranking position, and fulfillment stability are operating levers inside this chain. Their meaning depends on the observed store-period context, local competition, activity involvement, activity intensity, product mix, and reporting-window alignment.


The project organizes multi-store operating evidence so that each decision can be traced to defined fields, reporting periods, and documented limits as the business expands.

## Evidence Workflow

The repository follows one evidence path:

```text
selected merchant-backend metrics
-> canonical field dictionary
-> SQL diagnostic outputs
-> source-bounded retail memory facts
-> boundary and retrieval checks
-> RAC grounded review
```

Store-level source tables are manually transcribed from the merchant backend and anonymized. Metric meanings and canonical field names are fixed before transformation.

SQL organizes the selected store-period records. Generated facts retain the entity, reporting period, source fields, observed values, source paths, evidence-trace confidence, and limitations. Later checks test whether retrieval and answers remain inside that evidence.

`retail_ops/data/DATA_DICTIONARY.md` is the naming authority for retail fields and metric definitions.

## Current Implemented Scope

The current retail decision-support path has three implemented evidence layers: Demo 1 for Store A month-over-month diagnosis, Demo 2 for selected Stores B-F under one March 2026 reporting window, and the repeated-window B-F panel across 2026-02, 2026-03, and 2026-04.

The operating problem came from a 48-store business. The evidence committed to this repository covers six anonymized stores: Store A and selected Stores B-F under the reporting windows listed below. The 48-store figure describes the operating context; the committed analytical evidence covers the selected six stores.

Demo 2 keeps the `cross_store_comparability` file-path term for reference stability and provides same-period B-F diagnostic evidence with interpretation guardrails. Question-specific pairwise comparability is specified separately in `retail_ops/COMPARABILITY_GATE_V0.md`.

| Area | Current implementation | Current boundary |
| --- | --- | --- |
| Livestream memory layer | Typed product facts, overwrite control, soft deactivation, active-state retrieval, fallback/refusal, scenario evaluation. | Local prototype for lifecycle-aware memory behavior. |
| Data dictionary | Preserves selected Meituan merchant-backend metric meanings and canonical field names. | Manual normalization of selected backend evidence; field meanings follow `retail_ops/data/DATA_DICTIONARY.md`. |
| Retail Demo 1 | Store A month-over-month diagnostic across February, March, and April 2026. | Multi-metric interpretation rather than single-cause monthly explanation. |
| Retail Demo 2 | Same-period B-F diagnostic for March 2026. | Same-period diagnostic evidence with explicit interpretation limits before pairwise comparability rules. |
| Repeated-window panel | B-F coverage and descriptive summary across 2026-02, 2026-03, and 2026-04. | Descriptive repeated-window evidence before future pairwise comparability rules. |
| Memory facts | Converts diagnostic outputs into source-bounded facts with observed values, source fields, source paths, evidence-trace confidence labels, and limitations. | File-backed evidence records derived from manually structured source tables. |
| Answer-boundary checks | Tests whether answers stay within entity, period, metric-definition, source, and interpretation boundaries. | Scenario-based checks tied to the current evidence path; not broad LLM robustness tests. |
| Factor-aware grounded review layer (RAC) | Provides deterministic factor expansion, evidence routing, critique, rule-based checks for unsupported claims and definition conflicts, review-state updates, and grounded report generation over local project evidence. | File-grounded review with explicit source paths, competing hypotheses, and limitations. |

## Key Design Principles

This prototype emphasizes:

- preserving Meituan backend metric semantics and reporting-window grain;
- structuring store-period observations before pairwise comparison;
- converting diagnostics into retrieval-facing evidence records with source fields and observed values;
- carrying source paths, evidence-trace confidence labels, and limitations into memory facts;
- checking whether generated answers remain tied to entity, period, metric definitions, and documented evidence boundaries;
- returning boundary-preserving answers when the evidence does not support an operating conclusion.

## Admissions Review Path

[`PROJECT_SUMMARY_FOR_ADMISSIONS.md`](PROJECT_SUMMARY_FOR_ADMISSIONS.md)
is the single application-facing starting point. A first-pass review can
follow five files in one narrative order:

| Step | File | Review purpose |
|---:|---|---|
| 1 | [Project summary](PROJECT_SUMMARY_FOR_ADMISSIONS.md) | Understand the business origin, evidence coverage, implemented architecture, and decision boundary. |
| 2 | [Design evolution: from livestream product memory to retail decision support](case_studies/from_livestream_to_retail_decision_support.md) | See how lifecycle-aware product memory developed into store-period evidence management. |
| 3 | [Demo 1: Store A month-over-month diagnostic](retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md) | Inspect repeated-window Store A evidence and multi-metric interpretation. |
| 4 | [Demo 2: same-period B-F diagnostic](retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md) | Inspect multi-store evidence organization and interpretation guardrails. |
| 5 | [RAC grounded-review demo index](rac/DEMO_INDEX.md) | Inspect factor routing, structured-record grounding, boundary evidence, competing hypotheses, and report validation. |

Supporting evidence after the first pass:

- [Repeated-window B-F coverage](retail_ops/outputs/store_period_panel_coverage_output.csv)
  and [descriptive summary](retail_ops/outputs/repeated_window_panel_summary_output.csv)
  show the implemented February-April panel.
- [Experiment results](retail_ops/EXPERIMENT_RESULTS.md) records the
  reviewer-facing analytical results, validation procedures, and visible
  failure modes.
- [Promotion-review report](rac/outputs/grounded_rac_promotion_strategy_001.md)
  separates available cost, subsidy, and conversion evidence from
  unresolved decision requirements.
- [Data dictionary](retail_ops/data/DATA_DICTIONARY.md) and
  [technical appendix](retail_ops/TECHNICAL_APPENDIX.md) provide the
  field contract and source-to-claim audit references.
- [Comparability Gate V0](retail_ops/COMPARABILITY_GATE_V0.md) documents
  the future question-specific pairwise comparability contract.

## Architecture

The repository has three connected layers.

| Layer | Responsibility | Main files |
|---|---|---|
| Lifecycle-aware memory layer | Stores typed product facts, controls updates and active state, retrieves current evidence, and falls back when knowledge is unsupported. | `api/`, `scripts/`, `eval/` |
| Retail evidence layer | Preserves merchant-backend metric definitions, structures store-period evidence with SQL, generates source-bounded memory facts, and validates lineage and interpretation limits. | `retail_ops/` |
| Factor-aware grounded review layer (RAC) | Expands decision factors, routes local evidence, records competing hypotheses, applies critique and rule-based checks for unsupported claims and definition conflicts, and reports evidence coverage and limitations. | `rac/` |

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


## Evaluation Questions

For retail field names and metric meanings,
`retail_ops/data/DATA_DICTIONARY.md` is authoritative. Detailed
procedures, outputs, and failure cases are recorded in
`retail_ops/EXPERIMENT_RESULTS.md`, `retail_ops/outputs/`, and
`eval/retail_decision_support_eval_results/`.

The evaluation layer asks four questions:

1. Can committed diagnostic values be regenerated and traced to their
   source fields and formulas?
2. Do entity, period, metric-definition, and comparison boundaries remain
   intact through fact generation and endpoint answers?
3. Where do retrieval thresholds and wording variations route unsupported
   or mismatched queries incorrectly?
4. Does RAC keep a final judgment connected to relevant factors, source
   paths, competing hypotheses, and unresolved limitations?

The current results are:

| Check | Observed result | How to read it |
|---|---|---|
| Store A value lineage | The check covers 3 source rows, 3 SQL output rows, 9 top-SKU rows, 180 source, formula, movement, ranking, and trade-off comparisons, and 5 generated facts. | The lineage from source tables to SQL outputs and generated facts can be checked field by field, while the operating result remains multi-metric rather than a single-cause conclusion. |
| Demo 2 guardrail sensitivity | Baseline notes reproduce for all 5 rows. Raising the current thresholds by 5 percentage points changes Stores C-F; lowering them by 5 percentage points changes no rows. | The thresholds are prototype diagnostic warnings rather than optimized peer-selection rules. |
| Retrieval wording stress | Supported variants retain expected evidence in 34/34 cases. Hard-negative, entity/period-mismatch, and ambiguous variants cross the `0.5720` reference threshold in 23/33, 15/18, and 5/16 cases. | Semantic similarity helps route evidence but does not establish entity, period, or decision-scope support. |
| Repeated-window B-F panel | Stores B-F each retain February-April 2026 coverage across 11 selected metrics. | The panel supports descriptive review and later rule preparation, not a completed pairwise decision or monthly guardrail-stability result. |

These descriptive analyses, retrieval stress tests, and contract checks
have different meanings and are not combined into one accuracy score.
Difficult retrieval cases remain visible because they show why similarity
is only one evidence-routing signal.

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

## Repository Takeaway

This repository presents a reproducible path from selected merchant-backend observations to evidence-bounded decision support.

| Component | What it establishes |
| --- | --- |
| Metric dictionary | Preserves the selected Meituan backend definitions and canonical field names used by later transformations. |
| Demo 1 | Reconstructs Store A's February-April 2026 movement as a multi-metric store-period diagnostic. |
| Demo 2 | Structures selected B-F March 2026 records under one reporting window and records row-level interpretation limits. |
| Repeated-window panel | Makes B-F coverage and descriptive movement across February-April 2026 inspectable. |
| Evaluation and RAC | Tests value lineage, boundary preservation, retrieval failure modes, multi-factor evidence routing, and unresolved limitations. |

The documented pairwise comparability gate remains a separate experiment. Its purpose is to test whether two store-period records are suitable for one defined operating question before a cross-store interpretation is made.

## Editing and Scope Guardrails

Retail field names and metric meanings must follow `retail_ops/data/DATA_DICTIONARY.md`.

Retail experiment wording and validation claims should stay aligned with `retail_ops/EXPERIMENT_RESULTS.md`.

Future pairwise comparability-gate wording must follow `retail_ops/COMPARABILITY_GATE_V0.md`.

## Factor-Aware Grounded Review (RAC)

The `rac/` module operates over the structured retail evidence and records
factor selection, evidence routing, competing hypotheses, rule-based checks,
review-state updates, and unresolved limitations before a report is accepted.

Its implemented workflow covers:

- question analysis and factor expansion;
- interpretable factor weighting;
- source-aware local evidence routing;
- explicit boundary evidence for unavailable requirements;
- competing hypotheses;
- critique and rule-based checks for unsupported claims and definition conflicts;
- review-state updates that record evidence coverage and limitations;
- grounded report generation;
- a deterministic report-contract quality gate.

Start with `rac/DEMO_INDEX.md` for the reviewer-facing cases, generated
reports, execution commands, and quality-gate results.

The current implementation is deterministic and file-grounded. It
complements the field dictionary, SQL diagnostics, generated facts, and
answer-boundary evaluations rather than replacing them.
