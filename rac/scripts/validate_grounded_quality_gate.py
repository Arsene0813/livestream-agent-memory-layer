from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rac.src.grounded_pipeline import run_grounded_pipeline, save_grounded_outputs
from rac.src.local_evidence_resolver import SOURCE_FACTOR_KEYWORDS


REQUIRED_REPORT_SECTIONS = [
    "# Grounded RAC Report",
    "## 1. Direct Answer",
    "## 2. Question Type",
    "## 3. Factor Weights",
    "### 3a. How Factor Weights Are Generated",
    "### 3b. Factor Weights Used in This Report",
    "## 4. Local Evidence Grounding",
    "## 5. Competing Hypotheses",
    "## 6. Critic Findings",
    "## 7. Claim and Definition Check",
    "## 8. Final Judgment",
    "## 9. Evidence-Coverage Score",
    "## 10. What Cannot Be Concluded",
    "## 11. Review-State Update",
]

FORBIDDEN_POSITIVE_CLAIMS = [
    "Search exposure caused April growth",
    "Search exposure alone explains April growth",
    "The system proves causality",
    "The system has live Meituan backend access",
    "Stores B-F are fully comparable",
    "Demo 2 implements a pairwise comparability gate",
    "Region type is a strong market classifier",
    "The system proves which store is better",
    "Activity cost ratio is ROI",
    "The system updates neural network weights",
    "The system has a true autonomous world model",
    "The system solves causal inference automatically"
]

ALLOWED_GROUNDING_STATUSES = {
    "keyword_matched",
    "boundary_matched",
    "source_found_no_keyword_match"
}

MIN_SNIPPET_CHARS = 8

CROSS_STORE_REQUIRED_QUANTITATIVE_SOURCES = {
    "order_volume": "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "transaction_amount": "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "sku_structure": "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "repeated_reporting_windows": "retail_ops/outputs/repeated_window_panel_summary_output.csv"
}

CROSS_STORE_REQUIRED_BOUNDARY_SOURCES = {
    "competition": "retail_ops/COMPARABILITY_GATE_V0.md",
}


def fail(message: str) -> None:
    raise SystemExit(f"[RAC report-contract quality gate failed] {message}")


def load_eval_cases() -> list[dict[str, Any]]:
    path = ROOT / "rac" / "eval" / "rac_eval_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def validate_report_sections(report: str) -> list[str]:
    issues: list[str] = []

    for section in REQUIRED_REPORT_SECTIONS:
        if section not in report:
            issues.append(f"missing report section: {section}")

    return issues


def validate_forbidden_claims(report: str) -> list[str]:
    issues: list[str] = []
    normalized_report = normalize(report)

    for claim in FORBIDDEN_POSITIVE_CLAIMS:
        if normalize(claim) in normalized_report:
            issues.append(f"forbidden positive claim found: {claim}")

    return issues


def validate_grounded_rows(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, int]]:
    issues: list[str] = []
    status_counts: dict[str, int] = {}

    if not rows:
        issues.append("no grounded evidence rows")
        return issues, status_counts

    for index, row in enumerate(rows):
        status = row.get("grounding_status", "")
        status_counts[status] = status_counts.get(status, 0) + 1

        factor_id = str(row.get("factor_id", ""))
        source_path = str(row.get("source_path", ""))
        line_range = str(row.get("line_range", ""))
        snippet = str(row.get("snippet", ""))
        grounding_role = str(row.get("grounding_role", ""))

        if not factor_id:
            issues.append(f"row {index} missing factor_id")

        if not source_path:
            issues.append(f"row {index} missing source_path")
        else:
            absolute_path = ROOT / source_path
            if not absolute_path.exists():
                issues.append(f"row {index} source file does not exist: {source_path}")

        if not grounding_role:
            issues.append(f"row {index} missing grounding_role")

        if status not in ALLOWED_GROUNDING_STATUSES:
            issues.append(f"row {index} invalid grounding_status: {status}")

        if not re.match(r"^\d+-\d+$", line_range):
            issues.append(f"row {index} invalid line_range: {line_range}")

        if len(snippet.strip()) < MIN_SNIPPET_CHARS:
            issues.append(f"row {index} snippet too short for factor {factor_id}")

        required_anchors = SOURCE_FACTOR_KEYWORDS.get(
            (
                factor_id,
                source_path,
            )
        )

        if required_anchors:
            normalized_snippet = normalize(
                snippet
            )

            anchor_matched = any(
                normalize(anchor)
                in normalized_snippet
                for anchor in required_anchors
            )

            if (
                status
                == "source_found_no_keyword_match"
            ):
                issues.append(
                    f"row {index} used fallback "
                    "context for anchored factor "
                    f"{factor_id}"
                )

            if not anchor_matched:
                issues.append(
                    f"row {index} semantic anchor "
                    f"mismatch for {factor_id}: "
                    f"expected one of "
                    f"{required_anchors!r}"
                )

    return issues, status_counts


def validate_cross_store_grounding(state: dict[str, Any]) -> list[str]:
    issues: list[str] = []

    if state.get("question_type") != "comparability_judgment":
        return issues

    rows = {
        row.get("factor_id"): row
        for row in state.get("grounded_evidence_rows", [])
    }

    summary = state.get("grounded_evidence", {}).get("summary", {})
    fallback_count = summary.get("fallback_count", 0)

    if fallback_count > 1:
        issues.append(f"cross-store fallback_count too high: {fallback_count}")

    for factor_id, expected_source in CROSS_STORE_REQUIRED_QUANTITATIVE_SOURCES.items():
        row = rows.get(factor_id)

        if not row:
            issues.append(f"cross-store missing factor row: {factor_id}")
            continue

        if row.get("source_path") != expected_source:
            issues.append(
                f"cross-store factor {factor_id} expected source {expected_source}, "
                f"got {row.get('source_path')}"
            )

        if row.get("grounding_role") != "quantitative_evidence":
            issues.append(
                f"cross-store factor {factor_id} expected quantitative_evidence role, "
                f"got {row.get('grounding_role')}"
            )

        if row.get("grounding_status") != "keyword_matched":
            issues.append(
                f"cross-store factor {factor_id} expected keyword_matched status, "
                f"got {row.get('grounding_status')}"
            )

    for factor_id, expected_source in CROSS_STORE_REQUIRED_BOUNDARY_SOURCES.items():
        row = rows.get(factor_id)

        if not row:
            issues.append(f"cross-store missing boundary factor row: {factor_id}")
            continue

        if row.get("source_path") != expected_source:
            issues.append(
                f"cross-store boundary factor {factor_id} expected source {expected_source}, "
                f"got {row.get('source_path')}"
            )

        if row.get("grounding_role") != "boundary_evidence":
            issues.append(
                f"cross-store boundary factor {factor_id} expected boundary_evidence role, "
                f"got {row.get('grounding_role')}"
            )

        if row.get("grounding_status") != "boundary_matched":
            issues.append(
                f"cross-store boundary factor {factor_id} expected boundary_matched status, "
                f"got {row.get('grounding_status')}"
            )

    report = state.get("final_report", "")

    required_report_phrases = [
        "same-period diagnostic review",
        "should not be treated as directly comparable",
        "Pairwise quantitative gates are not defined in the current contract."
    ]

    for phrase in required_report_phrases:
        if phrase not in report:
            issues.append(f"cross-store report missing boundary phrase: {phrase}")

    return issues


def validate_state(case: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    case_id = case["case_id"]
    issues: list[str] = []

    report = state.get("final_report", "")
    grounded = state.get("grounded_evidence", {})
    summary = grounded.get("summary", {})
    rows = state.get("grounded_evidence_rows", [])

    issues.extend(validate_report_sections(report))
    issues.extend(validate_forbidden_claims(report))

    row_issues, row_status_counts = validate_grounded_rows(rows)
    issues.extend(row_issues)

    issues.extend(validate_cross_store_grounding(state))

    factor_count = len(state.get("factors", []))
    factor_weight_count = len(state.get("factor_weights", []))
    hypothesis_count = len(state.get("hypotheses", []))
    critic_count = len(state.get("critic_findings", []))
    limitation_count = len(state.get("belief_update", {}).get("limitations", []))

    if factor_count == 0:
        issues.append("no factors found")

    if factor_weight_count != factor_count:
        issues.append(
            f"factor weight count mismatch: factors={factor_count}, "
            f"weights={factor_weight_count}"
        )

    if hypothesis_count < 2:
        issues.append("fewer than 2 hypotheses")

    if critic_count == 0:
        issues.append("no critic findings")

    if limitation_count == 0:
        issues.append("no belief limitations")

    if summary.get("source_missing_count") != 0:
        issues.append(f"source_missing_count is not zero: {summary.get('source_missing_count')}")

    if summary.get("total_packets") != len(rows):
        issues.append(
            f"packet/row mismatch: total_packets={summary.get('total_packets')}, "
            f"rows={len(rows)}"
        )

    if (summary.get("keyword_matched_count", 0) + summary.get("boundary_matched_count", 0)) == 0:
        issues.append("zero keyword-or-boundary matched packets")

    required_report_contract_phrases = [
        'Deterministic local-file review',
        'fixed review-priority buckets assigned by explicit rules',
        'Scenario-Template Confidence',
        'Unsupported claims detected by current rules',
        'Definition conflicts detected by current rules',
        'The judgment is bounded by the cited local evidence',
        'Score contract:',
        'The score summarizes evidence-routing coverage under the current rules.',
        'Alternative weights are a formula sensitivity check; the report judgment is produced separately.',
        'Reading the score:',
        'Read the score as coverage rather than evidence strength, causal validity, decision quality, or business impact.',
    ]

    for phrase in required_report_contract_phrases:
        if phrase not in report:
            issues.append(
                "report missing concise contract phrase: "
                f"{phrase}"
            )

    legacy_defensive_phrases = [
        'It does not call an LLM, vector database, or live backend service.',
        'not by a learned model',
        'not by direct calculation from observed metric tables',
        'They are not learned probabilities, calibrated likelihoods',
        'Unsupported claims: none',
        'Definition conflicts: none',
        'but it does not prove causality or replace a full retrieval system.',
        'Formula limitation:',
        'Future sensitivity check:',
        'They are not learned parameters, optimized thresholds, calibrated probabilities, or business-performance predictors.',
        'This is a deterministic evidence-routing coverage score for the current local report.',
        'It is not a learned probability, Bayesian posterior, causal confidence score, or business-success probability.',
        'It does not measure evidence strength, conclusion correctness, decision quality, or business impact',
    ]

    for phrase in legacy_defensive_phrases:
        if phrase in report:
            issues.append(
                "legacy defensive report phrase found: "
                f"{phrase}"
            )

    for row in state.get("factor_weights", []):
        if "weight_bucket" not in row:
            issues.append(f"factor weight row missing weight_bucket: {row.get('factor_id')}")
        if "weighting_method" not in row:
            issues.append(f"factor weight row missing weighting_method: {row.get('factor_id')}")
        if "weight_source" not in row:
            issues.append(f"factor weight row missing weight_source: {row.get('factor_id')}")

    if "Evidence Fields" not in report:
        issues.append("report does not expose Evidence Fields column")

    if "Matched Terms" in report:
        issues.append("report exposes raw matched terms instead of curated evidence fields")

    if "Missing source files: 0" not in report:
        issues.append("report does not explicitly show Missing source files: 0")

    passed = len(issues) == 0

    return {
        "case_id": case_id,
        "question_type": state.get("question_type"),
        "passed": passed,
        "issues": issues,
        "metrics": {
            "factor_count": factor_count,
            "factor_weight_count": factor_weight_count,
            "hypothesis_count": hypothesis_count,
            "critic_count": critic_count,
            "limitation_count": limitation_count,
            "grounded_row_count": len(rows),
            "total_packets": summary.get("total_packets", 0),
            "keyword_matched_count": summary.get("keyword_matched_count", 0),
            "boundary_matched_count": summary.get("boundary_matched_count", 0),
            "fallback_count": summary.get("fallback_count", 0),
            "source_missing_count": summary.get("source_missing_count", 0),
            "row_status_counts": row_status_counts
        }
    }


def write_markdown_summary(results: list[dict[str, Any]], output_path: Path) -> None:
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result["passed"])
    failed_cases = total_cases - passed_cases

    total_packets = sum(result["metrics"]["total_packets"] for result in results)
    total_keyword = sum(result["metrics"]["keyword_matched_count"] for result in results)
    total_boundary = sum(result["metrics"]["boundary_matched_count"] for result in results)
    total_fallback = sum(result["metrics"]["fallback_count"] for result in results)
    total_missing = sum(result["metrics"]["source_missing_count"] for result in results)

    lines: list[str] = []

    lines.append("# RAC Report-Contract Summary")
    lines.append("")
    lines.append(
        "This file is generated by "
        "rac/scripts/validate_grounded_quality_gate.py."
    )
    lines.append("")
    lines.append("## Contract Scope")
    lines.append("")
    lines.append(
        "A contract pass means the case satisfied the checks "
        "implemented by the current deterministic rule set: "
        "report structure, source traceability, semantic anchors, "
        "and selected claim boundaries. It does not establish "
        "causal validity, decision quality, or business impact."
    )
    lines.append("")
    lines.append("## Overall Result")
    lines.append("")
    lines.append(f"- Total cases: {total_cases}")
    lines.append(
        f"- Report-contract passed cases: {passed_cases}"
    )
    lines.append(
        f"- Report-contract failed cases: {failed_cases}"
    )
    lines.append(f"- Total grounded packets: {total_packets}")
    lines.append(f"- Keyword matched packets: {total_keyword}")
    lines.append(f"- Boundary matched packets: {total_boundary}")
    lines.append(f"- Fallback packets: {total_fallback}")
    lines.append(f"- Missing source files: {total_missing}")
    lines.append("")
    lines.append("## Case Results")
    lines.append("")
    lines.append(
        "| Case | Contract Pass | Factors | Hypotheses "
        "| Critic Findings | Grounded Rows | Keyword Matched "
        "| Boundary Matched | Fallback | Missing Sources |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for result in results:
        metrics = result["metrics"]
        lines.append(
            f"| {result['case_id']} "
            f"| {result['passed']} "
            f"| {metrics['factor_count']} "
            f"| {metrics['hypothesis_count']} "
            f"| {metrics['critic_count']} "
            f"| {metrics['grounded_row_count']} "
            f"| {metrics['keyword_matched_count']} "
            f"| {metrics['boundary_matched_count']} "
            f"| {metrics['fallback_count']} "
            f"| {metrics['source_missing_count']} |"
        )

    lines.append("")
    lines.append("## Cross-Store Grounding Requirement")
    lines.append("")
    lines.append("For rac_cross_store_comparability_001, the report-contract quality gate requires:")
    lines.append("")
    lines.append("- fallback_count <= 1")
    lines.append("- order_volume -> retail_ops/outputs/demo2_cross_store_comparability_output.csv")
    lines.append("- transaction_amount -> retail_ops/outputs/demo2_cross_store_comparability_output.csv")
    lines.append("- sku_structure -> retail_ops/outputs/demo2_cross_store_comparability_output.csv")
    lines.append("- competition -> retail_ops/COMPARABILITY_GATE_V0.md as boundary_evidence")
    lines.append("- repeated_reporting_windows -> retail_ops/outputs/repeated_window_panel_summary_output.csv as quantitative_evidence")
    lines.append("")
    lines.append("## Report-Contract Issues")
    lines.append("")

    any_issue = False

    for result in results:
        if not result["issues"]:
            continue

        any_issue = True
        lines.append(f"### {result['case_id']}")
        lines.append("")

        for issue in result["issues"]:
            lines.append(f"- {issue}")

        lines.append("")

    if not any_issue:
        lines.append(
            "No report-contract issues were detected "
            "by the current rule set."
        )
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cases = load_eval_cases()
    output_dir = ROOT / "rac" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not cases:
        fail("No eval cases found")

    results: list[dict[str, Any]] = []

    for case in cases:
        state = run_grounded_pipeline(case["question"], root=ROOT)
        save_grounded_outputs(state, output_dir, case["case_id"])

        result = validate_state(case, state)
        results.append(result)

    summary = {
        "validation_scope": (
            "Current deterministic report-contract checks: "
            "report structure, source traceability, semantic "
            "anchors, and selected claim boundaries."
        ),
        "pass_interpretation": (
            "A passed case satisfies the current rule set; "
            "it does not establish causal validity, decision "
            "quality, or business impact."
        ),
        "total_cases": len(results),
        "passed_cases": sum(
            1
            for result in results
            if result["passed"]
        ),
        "failed_cases": sum(
            1
            for result in results
            if not result["passed"]
        ),
        "results": results,
    }

    json_path = output_dir / "grounded_quality_summary.json"
    md_path = output_dir / "grounded_quality_summary.md"

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    write_markdown_summary(results, md_path)

    failed = [result for result in results if not result["passed"]]

    if failed:
        for result in failed:
            print(f"[FAIL] {result['case_id']}")

            for issue in result["issues"]:
                print(f"  - {issue}")

        fail(f"{len(failed)} grounded quality case(s) failed")

    total_packets = sum(result["metrics"]["total_packets"] for result in results)
    total_keyword = sum(result["metrics"]["keyword_matched_count"] for result in results)
    total_boundary = sum(result["metrics"]["boundary_matched_count"] for result in results)
    total_fallback = sum(result["metrics"]["fallback_count"] for result in results)
    total_missing = sum(result["metrics"]["source_missing_count"] for result in results)

    print("[OK] RAC report-contract quality gate passed")
    print(f"[OK] Cases checked: {len(results)}")
    print(f"[OK] Total grounded packets: {total_packets}")
    print(f"[OK] Keyword matched packets: {total_keyword}")
    print(f"[OK] Boundary matched packets: {total_boundary}")
    print(f"[OK] Fallback packets: {total_fallback}")
    print(f"[OK] Missing source files: {total_missing}")
    print(f"[OK] Summary JSON: {json_path}")
    print(f"[OK] Summary MD: {md_path}")


if __name__ == "__main__":
    main()
