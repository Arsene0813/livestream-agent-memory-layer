from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rac.src.grounded_pipeline import run_grounded_pipeline, save_grounded_outputs


REQUIRED_REPORT_SECTIONS = [
    "# Grounded RAC Report",
    "## 1. Direct Answer",
    "## 2. Question Type",
    "## 3. Factor Weights",
    "## 4. Local Evidence Grounding",
    "## 5. Competing Hypotheses",
    "## 6. Critic Findings",
    "## 7. Claim and Definition Check",
    "## 8. Final Judgment",
    "## 9. Evidence-Coverage Score",
    "## 10. What Cannot Be Concluded",
    "## 11. Review-State Update",
]

ALLOWED_GROUNDING_STATUSES = {
    "keyword_matched",
    "boundary_matched",
    "source_found_no_keyword_match"
}


def fail(message: str) -> None:
    raise SystemExit(f"[RAC grounded pipeline validation failed] {message}")


def load_eval_cases() -> list[dict]:
    path = ROOT / "rac" / "eval" / "rac_eval_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    cases = load_eval_cases()
    output_dir = ROOT / "rac" / "outputs"

    if not cases:
        fail("No eval cases found")

    total_packets = 0
    total_keyword_matches = 0
    total_boundary_matches = 0
    total_fallbacks = 0

    for case in cases:
        state = run_grounded_pipeline(case["question"], root=ROOT)
        save_grounded_outputs(state, output_dir, case["case_id"])

        if "grounded_evidence" not in state:
            fail(f"{case['case_id']} missing grounded_evidence")

        summary = state["grounded_evidence"]["summary"]
        rows = state.get("grounded_evidence_rows", [])

        if summary["total_packets"] == 0:
            fail(f"{case['case_id']} has zero grounded packets")

        if summary["source_missing_count"] != 0:
            fail(f"{case['case_id']} has missing sources")

        if not rows:
            fail(f"{case['case_id']} has no grounded evidence rows")

        for row in rows:
            if not row["source_path"]:
                fail(f"{case['case_id']} has row without source path")

            if not row["snippet"]:
                fail(f"{case['case_id']} has row without local snippet")

            if not row.get("grounding_role"):
                fail(f"{case['case_id']} has row without grounding_role")

            if row["grounding_status"] not in ALLOWED_GROUNDING_STATUSES:
                fail(
                    f"{case['case_id']} has invalid grounding status: "
                    f"{row['grounding_status']}"
                )

        report = state["final_report"]

        for section in REQUIRED_REPORT_SECTIONS:
            if section not in report:
                fail(f"{case['case_id']} report missing section: {section}")

        if "Source Lines" not in report or "Evidence Fields" not in report:
            fail(
                f"{case['case_id']} report does not expose "
                "source-line audit pointers and evidence fields"
            )

        if "Missing source files: 0" not in report:
            fail(f"{case['case_id']} report does not show zero missing sources")

        total_packets += summary["total_packets"]
        total_keyword_matches += summary["keyword_matched_count"]
        total_boundary_matches += summary.get("boundary_matched_count", 0)
        total_fallbacks += summary["fallback_count"]

    print("[OK] RAC grounded pipeline validation passed")
    print(f"[OK] Eval cases checked: {len(cases)}")
    print(f"[OK] Total grounded packets: {total_packets}")
    print(f"[OK] Keyword matched packets: {total_keyword_matches}")
    print(f"[OK] Boundary matched packets: {total_boundary_matches}")
    print(f"[OK] Fallback packets: {total_fallbacks}")
    print(f"[OK] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
