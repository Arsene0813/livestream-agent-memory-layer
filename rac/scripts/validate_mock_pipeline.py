from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rac.src.mock_pipeline import run_mock_pipeline, save_state_outputs


REQUIRED_REPORT_SECTIONS = [
    "## 1. Direct Answer",
    "## 2. Question Type",
    "## 3. Relevant Factors Considered",
    "## 4. Evidence Used",
    "## 5. Competing Hypotheses",
    "## 6. Critic Findings",
    "## 7. Final Judgment",
    "## 8. Scenario-Template Confidence",
    "## 9. What Cannot Be Concluded",
    "## 10. Review-State Update",
]


def fail(message: str) -> None:
    raise SystemExit(f"[RAC mock pipeline validation failed] {message}")


def normalize(text: str) -> str:
    return text.lower().replace("-", " ").replace("_", " ")


def main() -> None:
    eval_path = ROOT / "rac" / "eval" / "rac_eval_cases.json"
    output_dir = ROOT / "rac" / "outputs"
    cases = json.loads(eval_path.read_text(encoding="utf-8"))

    if not cases:
        fail("No eval cases found")

    for case in cases:
        state = run_mock_pipeline(case["question"])
        save_state_outputs(state, output_dir, case["case_id"])

        if state["question_type"] != case["question_type"]:
            fail(f"{case['case_id']} expected {case['question_type']}, got {state['question_type']}")

        factor_ids = {factor["factor_id"] for factor in state["factors"]}
        missing = [factor for factor in case["must_include_factors"] if factor not in factor_ids]
        if missing:
            fail(f"{case['case_id']} missing factors: {missing}")

        if not state["hypotheses"]:
            fail(f"{case['case_id']} has no hypotheses")

        if not state["critic_findings"]:
            fail(f"{case['case_id']} has no critic findings")

        final_report = state["final_report"]

        for section in REQUIRED_REPORT_SECTIONS:
            if section not in final_report:
                fail(f"{case['case_id']} final report missing section: {section}")

        normalized_report = normalize(final_report)
        for banned_claim in case["must_not_claim"]:
            if normalize(banned_claim) in normalized_report:
                fail(f"{case['case_id']} final report contains banned claim: {banned_claim}")

    print("[OK] RAC deterministic mock pipeline validation passed")
    print(f"[OK] Eval cases checked: {len(cases)}")
    print(f"[OK] Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
