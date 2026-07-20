from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rac.src.local_evidence_resolver import resolve_state_evidence
from rac.src.mock_pipeline import run_mock_pipeline


def fail(message: str) -> None:
    raise SystemExit(f"[RAC local evidence resolver validation failed] {message}")


def load_eval_cases() -> list[dict]:
    path = ROOT / "rac" / "eval" / "rac_eval_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


PROMOTION_BOUNDARY_FACTORS = {
    "sku_margin_structure",
    "competitor_context",
}


def validate_promotion_boundary_routing(
    case_id: str,
    packets: list[dict],
) -> None:
    if case_id != "rac_promotion_strategy_001":
        return

    packets_by_factor = {
        packet["factor_id"]: packet
        for packet in packets
    }

    missing = sorted(
        PROMOTION_BOUNDARY_FACTORS
        - set(packets_by_factor)
    )

    if missing:
        fail(
            "Promotion case is missing factors: "
            f"{missing}"
        )

    actual_boundary = {
        factor_id
        for factor_id, packet in packets_by_factor.items()
        if packet["grounding_status"] == "boundary_matched"
    }

    if actual_boundary != PROMOTION_BOUNDARY_FACTORS:
        fail(
            "Promotion boundary routing mismatch. "
            f"Expected {sorted(PROMOTION_BOUNDARY_FACTORS)}, "
            f"found {sorted(actual_boundary)}"
        )

    for factor_id in sorted(PROMOTION_BOUNDARY_FACTORS):
        role = packets_by_factor[factor_id].get(
            "grounding_role"
        )

        if role != "boundary_evidence":
            fail(
                f"{factor_id} must use boundary_evidence, "
                f"found {role}"
            )


def main() -> None:
    cases = load_eval_cases()

    if not cases:
        fail("No eval cases found")

    total_packets = 0
    total_keyword_matches = 0
    total_fallbacks = 0

    for case in cases:
        state = run_mock_pipeline(case["question"])
        resolved = resolve_state_evidence(state, root=ROOT)

        summary = resolved["summary"]
        packets = resolved["resolved_packets"]

        validate_promotion_boundary_routing(
            case["case_id"],
            packets,
        )

        if summary["total_packets"] == 0:
            fail(f"{case['case_id']} produced no evidence packets")

        if summary["source_missing_count"] > 0:
            missing = [
                packet["source_path"]
                for packet in packets
                if packet["grounding_status"] == "source_missing"
            ]
            fail(f"{case['case_id']} has missing source files: {missing}")

        for packet in packets:
            if not packet["snippets"]:
                fail(
                    f"{case['case_id']} packet {packet['evidence_id']} "
                    "has no snippets"
                )

            for snippet in packet["snippets"]:
                if not snippet["text"].strip():
                    fail(
                        f"{case['case_id']} packet {packet['evidence_id']} "
                        "has empty snippet text"
                    )

        if summary["keyword_matched_count"] == 0:
            fail(f"{case['case_id']} has zero keyword-matched packets")

        total_packets += summary["total_packets"]
        total_keyword_matches += summary["keyword_matched_count"]
        total_fallbacks += summary["fallback_count"]

    print("[OK] RAC local evidence resolver validation passed")
    print(f"[OK] Eval cases checked: {len(cases)}")
    print(f"[OK] Total evidence packets: {total_packets}")
    print(f"[OK] Keyword matched packets: {total_keyword_matches}")
    print(f"[OK] Fallback packets: {total_fallbacks}")


if __name__ == "__main__":
    main()
