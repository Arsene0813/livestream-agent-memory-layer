from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "retail_ops" / "outputs" / "demo2_cross_store_comparability_output.csv"
OUTPUT_CSV_PATH = ROOT / "retail_ops" / "outputs" / "demo2_guardrail_sensitivity_summary.csv"
OUTPUT_TXT_PATH = ROOT / "retail_ops" / "outputs" / "demo2_guardrail_sensitivity_result.txt"



def repo_rel(path: Path) -> str:
    """Return a repository-relative path for reproducible committed outputs."""
    return path.relative_to(ROOT).as_posix()

BASE_THRESHOLDS = {
    "activity_high": 80.0,
    "activity_moderate": 65.0,
    "refund_high": 15.0,
    "refund_moderate": 10.0,
    "invalid_high": 12.0,
    "invalid_moderate": 8.0,
    "top3_sku_share_high": 25.0,
}


SCENARIOS = {
    "looser_minus_5pp": -5.0,
    "baseline_sql_thresholds": 0.0,
    "stricter_plus_5pp": 5.0,
}


REQUIRED_COLUMNS = {
    "store_id",
    "activity_order_share_pct",
    "refund_pressure_pct",
    "invalid_order_pressure_pct",
    "top3_sku_transaction_amount_share_pct",
}


def parse_float(row: dict[str, str], field: str) -> float | None:
    raw = (row.get(field) or "").strip()
    if raw == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def shifted_threshold(name: str, shift: float) -> float:
    value = BASE_THRESHOLDS[name] + shift
    return max(value, 0.0)


def guardrail_notes(row: dict[str, str], shift: float) -> list[str]:
    notes: list[str] = []
    activity_share = parse_float(row, "activity_order_share_pct")
    refund_pressure = parse_float(row, "refund_pressure_pct")
    invalid_pressure = parse_float(row, "invalid_order_pressure_pct")
    top3_share = parse_float(row, "top3_sku_transaction_amount_share_pct")
    if activity_share is not None:
        if activity_share >= shifted_threshold("activity_high", shift):
            notes.append("high_activity_involvement")
        elif activity_share >= shifted_threshold("activity_moderate", shift):
            notes.append("moderate_activity_involvement")

    if refund_pressure is not None:
        if refund_pressure >= shifted_threshold("refund_high", shift):
            notes.append("high_refund_pressure")
        elif refund_pressure >= shifted_threshold("refund_moderate", shift):
            notes.append("moderate_refund_pressure")

    if invalid_pressure is not None:
        if invalid_pressure >= shifted_threshold("invalid_high", shift):
            notes.append("high_invalid_order_pressure")
        elif invalid_pressure >= shifted_threshold("invalid_moderate", shift):
            notes.append("moderate_invalid_order_pressure")

    if (
        top3_share is not None
        and top3_share >= shifted_threshold("top3_sku_share_high", shift)
    ):
        notes.append("top3_sku_amount_concentration")

    if not notes:
        notes.append("no_threshold_guardrail_triggered")

    return notes


def load_rows() -> list[dict[str, str]]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {repo_rel(INPUT_PATH)}")

    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - headers)
        if missing:
            raise ValueError(f"Missing required columns in {repo_rel(INPUT_PATH)}: {missing}")
        return list(reader)


def main() -> int:
    rows = load_rows()

    output_rows: list[dict[str, Any]] = []
    fragile_store_ids: set[str] = set()

    baseline_by_store = {
        row["store_id"]: guardrail_notes(row, SCENARIOS["baseline_sql_thresholds"])
        for row in rows
    }

    for row in rows:
        store_id = row["store_id"]
        baseline_notes = baseline_by_store[store_id]

        for scenario_name, shift in SCENARIOS.items():
            notes = guardrail_notes(row, shift)
            changed = notes != baseline_notes

            if scenario_name != "baseline_sql_thresholds" and changed:
                fragile_store_ids.add(store_id)

            output_rows.append(
                {
                    "store_id": store_id,
                    "scenario": scenario_name,
                    "threshold_shift_pp": shift,
                    "activity_order_share_pct": row.get("activity_order_share_pct", ""),
                    "refund_pressure_pct": row.get("refund_pressure_pct", ""),
                    "invalid_order_pressure_pct": row.get("invalid_order_pressure_pct", ""),
                    "top3_sku_transaction_amount_share_pct": row.get(
                        "top3_sku_transaction_amount_share_pct", ""
                    ),
                    "triggered_guardrail_notes": ";".join(notes),
                    "changed_from_baseline": "true" if changed else "false",
                }
            )

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "store_id",
                "scenario",
                "threshold_shift_pp",
                            "activity_order_share_pct",
                "refund_pressure_pct",
                "invalid_order_pressure_pct",
                "top3_sku_transaction_amount_share_pct",
                "triggered_guardrail_notes",
                "changed_from_baseline",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    lines = [
        "Demo 2 guardrail sensitivity check completed.",
        f"Input: {repo_rel(INPUT_PATH)}",
        f"Output CSV: {repo_rel(OUTPUT_CSV_PATH)}",
        "",
        "Interpretation:",
        "- This check does not optimize thresholds.",
        "- It tests whether current Demo 2 guardrail notes are fragile under +/- 5 percentage-point threshold shifts.",
        "- Any changed store should be treated as threshold-sensitive evidence, not as a stable peer-comparison rule.",
        "",
        f"Stores with changed guardrail notes under sensitivity scenarios: {', '.join(sorted(fragile_store_ids)) if fragile_store_ids else 'none'}",
    ]

    OUTPUT_TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
