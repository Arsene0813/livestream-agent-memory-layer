from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_TERMS = {
    "retail_ops/FIELD_USAGE_REVIEW.md": [
        "Consolidated Scope Notes",
        "DATA_DICTIONARY.md",
        "Demo 1 remains a Store A month-over-month diagnostic",
        "Demo 2 remains a same-period B-F diagnostic",
        "not a completed pairwise comparability gate",
        "region_type",
        "weak region or market-context evidence",
        "deterministic source-aware review scaffold",
    ],
    "README.md": [
        "Editing and Scope Guardrails",
        "DATA_DICTIONARY.md",
        "FIELD_USAGE_REVIEW.md",
        "COMPARABILITY_GATE_V0.md",
    ],
}


def main() -> int:
    failures = []

    for rel_path, required_terms in REQUIRED_TERMS.items():
        path = ROOT / rel_path
        if not path.exists():
            failures.append(f"Missing required guardrail file: {rel_path}")
            continue

        text = path.read_text(encoding="utf-8")
        for term in required_terms:
            if term not in text:
                failures.append(f"{rel_path}: missing required term: {term}")

    if failures:
        print("Narrative / scope guardrail validation FAILED.")
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("[OK] Consolidated narrative and scope guardrails validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
