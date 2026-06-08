from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Maintainer note:
# Required terms below are contract anchors, not writing templates.
# They protect scope, file presence, and metric-boundary facts without forcing
# repeated explanatory disclaimers across every Markdown file.


REVIEWER_FACING_PATHS = [
    "README.md",
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "retail_ops/README.md",
    "retail_ops/data/DATA_DICTIONARY.md",
    "retail_ops/FIELD_USAGE_REVIEW.md",
    "retail_ops/LINEAGE.md",
    "retail_ops/ARCHITECTURE.md",
    "retail_ops/EXPERIMENTS.md",
    "retail_ops/EXPERIMENT_RESULTS.md",
    "retail_ops/EXPERIMENT_REVIEW_MAP.md",
    "retail_ops/COMPARABILITY_GATE_V0.md",
    "retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md",
    "retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md",
]

REQUIRED_FILES = [
    "README.md",
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "retail_ops/README.md",
    "retail_ops/ARCHITECTURE.md",
    "retail_ops/EXPERIMENT_RESULTS.md",
    "retail_ops/EXPERIMENTS.md",
    "retail_ops/COMPARABILITY_GATE_V0.md",
    "retail_ops/FIELD_USAGE_REVIEW.md",
    "retail_ops/LINEAGE.md",
    "retail_ops/data/DATA_DICTIONARY.md",
    "retail_ops/sql/01_store_a_month_over_month_diagnostic.sql",
    "retail_ops/sql/02_demo2_cross_store_comparability.sql",
    "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "retail_ops/outputs/retail_data_contract_validation_result.txt",
    "scripts/validate_demo2_retail_endpoint_boundary.py",
    "eval/eval_retail_demo2_facts.py",
    "eval/eval_retail_demo2_answer_behavior.py",
    "eval/eval_retail_demo2_endpoint_behavior.py",
]


CURRENT_SCOPE_REQUIRED_TERMS = {
    "README.md": [
    "Retail Demo 2",
    "pairwise comparability gate",
],
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md": [
        "pairwise comparability gate",
        "staged local prototype",
    ],
    "retail_ops/README.md": [
        "same-period",
        "generated retail memory facts",
    ],
    "retail_ops/COMPARABILITY_GATE_V0.md": [
        "transaction order volume",
        "transaction amount",
        "activity status",
        "activity intensity",
        "region_type",
        "repeated reporting windows",
    ],
    "retail_ops/data/DATA_DICTIONARY.md": [
        "region_type remains weak context only",
        "not a hard market-area classification",
        "Pairwise comparability-gate fields are not currently implemented",
    ],
}


REQUIRED_MARKDOWN_TABLE_PATTERNS = {
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md": [
        "| Operating step | Practical meaning |",
        "| Layer | What it does |",
        "| Field | Boundary |",
    ],
    "retail_ops/README.md": [
        "| Component | Purpose |",
        "|---|---|",
    ],
    "retail_ops/ARCHITECTURE.md": [
        "| Demo | Status | Purpose |",
        "| Layer | Input | Output | Boundary |",
    ],
    "retail_ops/COMPARABILITY_GATE_V0.md": [
        "| Future factor | Current evidence available | Current limitation | Future evidence needed |",
    ],
}


REQUIRED_API_PATTERNS = [
    '@app.post("/chat_retail_ops_kb")',
    '@app.post("/chat_retail_ops_demo2_kb")',
]


FORBIDDEN_DOC_OVERCLAIMS = [
    "is a production retail API endpoint",
    "production retail API endpoint is implemented",
    "production-ready retail API endpoint",
    "full 48-store automation is implemented",
    "comparability gate is implemented",
    "implemented comparability gate",
    "finished comparability gate",
    "finished pairwise comparability gate",
]


STALE_TERMS = [
    "Demo 3",
    "demo3",
    "demo_3_",
    "demo3_",
    "run_demo3",
    "validate_demo3",
    "eval_retail_demo3",
    "03_demo",
]


SCAN_SUFFIXES = {
    ".md",
    ".py",
    ".sql",
    ".csv",
    ".json",
    ".txt",
    ".yml",
    ".yaml",
}


SKIP_STALE_SCAN_FILES = {
    "scripts/validate_project_consistency.py",
    "retail_ops/scripts/validate_retail_data_contract.py",
    "scripts/validate_demo2_retail_endpoint_boundary.py",
}


def git_tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        ROOT / line
        for line in result.stdout.splitlines()
        if (ROOT / line).exists()
    ]
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in SCAN_SUFFIXES



def check_reviewer_facing_paths(failures: list[str]) -> None:
    """Ensure the admissions/reviewer reading path points to existing files."""

    for rel_path in REVIEWER_FACING_PATHS:
        if not (ROOT / rel_path).exists():
            failures.append(f"Missing reviewer-facing path: {rel_path}")


def check_required_files(failures: list[str]) -> None:
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not path.exists():
            failures.append(f"Missing required current-scope file: {rel}")


def check_required_terms(failures: list[str]) -> None:
    for rel, required_terms in CURRENT_SCOPE_REQUIRED_TERMS.items():
        path = ROOT / rel
        if not path.exists():
            failures.append(f"Missing file for content check: {rel}")
            continue

        text = read_text(path)
        for term in required_terms:
            if term not in text:
                failures.append(f"{rel} missing required current-scope term: {term}")


def normalize_markdown_table_text(text: str) -> str:
    normalized_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if not stripped.startswith("|"):
            normalized_lines.append(stripped)
            continue

        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        normalized_cells: list[str] = []

        for cell in cells:
            compact = cell.replace(" ", "")
            if compact and "-" in compact and set(compact) <= {"-", ":"}:
                normalized_cells.append("---")
            else:
                normalized_cells.append(cell)

        normalized_lines.append("| " + " | ".join(normalized_cells) + " |")

    return "\n".join(normalized_lines)


def check_markdown_tables(failures: list[str]) -> None:
    for rel, required_patterns in REQUIRED_MARKDOWN_TABLE_PATTERNS.items():
        path = ROOT / rel
        if not path.exists():
            failures.append(f"Missing file for Markdown table check: {rel}")
            continue

        normalized_text = normalize_markdown_table_text(read_text(path))

        for pattern in required_patterns:
            normalized_pattern = normalize_markdown_table_text(pattern)

            if normalized_pattern not in normalized_text:
                failures.append(f"{rel} missing Markdown table pattern: {pattern}")


def check_api_patterns(failures: list[str]) -> None:
    api_path = ROOT / "api/main.py"
    if not api_path.exists():
        failures.append("Missing api/main.py")
        return

    api_text = read_text(api_path)
    for pattern in REQUIRED_API_PATTERNS:
        if pattern not in api_text:
            failures.append(f"api/main.py missing expected local retail endpoint pattern: {pattern}")


def check_doc_overclaims(failures: list[str]) -> None:
    docs_to_check = [
        "README.md",
        "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
        "retail_ops/README.md",
        "retail_ops/ARCHITECTURE.md",
        "retail_ops/EXPERIMENT_RESULTS.md",
    "retail_ops/EXPERIMENTS.md",
        "retail_ops/COMPARABILITY_GATE_V0.md",
        "retail_ops/FIELD_USAGE_REVIEW.md",
    ]

    for rel in docs_to_check:
        path = ROOT / rel
        if not path.exists():
            continue

        lowered = read_text(path).lower()
        for claim in FORBIDDEN_DOC_OVERCLAIMS:
            if claim.lower() in lowered:
                failures.append(f"{rel} contains overclaim: {claim}")


def check_stale_terms(failures: list[str]) -> None:
    for path in git_tracked_files():
        if not is_text_file(path):
            continue

        rel = path.relative_to(ROOT).as_posix()
        if rel in SKIP_STALE_SCAN_FILES:
            continue

        try:
            text = read_text(path)
        except UnicodeDecodeError:
            continue

        lowered = text.lower()
        for term in STALE_TERMS:
            if term.lower() in lowered:
                failures.append(f"{rel} still contains stale Demo 3 term: {term}")


def check_dictionary_boundaries(failures: list[str]) -> None:
    dictionary_path = ROOT / "retail_ops/data/DATA_DICTIONARY.md"
    if not dictionary_path.exists():
        failures.append("Missing retail_ops/data/DATA_DICTIONARY.md")
        return

    dictionary = read_text(dictionary_path)
    dictionary_required_boundaries = [
        "activity_cost_ratio_pct",
        "not traditional ROI",
        "estimated_income_proxy",
        "not audited profit",
        "order_conversion_rate_pct",
        "order_users / entry_users",
        "top3_sku_transaction_amount_share_pct",
        "not full product-category share",
        "region_type remains weak context only",
    ]

    for term in dictionary_required_boundaries:
        if term not in dictionary:
            failures.append(f"DATA_DICTIONARY.md missing boundary term: {term}")


def main() -> int:
    failures: list[str] = []

    check_required_files(failures)
    check_reviewer_facing_paths(failures)
    check_required_terms(failures)
    check_markdown_tables(failures)
    check_api_patterns(failures)
    check_doc_overclaims(failures)
    check_stale_terms(failures)
    check_dictionary_boundaries(failures)

    if failures:
        print("Project consistency validation FAILED.")
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("[PASS] Project consistency validation passed.")
    print("[PASS] Retail Demo 2 is the current same-period diagnostic endpoint.")
    print("[PASS] Local retail endpoints exist in api/main.py.")
    print("[PASS] Demo 2 is documented as file-backed/local evidence.")
    print("[PASS] Comparability gate is documented as future work.")
    print("[PASS] Markdown tables required for review readability are present.")
    print("[PASS] Stale Demo 3 references are absent from tracked text artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
