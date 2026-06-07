from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOC_PATHS = [
    "README.md",
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "retail_ops/README.md",
    "retail_ops/ARCHITECTURE.md",
    "retail_ops/EXPERIMENT_RESULTS.md",
    "retail_ops/COMPARABILITY_GATE_V0.md",
]

REQUIRED_FILES = [
    "retail_ops/outputs/generated_demo2_retail_memory_facts.json",
    "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "eval/eval_retail_demo2_facts.py",
    "eval/eval_retail_demo2_answer_behavior.py",
    "eval/results/eval_retail_demo2_facts_result.txt",
    "eval/results/eval_retail_demo2_answer_behavior_result.txt",
]

REQUIRED_API_PATTERNS = [
    '@app.post("/chat_retail_ops_kb")',
    '@app.post("/chat_retail_ops_demo2_kb")',
]

REQUIRED_BOUNDARY_PHRASES = [
    "Demo 2",
    "file-backed",
    "generated retail memory facts",
    "same-period",
    "future comparability gate",
]

FORBIDDEN_OVERCLAIMS = [
    "is a production retail API endpoint",
    "production retail API endpoint is implemented",
    "production-ready retail API endpoint",
    "full 48-store automation is implemented",
    "comparability gate is implemented",
    "implemented comparability gate",
    "finished comparability gate",
]


def read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def main() -> int:
    failures: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).exists():
            failures.append(f"Missing Demo 2 evidence file: {relative_path}")

    api_text = read("api/main.py")
    for pattern in REQUIRED_API_PATTERNS:
        if pattern not in api_text:
            failures.append(f"api/main.py missing expected local retail endpoint pattern: {pattern}")

    combined_docs = ""
    for relative_path in DOC_PATHS:
        path = ROOT / relative_path
        if not path.exists():
            failures.append(f"Missing doc file: {relative_path}")
            continue

        text = read(relative_path)
        combined_docs += "\n" + text

        lowered = text.lower()
        for claim in FORBIDDEN_OVERCLAIMS:
            if claim.lower() in lowered:
                failures.append(f"{relative_path} contains overclaim: {claim}")

    lowered_docs = combined_docs.lower()
    for phrase in REQUIRED_BOUNDARY_PHRASES:
        if phrase.lower() not in lowered_docs:
            failures.append(f"Docs missing Demo 2 endpoint-boundary phrase: {phrase}")

    if failures:
        print("Demo 2 retail endpoint-boundary validation FAILED.")
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1

    print("Demo 2 retail endpoint-boundary validation PASSED.")
    print("Checked local retail endpoints exist in api/main.py.")
    print("Checked Demo 2 generated facts and evaluation artifacts.")
    print("Checked docs describe Demo 2 as file-backed/local evidence.")
    print("Checked docs do not claim a finished comparability gate or full 48-store automation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
