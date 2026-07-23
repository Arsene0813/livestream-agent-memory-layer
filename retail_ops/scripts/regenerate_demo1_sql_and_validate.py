from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_OUTPUTS = (
    ROOT
    / "retail_ops/outputs/store_a_demo1_sql_output.csv",
    ROOT
    / "retail_ops/outputs/"
    "store_a_demo1_interpretation_summary.csv",
    ROOT
    / "retail_ops/outputs/generated_retail_memory_facts.json",
)


def run(script, *arguments):
    command = [
        sys.executable,
        script,
        *arguments,
    ]

    print()
    print("$ " + " ".join(command))

    subprocess.run(
        command,
        cwd=ROOT,
        check=True,
    )


def main():
    print(
        "Regenerating Demo 1 outputs and validating "
        "their evidence contracts."
    )

    run(
        "retail_ops/scripts/export_demo1_sql_output.py"
    )

    run(
        "retail_ops/scripts/"
        "sync_demo1_interpretation_summary.py",
        "--write",
    )

    for path in REQUIRED_OUTPUTS:
        if not path.exists():
            raise SystemExit(
                "Missing required output: "
                + str(path.relative_to(ROOT))
            )

    run(
        "retail_ops/scripts/"
        "validate_retail_data_contract.py"
    )

    run(
        "retail_ops/scripts/"
        "sync_demo1_interpretation_summary.py",
        "--check",
    )

    run(
        "retail_ops/scripts/"
        "validate_demo1_value_lineage.py"
    )

    run(
        "retail_ops/scripts/"
        "validate_csv_physical_rows.py"
    )

    print()
    print(
        "Demo 1 regeneration and validation "
        "completed successfully."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
