#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retail_ops.ingestion.contracts import (  # noqa: E402
    load_dataset_contracts,
    validate_dataset_contracts,
)


def main() -> int:
    contracts = load_dataset_contracts(ROOT)
    overlaps = validate_dataset_contracts(ROOT)

    print("Retail dataset contract validation PASSED.")
    print(f"Checked registered datasets: {len(contracts)}")

    for overlap in overlaps:
        print(
            "Checked non-additive overlap: "
            f"{overlap.left_dataset_id} <-> "
            f"{overlap.right_dataset_id}; "
            f"records={overlap.record_count}; "
            f"policy={overlap.overlap_policy}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
