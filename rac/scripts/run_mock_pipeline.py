from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rac.src.mock_pipeline import run_mock_pipeline, save_state_outputs


def load_eval_cases() -> list[dict]:
    path = ROOT / "rac" / "eval" / "rac_eval_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic RAC mock pipeline.")
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--all-eval", action="store_true")
    args = parser.parse_args()

    output_dir = ROOT / "rac" / "outputs"

    if args.all_eval:
        for case in load_eval_cases():
            state = run_mock_pipeline(case["question"])
            paths = save_state_outputs(state, output_dir, case["case_id"])
            print(f"[OK] {case['case_id']}")
            print(f"  JSON: {paths['json']}")
            print(f"  MD:   {paths['markdown']}")
        return

    if args.case_id:
        cases = {case["case_id"]: case for case in load_eval_cases()}
        if args.case_id not in cases:
            raise SystemExit(f"Unknown case id: {args.case_id}")
        state = run_mock_pipeline(cases[args.case_id]["question"])
        paths = save_state_outputs(state, output_dir, args.case_id)
        print(f"[OK] {args.case_id}")
        print(f"JSON: {paths['json']}")
        print(f"MD:   {paths['markdown']}")
        return

    if args.question:
        state = run_mock_pipeline(args.question)
        paths = save_state_outputs(state, output_dir)
        print("[OK] Custom question")
        print(f"JSON: {paths['json']}")
        print(f"MD:   {paths['markdown']}")
        return

    raise SystemExit("Provide --question, --case-id, or --all-eval")


if __name__ == "__main__":
    main()
