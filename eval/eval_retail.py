import json
import os
import sys
import urllib.request
from pathlib import Path

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
CASES_PATH = Path("eval/eval_retail_cases.json")
RESULTS_PATH = Path("eval/results/eval_retail_result.txt")


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def extract_returned_slots(result: dict) -> set[str]:
    facts = result.get("facts", [])
    if not isinstance(facts, list):
        return set()

    returned_slots = set()
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        slot = fact.get("slot")
        if isinstance(slot, str) and slot:
            returned_slots.add(slot)

    return returned_slots


def main() -> int:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    passed = 0
    failed = 0
    lines = []

    for case in cases:
        payload = {
            "message": case["message"],
            "entity_id": case.get("entity_id", "store_A"),
        }

        try:
            result = post_json(f"{API_BASE_URL}/chat_retail_ops_kb", payload)
        except Exception as exc:
            failed += 1
            lines.append(f"FAIL {case['name']}: request_error={exc}")
            continue

        answer = result.get("answer", "")
        supported = bool(result.get("supported"))

        support_ok = supported == bool(case["should_support"])

        substrings_ok = all(
            expected.lower() in answer.lower()
            for expected in case.get("expected_substrings", [])
        )

        returned_slots = extract_returned_slots(result)

        slots_ok = all(
            expected_slot in returned_slots
            for expected_slot in case.get("expected_slots", [])
        )

        if support_ok and substrings_ok and slots_ok:
            passed += 1
            lines.append(f"PASS {case['name']}")
        else:
            failed += 1
            lines.append(
                (
                    "FAIL {name}: supported={supported}, expected_supported={expected}, "
                    "returned_slots={returned_slots}, expected_slots={expected_slots}, "
                    "answer={answer}"
                ).format(
                    name=case["name"],
                    supported=supported,
                    expected=case["should_support"],
                    returned_slots=sorted(returned_slots),
                    expected_slots=case.get("expected_slots", []),
                    answer=answer.replace("\n", " ")[:500],
                )
            )

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    summary = [
        f"Retail eval result: {passed}/{len(cases)} passed",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
        *lines,
        "",
    ]

    RESULTS_PATH.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
