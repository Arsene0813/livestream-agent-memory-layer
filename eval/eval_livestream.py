import json
import uuid
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
CASES_FILE = Path(__file__).parent / "eval_livestream_cases.json"


def load_cases():
    with CASES_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def post_json(client, endpoint, payload):
    response = client.post(f"{BASE_URL}{endpoint}", json=payload)
    response.raise_for_status()
    return response.json()


def get_reply(result):
    return result.get("reply") or result.get("answer") or ""


def check_case(case, result):
    errors = []
    reply = get_reply(result)

    expected_refusal = case.get("expected_refusal")
    if expected_refusal is not None:
        actual_refusal = result.get("refusal")
        if actual_refusal != expected_refusal:
            errors.append(f"refusal expected {expected_refusal}, got {actual_refusal}")

    expected_type = case.get("expected_routed_fact_type")
    if expected_type is not None:
        actual_type = result.get("routed_fact_type")
        if actual_type != expected_type:
            errors.append(f"routed_fact_type expected {expected_type}, got {actual_type}")

    for text in case.get("expected_contains", []):
        if text not in reply:
            errors.append(f"reply does not contain expected text: {text}")

    for text in case.get("forbidden_contains", []):
        if text in reply:
            errors.append(f"reply contains forbidden text: {text}")

    if case.get("requires_sources"):
        sources = result.get("sources") or []
        if not sources:
            errors.append("expected non-empty sources, got empty sources")

    return len(errors) == 0, errors


def run_case(client, case):
    session_id = case.get("session_id") or f"eval-{case['id']}-{uuid.uuid4().hex[:8]}"

    setup_endpoint = case.get("setup_endpoint", "/chat_mem")
    query_endpoint = case.get("query_endpoint", "/chat_livestream_kb")

    for message in case.get("setup_messages", []):
        setup_payload = {
            "session_id": session_id,
            "message": message,
            "temperature": 0
        }
        setup_result = post_json(client, setup_endpoint, setup_payload)

        if case.get("require_setup_fact", True):
            fact_debug = setup_result.get("fact_debug") or {}
            if fact_debug and fact_debug.get("fact_ready") is False:
                raise RuntimeError(f"setup message did not produce a fact: {message}")

    query_payload = {
        "session_id": session_id,
        "message": case["query"],
        "temperature": 0,
        "top_k": case.get("top_k", 3)
    }

    if "score_threshold" in case:
        query_payload["score_threshold"] = case["score_threshold"]

    return post_json(client, query_endpoint, query_payload)


def main():
    cases = load_cases()

    passed = 0
    failed = 0

    with httpx.Client(timeout=120) as client:
        for case in cases:
            case_id = case["id"]

            try:
                result = run_case(client, case)
                ok, errors = check_case(case, result)
            except Exception as e:
                ok = False
                errors = [f"exception: {e}"]
                result = {}

            if ok:
                print(f"[PASS] {case_id}")
                passed += 1
            else:
                print(f"[FAIL] {case_id}")
                for err in errors:
                    print(f"  - {err}")
                if result:
                    print(f"  - reply: {get_reply(result)}")
                    print(f"  - routed_fact_type: {result.get('routed_fact_type')}")
                    print(f"  - refusal: {result.get('refusal')}")
                failed += 1

    print()
    print("=== Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total: {passed + failed}")


if __name__ == "__main__":
    main()
