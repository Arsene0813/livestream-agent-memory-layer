from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.main import RetailOpsDemo2KbReq, chat_retail_ops_demo2_kb  # noqa: E402


RESULT_PATH = ROOT / "eval" / "results" / "eval_retail_demo2_endpoint_behavior_result.txt"


def stringify(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True).lower()


def require_supported(name: str, result: dict[str, Any]) -> None:
    if result.get("supported") is not True:
        raise AssertionError(f"{name}: expected supported=True, got {result}")


def require_refusal(name: str, result: dict[str, Any]) -> None:
    if result.get("supported") is not False:
        raise AssertionError(f"{name}: expected supported=False, got {result}")


def require_demo2_endpoint_metadata(name: str, result: dict[str, Any]) -> None:
    if result.get("demo_scope") != "demo2_cross_store":
        raise AssertionError(
            f"{name}: expected demo_scope='demo2_cross_store', "
            f"got {result.get('demo_scope')!r}"
        )

    if result.get("retrieval_mode") != "file_backed_retail_memory_facts":
        raise AssertionError(
            f"{name}: expected retrieval_mode='file_backed_retail_memory_facts', "
            f"got {result.get('retrieval_mode')!r}"
        )


def require_slot(name: str, result: dict[str, Any], expected_slot: str) -> None:
    slots = [fact.get("slot") for fact in result.get("facts", [])]
    if expected_slot not in slots:
        raise AssertionError(
            f"{name}: expected slot {expected_slot!r}, got slots={slots}"
        )


def require_all_slots(name: str, result: dict[str, Any], expected_slot: str) -> None:
    slots = [fact.get("slot") for fact in result.get("facts", [])]
    if not slots:
        raise AssertionError(f"{name}: expected facts, got none")

    bad_slots = [slot for slot in slots if slot != expected_slot]
    if bad_slots:
        raise AssertionError(
            f"{name}: expected all returned slots to be {expected_slot!r}, "
            f"got slots={slots}"
        )


def require_contains(
    name: str,
    result: dict[str, Any],
    required_terms: list[str],
) -> None:
    text = stringify(result)
    missing = [term for term in required_terms if term.lower() not in text]
    if missing:
        raise AssertionError(
            f"{name}: missing required terms {missing}; result={result}"
        )


async def ask(
    message: str,
    entity_id: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    req = RetailOpsDemo2KbReq(
        message=message,
        entity_id=entity_id,
        top_k=top_k,
    )
    return await chat_retail_ops_demo2_kb(req)


async def run_checks() -> int:
    passed: list[str] = []
    failed: list[str] = []

    async def run_case(name: str, case_func) -> None:
        try:
            await case_func()
            passed.append(f"[PASS] {name}")
        except AssertionError as exc:
            failed.append(f"[FAIL] {name}: {exc}")

    async def case_store_e_transaction_conversion_profile() -> None:
        result = await ask(
            "For Store E in Demo 2, explain the March 2026 transaction and conversion profile. Do not make a final operating recommendation.",
            entity_id="store_E",
        )
        assert_contains(
            result,
            [
                "transaction_conversion_profile",
                "transaction_amount",
                "transaction_orders",
                "order_conversion_rate_pct",
                "payment_conversion_rate_pct",
            ],
            "Store E transaction-conversion endpoint behavior",
        )
        assert_not_contains(
            result,
            [
                "best store",
                "should copy",
            ],
            "Store E transaction-conversion endpoint behavior",
        )
