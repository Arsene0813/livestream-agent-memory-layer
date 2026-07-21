"""Tests for the shared retail retrieval evidence contract."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

try:
    from retrieval_contract_match import (
        expected_document_match,
        expected_hit_at_k,
    )
    from retail_retrieval_corpus import (
        load_retail_retrieval_documents,
    )
except ModuleNotFoundError:
    from eval.retrieval_contract_match import (
        expected_document_match,
        expected_hit_at_k,
    )
    from eval.retail_retrieval_corpus import (
        load_retail_retrieval_documents,
    )


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "eval" / "retrieval_threshold_cases.json"


def make_doc(
    *,
    entity_id: str = "store_B",
    slot: str = "visibility_entry_profile",
    period_label: str = "2026-03",
    text: str = (
        "search_entry_rate_pct "
        "search_entry_share_pct"
    ),
) -> dict[str, str]:
    return {
        "doc_id": "test-doc",
        "entity_id": entity_id,
        "entity": "legacy_wrong_value",
        "slot": slot,
        "period_label": period_label,
        "period_start": f"{period_label}-01",
        "period_end": f"{period_label}-31",
        "source_path": "test/source.csv",
        "text": text,
    }


def make_case(
    *,
    case_type: str = "positive_supported",
    expected_entity: str | None = "store_B",
    expected_slot: str | None = "visibility_entry_profile",
    expected_period_terms: list[str] | None = None,
    expected_terms: list[str] | None = None,
) -> dict[str, object]:
    return {
        "case_id": "test-case",
        "case_type": case_type,
        "query": "test query",
        "expected_entity": expected_entity,
        "expected_slot": expected_slot,
        "expected_period_terms": (
            ["2026-03"]
            if expected_period_terms is None
            else expected_period_terms
        ),
        "expected_terms": (
            [
                "search_entry_rate_pct",
                "search_entry_share_pct",
            ]
            if expected_terms is None
            else expected_terms
        ),
    }


class RetrievalContractMatchTest(unittest.TestCase):
    def test_uses_canonical_entity_id(self) -> None:
        case = make_case(expected_entity="store_B")
        doc = make_doc(entity_id="store_B")

        self.assertTrue(
            expected_document_match(case, doc)
        )

        wrong_case = make_case(expected_entity="store_C")

        self.assertFalse(
            expected_document_match(wrong_case, doc)
        )

    def test_requires_all_expected_terms(self) -> None:
        case = make_case(
            expected_terms=[
                "search_entry_rate_pct",
                "search_entry_share_pct",
            ]
        )

        partial_doc = make_doc(
            text="search_entry_rate_pct only"
        )

        self.assertFalse(
            expected_document_match(case, partial_doc)
        )

    def test_requires_expected_period(self) -> None:
        case = make_case(
            expected_period_terms=[
                "2026-04",
                "April 2026",
            ]
        )

        march_doc = make_doc(period_label="2026-03")

        self.assertFalse(
            expected_document_match(case, march_doc)
        )

    def test_negative_unsupported_never_counts_as_hit(self) -> None:
        case = make_case(
            case_type="negative_unsupported",
            expected_entity=None,
            expected_slot=None,
            expected_period_terms=[],
            expected_terms=[],
        )

        self.assertFalse(
            expected_document_match(case, make_doc())
        )

    def test_hit_at_k_does_not_use_partial_or_logic(self) -> None:
        case = make_case()

        wrong_entity = make_doc(entity_id="store_C")
        wrong_slot = make_doc(slot="activity_lever_profile")
        missing_term = make_doc(
            text="search_entry_rate_pct only"
        )
        wrong_period = make_doc(period_label="2026-04")

        self.assertFalse(
            expected_hit_at_k(
                case,
                [
                    wrong_entity,
                    wrong_slot,
                    missing_term,
                    wrong_period,
                ],
            )
        )

        self.assertTrue(
            expected_hit_at_k(
                case,
                [
                    wrong_entity,
                    make_doc(),
                ],
            )
        )

    def test_repository_positive_and_negative_examples(self) -> None:
        cases = json.loads(
            CASES_PATH.read_text(encoding="utf-8")
        )
        by_id = {
            case["case_id"]: case
            for case in cases
        }

        docs = load_retail_retrieval_documents()

        self.assertTrue(
            expected_hit_at_k(
                by_id["positive_store_b_visibility_entry"],
                docs,
            )
        )

        self.assertFalse(
            expected_hit_at_k(
                by_id["mismatch_store_b_april"],
                docs,
            )
        )

        self.assertFalse(
            expected_hit_at_k(
                by_id["negative_customer_lifetime_value"],
                docs,
            )
        )


if __name__ == "__main__":
    unittest.main()
