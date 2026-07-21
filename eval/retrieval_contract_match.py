"""Shared expected-evidence contract for retail retrieval experiments.

The case schema comes from ``eval/retrieval_threshold_cases.json``.
A retrieved document is an expected match only when it satisfies every
applicable constraint in that case:

- canonical ``entity_id``;
- slot;
- period;
- all expected evidence terms.

Semantic similarity or one matching keyword is not sufficient.
``negative_unsupported`` cases never have an expected evidence match.
"""

from __future__ import annotations

from typing import Any, Iterable


def text_contains_all(
    text: Any,
    expected_terms: Iterable[str],
) -> bool:
    """Return whether text contains every expected term."""

    normalized_text = str(text or "").casefold()

    return all(
        str(term).casefold() in normalized_text
        for term in expected_terms
    )


def period_term_match(
    doc: dict[str, Any],
    expected_period_terms: Iterable[str],
) -> bool:
    """Match any accepted representation of one expected period."""

    terms = [
        str(term).casefold()
        for term in expected_period_terms
    ]

    if not terms:
        return True

    period_text = " ".join(
        str(doc.get(field) or "")
        for field in (
            "period_label",
            "period_start",
            "period_end",
            "text",
        )
    ).casefold()

    return any(term in period_text for term in terms)


def expected_document_match(
    case: dict[str, Any],
    doc: dict[str, Any],
) -> bool:
    """Apply the complete expected-evidence contract to one document."""

    if case.get("case_type") == "negative_unsupported":
        return False

    expected_entity = case.get("expected_entity")
    expected_slot = case.get("expected_slot")
    expected_period_terms = case.get(
        "expected_period_terms",
        [],
    )
    expected_terms = case.get("expected_terms", [])

    entity_ok = (
        not expected_entity
        or str(doc.get("entity_id") or "")
        == str(expected_entity)
    )

    slot_ok = (
        not expected_slot
        or str(doc.get("slot") or "")
        == str(expected_slot)
    )

    period_ok = period_term_match(
        doc,
        expected_period_terms,
    )

    terms_ok = text_contains_all(
        doc.get("text", ""),
        expected_terms,
    )

    return bool(
        entity_ok
        and slot_ok
        and period_ok
        and terms_ok
    )


def expected_hit_at_k(
    case: dict[str, Any],
    docs: Iterable[dict[str, Any]],
) -> bool:
    """Return whether any retrieved document satisfies the full contract."""

    return any(
        expected_document_match(case, doc)
        for doc in docs
    )
