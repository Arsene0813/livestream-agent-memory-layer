"""Strict validation for retrieval experiment case files."""

from __future__ import annotations

from typing import Any


ALLOWED_CASE_TYPES = frozenset(
    {
        "positive_supported",
        "negative_unsupported",
        "hard_negative_boundary",
        "entity_period_mismatch",
        "ambiguous_comparison",
    }
)

CASE_LIST_KEYS = (
    "cases",
    "retrieval_threshold_cases",
    "items",
    "data",
)

REQUIRED_CASE_FIELDS = frozenset(
    {
        "case_id",
        "case_type",
        "query",
        "expected_entity",
        "expected_slot",
        "expected_period_terms",
        "expected_terms",
    }
)


def _extract_case_list(
    raw: Any,
    *,
    source: str,
) -> list[Any]:
    if isinstance(raw, list):
        return raw

    if isinstance(raw, dict):
        for key in CASE_LIST_KEYS:
            value = raw.get(key)

            if isinstance(value, list):
                return value

        expected = ", ".join(
            repr(key)
            for key in CASE_LIST_KEYS
        )

        raise ValueError(
            f"{source}: expected a case list or an object "
            f"containing one of: {expected}"
        )

    raise ValueError(
        f"{source}: expected a case list or object, "
        f"found {type(raw).__name__}"
    )


def _validate_string_list(
    value: Any,
    *,
    source: str,
    index: int,
    field: str,
) -> None:
    if not isinstance(value, list):
        raise ValueError(
            f"{source}: case[{index}].{field} must be a list, "
            f"found {type(value).__name__}"
        )

    for item_index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(
                f"{source}: case[{index}].{field}[{item_index}] "
                f"must be a string, found {type(item).__name__}"
            )

        if not item.strip():
            raise ValueError(
                f"{source}: case[{index}].{field}[{item_index}] "
                "must not be empty"
            )


def validate_retrieval_cases(
    raw: Any,
    *,
    source: str = "<retrieval cases>",
) -> list[dict[str, Any]]:
    """Return a strictly validated canonical retrieval case list.

    Validation is completed before any embedding call. The input objects are
    not modified.
    """

    raw_cases = _extract_case_list(
        raw,
        source=source,
    )

    validated: list[dict[str, Any]] = []
    first_id_index: dict[str, int] = {}
    first_query_index: dict[str, int] = {}

    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(
                f"{source}: case[{index}] must be an object, "
                f"found {type(raw_case).__name__}"
            )

        missing = REQUIRED_CASE_FIELDS - set(raw_case)

        if missing:
            raise ValueError(
                f"{source}: case[{index}] is missing fields: "
                + ", ".join(sorted(missing))
            )

        case_id = raw_case["case_id"]
        query = raw_case["query"]
        case_type = raw_case["case_type"]
        expected_entity = raw_case["expected_entity"]
        expected_slot = raw_case["expected_slot"]

        if not isinstance(case_id, str):
            raise ValueError(
                f"{source}: case[{index}].case_id must be a string, "
                f"found {type(case_id).__name__}"
            )

        case_id = case_id.strip()

        if not case_id:
            raise ValueError(
                f"{source}: case[{index}].case_id must not be empty"
            )

        if not isinstance(query, str):
            raise ValueError(
                f"{source}: case[{index}].query must be a string, "
                f"found {type(query).__name__}"
            )

        query = query.strip()

        if not query:
            raise ValueError(
                f"{source}: case[{index}].query must not be empty"
            )

        if not isinstance(case_type, str):
            raise ValueError(
                f"{source}: case[{index}].case_type must be a string, "
                f"found {type(case_type).__name__}"
            )

        if case_type not in ALLOWED_CASE_TYPES:
            allowed = ", ".join(
                sorted(ALLOWED_CASE_TYPES)
            )

            raise ValueError(
                f"{source}: case[{index}].case_type "
                f"has unsupported value {case_type!r}; "
                f"allowed values: {allowed}"
            )

        if (
            expected_entity is not None
            and not isinstance(expected_entity, str)
        ):
            raise ValueError(
                f"{source}: case[{index}].expected_entity "
                "must be a string or null, "
                f"found {type(expected_entity).__name__}"
            )

        if (
            expected_slot is not None
            and not isinstance(expected_slot, str)
        ):
            raise ValueError(
                f"{source}: case[{index}].expected_slot "
                "must be a string or null, "
                f"found {type(expected_slot).__name__}"
            )

        _validate_string_list(
            raw_case["expected_period_terms"],
            source=source,
            index=index,
            field="expected_period_terms",
        )

        _validate_string_list(
            raw_case["expected_terms"],
            source=source,
            index=index,
            field="expected_terms",
        )

        if case_id in first_id_index:
            raise ValueError(
                f"{source}: case[{index}].case_id duplicates "
                f"case[{first_id_index[case_id]}].case_id: "
                f"{case_id!r}"
            )

        if query in first_query_index:
            raise ValueError(
                f"{source}: case[{index}].query duplicates "
                f"case[{first_query_index[query]}].query: "
                f"{query!r}"
            )

        first_id_index[case_id] = index
        first_query_index[query] = index
        validated.append(raw_case)

    if not validated:
        raise ValueError(
            f"{source}: case list must not be empty"
        )

    return validated
