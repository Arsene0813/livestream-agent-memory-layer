from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from itertools import combinations
from pathlib import Path
from typing import Mapping


DEFAULT_REGISTRY_PATH = Path("retail_ops/contracts/datasets.v1.json")
ALLOWED_GRAINS = {
    "store_period",
    "store_sku_period",
    "store_search_term_period",
}
ALLOWED_RANKING_BASES = {
    "transaction_amount",
    "sales_volume",
    "not_recorded_in_current_fixture",
}
ALLOWED_SNAPSHOT_SEMANTICS = {
    "cumulative_period_snapshot",
    "incremental_period_extract",
}
ALLOWED_BATCH_STATUSES = {
    "received",
    "validated",
    "quarantined",
    "normalized",
    "failed",
}
NON_ADDITIVE_OVERLAP_POLICY = "keep_versions_do_not_sum"


class DatasetContractError(ValueError):
    pass


class BatchContractError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetContract:
    dataset_id: str
    source_path: str
    source_system: str
    source_name: str
    grain: str
    snapshot_semantics: str
    key_fields: tuple[str, ...]
    dimension_fields: tuple[str, ...]
    ranking_basis: str | None
    overlap_group: str
    overlap_policy: str


@dataclass(frozen=True)
class DatasetOverlap:
    left_dataset_id: str
    right_dataset_id: str
    record_count: int
    overlap_policy: str


@dataclass(frozen=True)
class BatchMetadata:
    batch_id: str
    dataset_id: str
    source_system: str
    source_name: str
    source_page: str
    extracted_at: str
    received_at: str
    file_sha256: str
    mapping_version: str
    snapshot_semantics: str
    coverage_start: str
    coverage_end: str
    status: str


def _text(
    payload: Mapping[str, object],
    field: str,
    error_type: type[ValueError],
) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise error_type(f"{field} must be a non-empty string")
    return value.strip()


def _strings(
    payload: Mapping[str, object],
    field: str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or (not value and not allow_empty):
        raise DatasetContractError(f"{field} must be a list of strings")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise DatasetContractError(f"{field} must contain non-empty strings")
    result = tuple(item.strip() for item in value)
    if len(result) != len(set(result)):
        raise DatasetContractError(f"{field} must not contain duplicates")
    return result


def _parse_contract(payload: Mapping[str, object]) -> DatasetContract:
    required_text = (
        "dataset_id",
        "source_path",
        "source_system",
        "source_name",
        "grain",
        "snapshot_semantics",
        "overlap_group",
        "overlap_policy",
    )
    values = {
        field: _text(payload, field, DatasetContractError)
        for field in required_text
    }
    dataset_id = values["dataset_id"]
    key_fields = _strings(payload, "key_fields")
    dimension_fields = _strings(
        payload,
        "dimension_fields",
        allow_empty=True,
    )
    ranking_basis = payload.get("ranking_basis")

    if values["grain"] not in ALLOWED_GRAINS:
        raise DatasetContractError(f"{dataset_id}: unsupported grain")
    if values["snapshot_semantics"] not in ALLOWED_SNAPSHOT_SEMANTICS:
        raise DatasetContractError(
            f"{dataset_id}: unsupported snapshot_semantics"
        )
    if values["overlap_policy"] != NON_ADDITIVE_OVERLAP_POLICY:
        raise DatasetContractError(f"{dataset_id}: additive overlap is forbidden")

    source_path = Path(values["source_path"])
    if source_path.is_absolute() or ".." in source_path.parts:
        raise DatasetContractError(
            f"{dataset_id}: source_path must be repository-relative"
        )

    if ranking_basis is not None and ranking_basis not in ALLOWED_RANKING_BASES:
        raise DatasetContractError(f"{dataset_id}: unsupported ranking_basis")
    if values["grain"] == "store_sku_period" and ranking_basis is None:
        raise DatasetContractError(
            f"{dataset_id}: store_sku_period requires ranking_basis"
        )
    if values["grain"] != "store_sku_period" and ranking_basis is not None:
        raise DatasetContractError(
            f"{dataset_id}: ranking_basis is only valid for SKU data"
        )

    return DatasetContract(
        key_fields=key_fields,
        dimension_fields=dimension_fields,
        ranking_basis=ranking_basis,
        **values,
    )


def load_dataset_contracts(
    root: Path,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
) -> dict[str, DatasetContract]:
    try:
        payload = json.loads((root / registry_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot load dataset registry: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("contract_version") != "1":
        raise DatasetContractError("dataset registry contract_version must be '1'")

    datasets = payload.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise DatasetContractError("dataset registry requires datasets")

    contracts: dict[str, DatasetContract] = {}
    source_paths: set[str] = set()
    for index, item in enumerate(datasets):
        if not isinstance(item, dict):
            raise DatasetContractError(f"datasets[{index}] must be an object")
        contract = _parse_contract(item)
        if contract.dataset_id in contracts:
            raise DatasetContractError(f"duplicate dataset_id: {contract.dataset_id}")
        if contract.source_path in source_paths:
            raise DatasetContractError(f"duplicate source_path: {contract.source_path}")
        contracts[contract.dataset_id] = contract
        source_paths.add(contract.source_path)
    return contracts


def _read_keys(root: Path, contract: DatasetContract) -> set[tuple[str, ...]]:
    path = (root / contract.source_path).resolve()
    if root.resolve() not in path.parents:
        raise DatasetContractError(f"{contract.dataset_id}: invalid source_path")

    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except FileNotFoundError as exc:
        raise DatasetContractError(
            f"{contract.dataset_id}: source CSV not found"
        ) from exc

    with handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        required = set(contract.key_fields) | set(contract.dimension_fields)
        if missing := sorted(required - columns):
            raise DatasetContractError(
                f"{contract.dataset_id}: missing contract columns {missing}"
            )

        keys: set[tuple[str, ...]] = set()
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise DatasetContractError(
                    f"{contract.dataset_id}: row {line_number} has extra columns"
                )
            key = tuple((row.get(field) or "").strip() for field in contract.key_fields)
            if any(not value for value in key):
                raise DatasetContractError(
                    f"{contract.dataset_id}: row {line_number} has an empty key"
                )
            if key in keys:
                raise DatasetContractError(
                    f"{contract.dataset_id}: duplicate logical key {key!r}"
                )
            keys.add(key)
    return keys


def validate_dataset_contracts(root: Path) -> tuple[DatasetOverlap, ...]:
    contracts = load_dataset_contracts(root)
    keys = {
        dataset_id: _read_keys(root, contract)
        for dataset_id, contract in contracts.items()
    }
    overlaps: list[DatasetOverlap] = []

    for left, right in combinations(contracts.values(), 2):
        same_logical_space = (
            left.grain == right.grain
            and left.key_fields == right.key_fields
            and left.overlap_group == right.overlap_group
            and left.ranking_basis == right.ranking_basis
        )
        if not same_logical_space:
            continue
        shared_keys = keys[left.dataset_id] & keys[right.dataset_id]
        if shared_keys:
            overlaps.append(
                DatasetOverlap(
                    left.dataset_id,
                    right.dataset_id,
                    len(shared_keys),
                    NON_ADDITIVE_OVERLAP_POLICY,
                )
            )
    return tuple(overlaps)


def _aware_datetime(field: str, value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise BatchContractError(f"{field} must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BatchContractError(f"{field} must include a timezone")
    return parsed


def _date(field: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise BatchContractError(f"{field} must be an ISO date") from exc


def build_batch_metadata(
    payload: Mapping[str, object],
    contracts: Mapping[str, DatasetContract],
) -> BatchMetadata:
    fields = tuple(BatchMetadata.__dataclass_fields__)
    values = {
        field: _text(payload, field, BatchContractError)
        for field in fields
    }
    contract = contracts.get(values["dataset_id"])
    if contract is None:
        raise BatchContractError("dataset_id is not registered")
    if values["source_system"] != contract.source_system:
        raise BatchContractError("source_system does not match dataset")
    if values["source_name"] != contract.source_name:
        raise BatchContractError("source_name does not match dataset")
    if values["snapshot_semantics"] != contract.snapshot_semantics:
        raise BatchContractError("snapshot_semantics does not match dataset")
    if not re.fullmatch(r"[0-9a-f]{64}", values["file_sha256"]):
        raise BatchContractError("file_sha256 must be lowercase SHA-256")

    extracted_at = _aware_datetime("extracted_at", values["extracted_at"])
    received_at = _aware_datetime("received_at", values["received_at"])
    if received_at < extracted_at:
        raise BatchContractError("received_at is earlier than extracted_at")

    coverage_start = _date("coverage_start", values["coverage_start"])
    coverage_end = _date("coverage_end", values["coverage_end"])
    if coverage_end < coverage_start:
        raise BatchContractError("coverage_end is earlier than coverage_start")
    if values["status"] not in ALLOWED_BATCH_STATUSES:
        raise BatchContractError("unsupported batch status")

    return BatchMetadata(**values)
