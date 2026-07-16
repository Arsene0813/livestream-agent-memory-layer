#!/usr/bin/env python3
"""Canonical retrieval-corpus construction for the retail experiments."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Keep these as repository-relative paths because doc IDs, source labels,
# experiment summaries, and the existing threshold output use this form.
FACT_JSON_PATHS = [
    Path("retail_ops/outputs/generated_retail_memory_facts.json"),
    Path(
        "retail_ops/outputs/"
        "generated_demo2_retail_memory_facts.json"
    ),
]

FIELD_CONTRACT_PATHS = [
    Path("retail_ops/data/DATA_DICTIONARY.md"),
    Path("retail_ops/data/demo2_source_notes.md"),
]

CORPUS_BUILDER_ID = (
    "eval/retail_retrieval_corpus.py::"
    "load_retail_retrieval_documents"
)


def corpus_sha256(
    docs: list[dict[str, Any]],
) -> str:
    """Return a deterministic SHA-256 digest for the ordered corpus."""

    payload = json.dumps(
        docs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def current_git_commit() -> str:
    """Return the repository HEAD used to generate an output."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"

    commit = completed.stdout.strip().lower()

    if re.fullmatch(r"[0-9a-f]{40}", commit):
        return commit

    return "unavailable"


def corpus_provenance(
    docs: list[dict[str, Any]],
    embedding_model: str,
) -> dict[str, Any]:
    """Build provenance metadata shared by retrieval outputs."""

    return {
        "corpus_document_count": len(docs),
        "corpus_sha256": corpus_sha256(docs),
        "embedding_model": embedding_model,
        "corpus_builder": CORPUS_BUILDER_ID,
        "generated_from_commit": current_git_commit(),
    }


def repository_path(path: Path) -> Path:
    """Resolve a repository-relative corpus path for file access."""

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def load_json(path: Path) -> Any:
    """Load a required repository JSON file."""

    actual_path = repository_path(path)

    if not actual_path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return json.loads(
        actual_path.read_text(encoding="utf-8")
    )


def compact_json(value: Any) -> str:
    """Serialize nested fact values deterministically."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
    )


def first_existing(
    fact: dict[str, Any],
    keys: list[str],
) -> str:
    """Return the first non-empty fact value from the supplied keys."""

    for key in keys:
        value = fact.get(key)

        if value not in (None, "", [], {}):
            return str(value)

    return ""


def fact_to_text(
    fact: dict[str, Any],
    source_file: str,
) -> str:
    """Serialize one generated memory fact for retrieval."""

    fields = [
        ("source_file", source_file),
        ("fact_id", fact.get("fact_id", "")),
        ("kind", fact.get("kind", "")),
        ("type", fact.get("type", "")),
        ("entity_id", fact.get("entity_id", "")),
        ("slot", fact.get("slot", "")),
        (
            "period_label",
            first_existing(
                fact,
                ["period_label", "period"],
            ),
        ),
        ("period_start", fact.get("period_start", "")),
        ("period_end", fact.get("period_end", "")),
        ("value", fact.get("value", "")),
        (
            "observed_values",
            compact_json(
                fact.get("observed_values", {})
            ),
        ),
        ("calculation", fact.get("calculation", "")),
        (
            "source_fields",
            compact_json(
                fact.get("source_fields", [])
            ),
        ),
        ("source_path", fact.get("source_path", "")),
        (
            "supporting_source_paths",
            compact_json(
                fact.get(
                    "supporting_source_paths",
                    [],
                )
            ),
        ),
        ("lineage_path", fact.get("lineage_path", "")),
        ("confidence", fact.get("confidence", "")),
        (
            "limitations",
            compact_json(
                fact.get("limitations", [])
            ),
        ),
        ("is_active", fact.get("is_active", "")),
    ]

    known = {
        key
        for key, _ in fields
    }

    extras = []

    for key in sorted(fact):
        if key in known:
            continue

        value = fact[key]

        if (
            isinstance(
                value,
                (str, int, float, bool),
            )
            or value is None
        ):
            extras.append((key, value))

    lines = [
        f"{key}: {value}"
        for key, value in fields
    ]

    lines.extend(
        f"{key}: {value}"
        for key, value in extras
    )

    return "\n".join(lines)


def fact_to_doc(
    fact: dict[str, Any],
    source_file: str,
    index: int,
) -> dict[str, Any]:
    """Convert one memory fact into the canonical document schema."""

    return {
        "doc_id": str(
            fact.get("fact_id")
            or f"{Path(source_file).name}:{index}"
        ),
        "doc_type": "generated_memory_fact",
        "source_file": source_file,
        "entity_id": str(
            fact.get("entity_id", "")
        ),
        "slot": str(
            fact.get("slot", "")
        ),
        "period_label": first_existing(
            fact,
            ["period_label", "period"],
        ),
        "period_start": str(
            fact.get("period_start", "")
        ),
        "period_end": str(
            fact.get("period_end", "")
        ),
        "source_path": str(
            fact.get("source_path", "")
        ),
        "text": fact_to_text(
            fact,
            source_file,
        ),
    }


def split_markdown_chunks(
    path: Path,
) -> list[dict[str, Any]]:
    """Create field-row and paragraph documents from one Markdown file."""

    actual_path = repository_path(path)

    if not actual_path.exists():
        return []

    text = actual_path.read_text(
        encoding="utf-8"
    )

    source_label = path.as_posix()
    chunks: list[dict[str, Any]] = []

    # Preserve dictionary table rows as individual field-contract
    # documents.
    for line_no, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        stripped = line.strip()

        if (
            stripped.startswith("| `")
            and "`" in stripped
        ):
            field_match = re.search(
                r"`([^`]+)`",
                stripped,
            )

            slot = (
                field_match.group(1)
                if field_match
                else ""
            )

            chunks.append(
                {
                    "doc_id": (
                        f"{source_label}:L{line_no}"
                    ),
                    "doc_type": (
                        "field_contract_row"
                    ),
                    "source_file": source_label,
                    "entity_id": "",
                    "slot": slot,
                    "period_label": "",
                    "period_start": "",
                    "period_end": "",
                    "source_path": source_label,
                    "text": (
                        f"source_file: {source_label}\n"
                        f"line: {line_no}\n"
                        f"slot: {slot}\n"
                        f"content: {stripped}"
                    ),
                }
            )

    # Preserve narrative boundaries as paragraph-level documents.
    parts = re.split(
        r"\n\s*\n",
        text,
    )

    for index, part in enumerate(parts):
        cleaned = part.strip()

        if len(cleaned) < 80:
            continue

        chunks.append(
            {
                "doc_id": (
                    f"{source_label}:chunk:{index}"
                ),
                "doc_type": (
                    "field_contract_note"
                ),
                "source_file": source_label,
                "entity_id": "",
                "slot": "",
                "period_label": "",
                "period_start": "",
                "period_end": "",
                "source_path": source_label,
                "text": (
                    f"source_file: {source_label}\n"
                    f"chunk: {index}\n"
                    f"content:\n{cleaned}"
                ),
            }
        )

    return chunks


def load_retail_retrieval_documents(
) -> list[dict[str, Any]]:
    """Build the canonical corpus shared by both retrieval experiments."""

    docs: list[dict[str, Any]] = []

    for path in FACT_JSON_PATHS:
        facts = load_json(path)

        if not isinstance(facts, list):
            raise ValueError(
                f"{path} should contain a list "
                "of generated memory facts"
            )

        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                raise ValueError(
                    f"{path} contains a non-object "
                    f"fact at index {index}"
                )

            docs.append(
                fact_to_doc(
                    fact,
                    path.as_posix(),
                    index,
                )
            )

    for path in FIELD_CONTRACT_PATHS:
        docs.extend(
            split_markdown_chunks(path)
        )

    if not docs:
        raise ValueError(
            "No retrieval documents loaded"
        )

    return docs
