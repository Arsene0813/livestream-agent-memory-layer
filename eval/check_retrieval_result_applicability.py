#!/usr/bin/env python3
"""
Check whether committed retail retrieval results remain applicable.

The repository HEAD may continue to move. A result becomes stale only
when code or inputs inside that experiment's declared scope change.

Newly generated summaries store a deterministic experiment-scope
SHA-256. The run commit is retained only as navigation metadata.

Legacy summaries without a scope hash fall back to a scoped Git diff
against their recorded run commit.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

THRESHOLD_SUMMARY_PATH = (
    "retail_ops/outputs/retrieval_threshold_summary.md"
)

SHARED_RETRIEVAL_SCOPE_PATHS = (
    "eval/retrieval_threshold_cases.json",
    "eval/retail_retrieval_corpus.py",
    "eval/retrieval_case_validation.py",
    "eval/retrieval_contract_match.py",
    (
        "retail_ops/outputs/"
        "generated_retail_memory_facts.json"
    ),
    (
        "retail_ops/outputs/"
        "generated_demo2_retail_memory_facts.json"
    ),
    "retail_ops/data/DATA_DICTIONARY.md",
    "retail_ops/data/demo2_source_notes.md",
)

THRESHOLD_EXPERIMENT_SCOPE_PATHS = (
    *SHARED_RETRIEVAL_SCOPE_PATHS,
    "eval/analyze_retail_embedding_score_distribution.py",
)

# Keep only score-affecting code and corpus inputs in these
# hashes. This checker is validation machinery, not a retrieval
# input. The robustness experiment's reference threshold is
# verified separately against THRESHOLD_SUMMARY_PATH below, so
# unrelated edits to that generated Markdown do not stale a run.
ROBUSTNESS_EXPERIMENT_SCOPE_PATHS = (
    *SHARED_RETRIEVAL_SCOPE_PATHS,
    "eval/analyze_retail_query_robustness.py",
)


@dataclass(frozen=True)
class Experiment:
    name: str
    summary_path: str
    scope_paths: tuple[str, ...]
    check_reference_threshold: bool = False


EXPERIMENTS = (
    Experiment(
        name="retrieval_threshold_inspection",
        summary_path=THRESHOLD_SUMMARY_PATH,
        scope_paths=THRESHOLD_EXPERIMENT_SCOPE_PATHS,
    ),
    Experiment(
        name="retrieval_query_wording_variation",
        summary_path=(
            "retail_ops/outputs/"
            "retrieval_query_robustness_summary.md"
        ),
        scope_paths=ROBUSTNESS_EXPERIMENT_SCOPE_PATHS,
        check_reference_threshold=True,
    ),
)


def git(
    *args: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        capture_output=True,
        text=True,
        timeout=30,
    )


def current_git_commit() -> str:
    completed = git("rev-parse", "HEAD")
    commit = completed.stdout.strip().lower()

    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(
            f"unexpected git commit value: {commit!r}"
        )

    return commit


def ensure_scope_clean(
    paths: Iterable[str],
    experiment_name: str,
) -> None:
    """
    Require only experiment-relevant files to be committed.

    Unrelated repository files may remain modified.
    """
    scope = tuple(dict.fromkeys(paths))

    completed = git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        *scope,
    )

    dirty = [
        line.rstrip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]

    if not dirty:
        return

    details = "\n".join(
        f"  {line}"
        for line in dirty
    )

    raise SystemExit(
        f"[FAIL] {experiment_name} was not run.\n"
        "The experiment-relevant scope contains "
        "uncommitted changes:\n"
        f"{details}\n"
        "Commit or stash these relevant files first.\n"
        "Unrelated repository changes do not need "
        "to be committed."
    )


def scope_sha256(
    paths: Iterable[str],
) -> str:
    """
    Hash repository-relative path names and current file bytes.

    This is the applicability key for a generated result.
    It is independent of unrelated commits and remains usable
    if Git history is later squashed.
    """
    digest = hashlib.sha256()

    for rel in sorted(set(paths)):
        path = ROOT / rel

        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")

        if path.is_file():
            digest.update(b"file\0")
            digest.update(path.read_bytes())
        elif path.exists():
            raise ValueError(
                f"scope path is not a regular file: {rel}"
            )
        else:
            digest.update(b"missing\0")

        digest.update(b"\0")

    return digest.hexdigest()


def parse_run_commit(
    summary_path: Path,
) -> str | None:
    text = summary_path.read_text(encoding="utf-8")

    match = re.search(
        (
            r"^- (?:Run|Execution) commit: "
            r"`([0-9a-fA-F]{40})`\s*$"
        ),
        text,
        flags=re.MULTILINE,
    )

    if not match:
        return None

    return match.group(1).lower()


def parse_scope_hash(
    summary_path: Path,
) -> str | None:
    text = summary_path.read_text(encoding="utf-8")

    match = re.search(
        (
            r"^- Experiment scope SHA-256: "
            r"`([0-9a-fA-F]{64})`\s*$"
        ),
        text,
        flags=re.MULTILINE,
    )

    if not match:
        return None

    return match.group(1).lower()


def parse_reference_threshold(
    summary_path: Path,
) -> float:
    text = summary_path.read_text(encoding="utf-8")

    patterns = (
        (
            r"^- Reference threshold: "
            r"`([0-9]+(?:\.[0-9]+)?)`\s*$"
        ),
        (
            r"exploratory reference threshold"
            r"[^`]*`([0-9]+(?:\.[0-9]+)?)`"
        ),
    )

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if match:
            return float(match.group(1))

    raise ValueError(
        "could not parse a reference threshold from "
        f"{summary_path}"
    )


def read_markdown_value(
    text: str,
    label: str,
) -> str | None:
    match = re.search(
        (
            rf"^- {re.escape(label)}: "
            r"`([^`]+)`\s*$"
        ),
        text,
        flags=re.MULTILINE,
    )

    if not match:
        return None

    return match.group(1).strip()


def changed_scope_paths(
    run_commit: str,
    paths: Iterable[str],
) -> list[str]:
    """
    Return relevant paths whose current content differs
    from the recorded run commit.

    This is used for legacy summaries and for readable
    diagnostics when a run commit remains available.
    """
    scope = tuple(dict.fromkeys(paths))

    commit_check = git(
        "cat-file",
        "-e",
        f"{run_commit}^{{commit}}",
        check=False,
    )

    if commit_check.returncode != 0:
        raise ValueError(
            "run commit is unavailable in local Git "
            f"history: {run_commit}"
        )

    completed = git(
        "diff",
        "--name-only",
        "--no-renames",
        run_commit,
        "--",
        *scope,
        check=False,
    )

    if completed.returncode != 0:
        detail = (
            completed.stderr.strip()
            or completed.stdout.strip()
        )
        raise RuntimeError(
            f"git diff failed: {detail}"
        )

    changed = {
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    }

    for rel in scope:
        current_exists = (ROOT / rel).exists()

        existed_at_run = (
            git(
                "cat-file",
                "-e",
                f"{run_commit}:{rel}",
                check=False,
            ).returncode
            == 0
        )

        if current_exists != existed_at_run:
            changed.add(rel)

    return sorted(changed)


def main() -> int:
    head = current_git_commit()

    print(f"Repository HEAD: {head}")
    print(
        "Rule: HEAD may move. A result becomes stale "
        "only when its declared experiment scope changes."
    )
    print(
        "Primary applicability key: deterministic "
        "experiment-scope SHA-256."
    )
    print(
        "Legacy fallback: scoped Git diff from the "
        "recorded run commit."
    )
    print(
        "Boundary: the checker records the embedding "
        "model name but does not fingerprint the local "
        "Ollama model binary."
    )

    has_stale = False
    has_unknown = False

    for experiment in EXPERIMENTS:
        summary_path = ROOT / experiment.summary_path

        print(f"\n{experiment.name}")

        if not summary_path.exists():
            has_unknown = True
            print(
                "  [UNKNOWN] Missing summary: "
                f"{experiment.summary_path}"
            )
            continue

        text = summary_path.read_text(
            encoding="utf-8"
        )

        run_commit = parse_run_commit(summary_path)
        recorded_scope_hash = parse_scope_hash(
            summary_path
        )

        changed: list[str] = []
        scope_issue: str | None = None

        if recorded_scope_hash:
            try:
                current_scope_hash = scope_sha256(
                    experiment.scope_paths
                )
            except (OSError, ValueError) as exc:
                has_unknown = True
                print(f"  [UNKNOWN] {exc}")
                continue

            print(
                "  applicability method: scope SHA-256"
            )
            print(
                "  recorded scope hash: "
                f"{recorded_scope_hash}"
            )
            print(
                "  current scope hash:  "
                f"{current_scope_hash}"
            )

            if current_scope_hash != recorded_scope_hash:
                scope_issue = (
                    "experiment-relevant scope hash changed"
                )

                if run_commit:
                    try:
                        changed = changed_scope_paths(
                            run_commit,
                            experiment.scope_paths,
                        )
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                    ):
                        changed = []
        else:
            print(
                "  applicability method: legacy scoped "
                "Git diff"
            )

            if not run_commit:
                has_unknown = True
                print(
                    "  [UNKNOWN] Summary has neither an "
                    "experiment scope hash nor a usable "
                    "run commit."
                )
                continue

            try:
                changed = changed_scope_paths(
                    run_commit,
                    experiment.scope_paths,
                )
            except (
                OSError,
                ValueError,
                RuntimeError,
            ) as exc:
                has_unknown = True
                print(f"  [UNKNOWN] {exc}")
                continue

            if changed:
                scope_issue = (
                    "experiment-relevant files changed "
                    "since the legacy run commit"
                )

        print(
            "  recorded run commit: "
            f"{run_commit or 'not recorded'}"
        )
        print(
            f"  current HEAD:        {head}"
        )

        threshold_issue: str | None = None

        if experiment.check_reference_threshold:
            mode = read_markdown_value(
                text,
                "Reference threshold mode",
            )
            source = read_markdown_value(
                text,
                "Reference threshold source",
            )

            try:
                recorded_threshold = (
                    parse_reference_threshold(
                        summary_path
                    )
                )
            except ValueError as exc:
                threshold_issue = str(exc)
            else:
                should_follow_source = (
                    mode != "cli_override"
                )

                if should_follow_source and source:
                    source_path = Path(source)

                    if not source_path.is_absolute():
                        source_path = ROOT / source_path

                    try:
                        current_threshold = (
                            parse_reference_threshold(
                                source_path
                            )
                        )
                    except (
                        OSError,
                        ValueError,
                    ) as exc:
                        threshold_issue = (
                            "could not verify current "
                            "reference-threshold source: "
                            f"{exc}"
                        )
                    else:
                        if abs(
                            current_threshold
                            - recorded_threshold
                        ) > 1e-12:
                            threshold_issue = (
                                "reference threshold changed: "
                                f"recorded="
                                f"{recorded_threshold}, "
                                f"current source="
                                f"{current_threshold}"
                            )

        if scope_issue or threshold_issue:
            has_stale = True

            print(
                "  [STALE] The recorded result no "
                "longer matches its current relevant "
                "scope."
            )

            if scope_issue:
                print(f"    - {scope_issue}")

            for path in changed:
                print(
                    f"    - changed path: {path}"
                )

            if threshold_issue:
                print(
                    f"    - {threshold_issue}"
                )
        else:
            print(
                "  [PASS] The recorded result "
                "remains applicable."
            )

    if has_stale:
        return 1

    if has_unknown:
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
