#!/usr/bin/env python3
"""
Analyze retrieval behavior under small query wording variations.

This script is intentionally self-contained:
- reads existing retrieval threshold cases;
- builds a small retail evidence corpus from generated memory facts and key docs;
- calls local Ollama bge-m3 embeddings;
- computes cosine similarity directly;
- writes robustness rows, threshold sweep rows, and a Markdown summary.

It does not modify the production API or the existing retrieval threshold inspection script.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from retail_retrieval_corpus import (
        corpus_provenance,
        load_retail_retrieval_documents,
    )
except ModuleNotFoundError:
    from eval.retail_retrieval_corpus import (
        corpus_provenance,
        load_retail_retrieval_documents,
    )

try:
    from retrieval_case_validation import validate_retrieval_cases
except ModuleNotFoundError:
    from eval.retrieval_case_validation import validate_retrieval_cases

try:
    from retrieval_contract_match import expected_hit_at_k
except ModuleNotFoundError:
    from eval.retrieval_contract_match import expected_hit_at_k

try:
    from check_retrieval_result_applicability import (
        ROBUSTNESS_EXPERIMENT_SCOPE_PATHS,
        ensure_scope_clean,
        parse_reference_threshold,
        scope_sha256,
    )
except ModuleNotFoundError:
    from eval.check_retrieval_result_applicability import (
        ROBUSTNESS_EXPERIMENT_SCOPE_PATHS,
        ensure_scope_clean,
        parse_reference_threshold,
        scope_sha256,
    )


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASES_PATH = ROOT / "eval" / "retrieval_threshold_cases.json"
DEFAULT_OUTPUT_DIR = ROOT / "retail_ops" / "outputs"


REFERENCE_THRESHOLD_SOURCE_DEFAULT = (
    "retail_ops/outputs/"
    "retrieval_threshold_summary.md"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path

    return ROOT / path


def extract_cases(
    raw: Any,
    *,
    source: str = "<retrieval cases>",
) -> list[dict[str, Any]]:
    """Validate and copy canonical retrieval cases."""

    cases = validate_retrieval_cases(
        raw,
        source=source,
    )

    return [dict(case) for case in cases]


def build_corpus() -> list[dict[str, Any]]:
    """Load the shared canonical retail retrieval corpus."""

    return load_retail_retrieval_documents()


def ollama_embed(text: str, model: str, ollama_url: str, retries: int = 3) -> list[float]:
    # Try legacy endpoint first.
    payloads = [
        ("/api/embeddings", {"model": model, "prompt": text}),
        ("/api/embed", {"model": model, "input": text}),
    ]

    last_error = None

    for endpoint, payload in payloads:
        url = ollama_url.rstrip("/") + endpoint
        body = json.dumps(payload).encode("utf-8")

        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                if "embedding" in data and isinstance(data["embedding"], list):
                    return [float(x) for x in data["embedding"]]

                if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                    return [float(x) for x in data["embeddings"][0]]

                raise RuntimeError(f"Ollama response did not contain embedding: {data.keys()}")

            except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(1.0 * attempt)

    raise RuntimeError(
        f"Failed to embed text with model={model}. "
        f"Check that Ollama is running and the model is pulled. Last error: {last_error}"
    )


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def stable_rng_for_text(text: str) -> random.Random:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    seed = int(digest[:8], 16)
    return random.Random(seed)


def shorten_query(query: str) -> str:
    fillers = {
        "please", "could", "would", "you", "tell", "me", "about", "the", "a", "an",
        "current", "given", "based", "on", "using", "show", "explain", "analyze",
    }
    tokens = query.split()
    kept = [t for t in tokens if re.sub(r"[^A-Za-z0-9_%-]", "", t).lower() not in fillers]
    if len(kept) >= 4:
        return " ".join(kept[:18])
    if len(tokens) > 8:
        return " ".join(tokens[:8])
    return query


def paraphrase_query(query: str) -> str:
    replacements = [
        (r"\bcompare\b", "evaluate"),
        (r"\bcomparison\b", "evaluation"),
        (r"\bshow\b", "summarize"),
        (r"\bexplain\b", "describe"),
        (r"\bwhich\b", "what"),
        (r"\bstore\b", "merchant store"),
        (r"\bstores\b", "merchant stores"),
        (r"\bshould\b", "can"),
        (r"\bpromotion\b", "activity"),
        (r"\bROI\b", "activity-cost interpretation"),
    ]

    out = query
    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.I)

    if out == query:
        out = "What evidence is available for: " + query

    return out


def typo_or_punctuation_noise(query: str) -> str:
    rng = stable_rng_for_text(query)
    tokens = query.split()
    candidates = [i for i, tok in enumerate(tokens) if len(re.sub(r"[^A-Za-z]", "", tok)) >= 5]

    if candidates:
        idx = rng.choice(candidates)
        tok = tokens[idx]
        letters = list(tok)
        if len(letters) >= 4:
            j = rng.randrange(1, len(letters) - 1)
            letters[j], letters[j + 1] = letters[j + 1], letters[j]
            tokens[idx] = "".join(letters)

    noisy = " ".join(tokens)
    noisy = noisy.replace(",", "").replace(";", "")
    if not noisy.endswith("??"):
        noisy += " ??"
    return noisy


def keyword_order_changed(query: str) -> str:
    tokens = query.split()
    if len(tokens) < 6:
        return query + " evidence"
    midpoint = len(tokens) // 2
    return " ".join(tokens[midpoint:] + tokens[:midpoint])


def make_variants(query: str) -> list[tuple[str, str]]:
    variants = [
        ("original", query),
        ("shortened", shorten_query(query)),
        ("paraphrased", paraphrase_query(query)),
        ("typo_punctuation_noise", typo_or_punctuation_noise(query)),
        ("keyword_order_changed", keyword_order_changed(query)),
    ]

    # Ensure variant text is unique while preserving order.
    seen = set()
    unique = []
    for variant_type, text in variants:
        text = re.sub(r"\s+", " ", text).strip()
        if text not in seen:
            seen.add(text)
            unique.append((variant_type, text))

    return unique


def pct(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return round(n / d * 100.0, 2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default="bge-m3")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument(
        "--reference-threshold",
        type=float,
        default=None,
        help=(
            "Optional explicit override. By default, read the "
            "exploratory reference value from "
            "--reference-threshold-source."
        ),
    )
    parser.add_argument(
        "--reference-threshold-source",
        default=REFERENCE_THRESHOLD_SOURCE_DEFAULT,
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases_path = resolve_repo_path(args.cases)
    output_dir = resolve_repo_path(args.output_dir)

    reference_threshold_source_path = resolve_repo_path(
        args.reference_threshold_source
    )

    if args.reference_threshold is None:
        reference_threshold = parse_reference_threshold(
            reference_threshold_source_path
        )
        reference_threshold_mode = "summary_source"
    else:
        reference_threshold = args.reference_threshold
        reference_threshold_mode = "cli_override"

    try:
        reference_threshold_source_label = (
            reference_threshold_source_path
            .relative_to(ROOT)
            .as_posix()
        )
    except ValueError:
        reference_threshold_source_label = str(
            reference_threshold_source_path
        )

    ensure_scope_clean(
        ROBUSTNESS_EXPERIMENT_SCOPE_PATHS,
        "retrieval query wording-variation stress test",
    )

    if not cases_path.exists():
        raise SystemExit(f"[FAIL] Cases file not found: {cases_path}")

    raw_cases = load_json(cases_path)
    cases = extract_cases(
        raw_cases,
        source=str(cases_path),
    )
    if not cases:
        raise SystemExit("[FAIL] No usable cases found.")

    docs = build_corpus()
    if not docs:
        raise SystemExit("[FAIL] No retrieval units found in the corpus.")

    provenance = corpus_provenance(
        docs,
        args.model,
    )
    provenance.update(
        {
            "reference_threshold": reference_threshold,
            "reference_threshold_source": (
                reference_threshold_source_label
            ),
            "reference_threshold_mode": (
                reference_threshold_mode
            ),
            "experiment_scope_sha256": scope_sha256(
                ROBUSTNESS_EXPERIMENT_SCOPE_PATHS
            ),
        }
    )

    print(f"[INFO] Loaded cases: {len(cases)}")
    print(f"[INFO] Built retrieval units: {len(docs)}")
    print(f"[INFO] Embedding corpus with model={args.model}")

    doc_embeddings = []
    for i, doc in enumerate(docs, start=1):
        if i % 25 == 0 or i == 1 or i == len(docs):
            print(f"[INFO] Embedding unit {i}/{len(docs)}")
        emb = ollama_embed(doc["text"], args.model, args.ollama_url)
        doc_embeddings.append(emb)

    rows: list[dict[str, Any]] = []
    original_by_case: dict[str, dict[str, Any]] = {}

    total_variants = sum(len(make_variants(case["query"])) for case in cases)
    done = 0

    for case in cases:
        variants = make_variants(case["query"])

        for variant_index, (variant_type, variant_query) in enumerate(variants, start=1):
            done += 1
            print(f"[INFO] Query {done}/{total_variants}: {case['case_id']} / {variant_type}")

            q_emb = ollama_embed(variant_query, args.model, args.ollama_url)
            scored = []
            for doc, d_emb in zip(docs, doc_embeddings):
                scored.append((cosine(q_emb, d_emb), doc))
            scored.sort(key=lambda x: x[0], reverse=True)

            top = scored[: args.top_k]
            top1_score, top1_doc = top[0]
            top_docs = [doc for _, doc in top]

            hit_at_k = expected_hit_at_k(case, top_docs)
            top5_doc_ids = [doc["doc_id"] for _, doc in top]
            top5_slots = [doc["slot"] for _, doc in top]
            top5_entities = [doc["entity_id"] for _, doc in top]

            row = {
                "case_id": case["case_id"],
                "variant_id": f"{case['case_id']}::{variant_index:02d}_{variant_type}",
                "variant_type": variant_type,
                "case_type": case["case_type"],
                "corpus_document_count": provenance["corpus_document_count"],
                "corpus_sha256": provenance["corpus_sha256"],
                "embedding_model": provenance["embedding_model"],
                "corpus_builder": provenance["corpus_builder"],
                "generated_from_commit": provenance["generated_from_commit"],
                "reference_threshold": provenance["reference_threshold"],
                "reference_threshold_source": provenance["reference_threshold_source"],
                "reference_threshold_mode": provenance["reference_threshold_mode"],
                "experiment_scope_sha256": provenance["experiment_scope_sha256"],
                "original_query": case["query"],
                "variant_query": variant_query,
                "top1_score": round(top1_score, 6),
                "top1_doc_id": top1_doc["doc_id"],
                "top1_slot": top1_doc["slot"],
                "top1_entity": top1_doc["entity_id"],
                "top5_doc_ids": " | ".join(top5_doc_ids),
                "top5_slots": " | ".join(top5_slots),
                "top5_entities": " | ".join(top5_entities),
                "expected_hit_at_5": str(bool(hit_at_k)),
                "above_reference_threshold": str(
                    bool(top1_score >= reference_threshold)
                ),
                "top1_changed_from_original": "",
                "score_delta_from_original": "",
            }

            if variant_type == "original":
                original_by_case[case["case_id"]] = row

            rows.append(row)

    # Fill original-comparison fields.
    for row in rows:
        base = original_by_case.get(row["case_id"])
        if not base:
            row["top1_changed_from_original"] = ""
            row["score_delta_from_original"] = ""
            continue

        row["top1_changed_from_original"] = str(row["top1_doc_id"] != base["top1_doc_id"])
        row["score_delta_from_original"] = round(float(row["top1_score"]) - float(base["top1_score"]), 6)

    robustness_path = output_dir / "retrieval_query_robustness.csv"
    robustness_fields = [
        "case_id",
        "variant_id",
        "variant_type",
        "case_type",
        "corpus_document_count",
        "corpus_sha256",
        "embedding_model",
        "corpus_builder",
        "generated_from_commit",
        "reference_threshold",
        "reference_threshold_source",
        "reference_threshold_mode",
        "experiment_scope_sha256",
        "original_query",
        "variant_query",
        "top1_score",
        "top1_doc_id",
        "top1_slot",
        "top1_entity",
        "top5_doc_ids",
        "top5_slots",
        "top5_entities",
        "expected_hit_at_5",
        "above_reference_threshold",
        "top1_changed_from_original",
        "score_delta_from_original",
    ]
    write_csv(robustness_path, rows, robustness_fields)

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    sweep_rows: list[dict[str, Any]] = []

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["case_type"]].append(row)

    for threshold in thresholds:
        for case_type in sorted(by_type):
            group = by_type[case_type]
            variant_count = len(group)
            above_count = sum(float(r["top1_score"]) >= threshold for r in group)
            expected_hit_count = sum(
                (r["expected_hit_at_5"] == "True") and (float(r["top1_score"]) >= threshold)
                for r in group
            )

            sweep_rows.append(
                {
                    "threshold": threshold,
                    "case_type": case_type,
                    "corpus_document_count": provenance["corpus_document_count"],
                    "corpus_sha256": provenance["corpus_sha256"],
                    "embedding_model": provenance["embedding_model"],
                    "corpus_builder": provenance["corpus_builder"],
                    "generated_from_commit": provenance["generated_from_commit"],
                    "reference_threshold": provenance["reference_threshold"],
                    "reference_threshold_source": provenance["reference_threshold_source"],
                    "reference_threshold_mode": provenance["reference_threshold_mode"],
                    "experiment_scope_sha256": provenance["experiment_scope_sha256"],
                    "variant_count": variant_count,
                    "above_threshold_count": above_count,
                    "above_threshold_rate_pct": pct(above_count, variant_count),
                    "expected_hit_count": expected_hit_count,
                    "expected_hit_retained_rate_pct": pct(expected_hit_count, variant_count),
                }
            )

    sweep_path = output_dir / "retrieval_query_threshold_sweep.csv"
    sweep_fields = [
        "threshold",
        "case_type",
        "corpus_document_count",
        "corpus_sha256",
        "embedding_model",
        "corpus_builder",
        "generated_from_commit",
        "reference_threshold",
        "reference_threshold_source",
        "reference_threshold_mode",
        "experiment_scope_sha256",
        "variant_count",
        "above_threshold_count",
        "above_threshold_rate_pct",
        "expected_hit_count",
        "expected_hit_retained_rate_pct",
    ]
    write_csv(sweep_path, sweep_rows, sweep_fields)

    # Summary stats.
    summary_rows = []
    for case_type in sorted(by_type):
        group = by_type[case_type]
        variant_count = len(group)
        expected_hit_count = sum(r["expected_hit_at_5"] == "True" for r in group)
        above_ref_count = sum(r["above_reference_threshold"] == "True" for r in group)
        changed_count = sum(
            r["variant_type"] != "original" and r["top1_changed_from_original"] == "True"
            for r in group
        )
        non_original_count = sum(r["variant_type"] != "original" for r in group)

        summary_rows.append(
            [
                case_type,
                variant_count,
                expected_hit_count,
                f"{pct(expected_hit_count, variant_count)}%",
                above_ref_count,
                f"{pct(above_ref_count, variant_count)}%",
                changed_count,
                f"{pct(changed_count, non_original_count)}%",
            ]
        )

    sweep_summary_rows = []
    for threshold in thresholds:
        group = [r for r in rows if float(r["top1_score"]) >= threshold]
        sweep_summary_rows.append(
            [
                threshold,
                len(group),
                f"{pct(len(group), len(rows))}%",
            ]
        )

    summary_md = f"""# Retrieval Query Wording-Variation Stress-Test Summary

## Purpose

This small-corpus stress test records how retrieval behavior changes when the same query intent is expressed with deterministic wording variations.

It is a diagnostic evaluation for the current file-backed retail decision-support prototype.

## Inputs

- Cases: `eval/retrieval_threshold_cases.json`
- Retail memory facts: `retail_ops/outputs/generated_retail_memory_facts.json`
- Demo 2 memory facts: `retail_ops/outputs/generated_demo2_retail_memory_facts.json`
- Dictionary context: `retail_ops/data/DATA_DICTIONARY.md`
- Demo 2 source notes: `retail_ops/data/demo2_source_notes.md`
- Retrieval units: {provenance["corpus_document_count"]}

- Unit definition: one generated memory fact or one chunked field-contract/source-note segment; this is not a store count or a count of independent business observations.
- Corpus SHA-256: `{provenance["corpus_sha256"]}`
- Corpus builder: `{provenance["corpus_builder"]}`
- Run commit: `{provenance["generated_from_commit"]}`
- Experiment scope SHA-256: `{provenance["experiment_scope_sha256"]}`
- Applicability note: the scope hash identifies the clean experiment-relevant code and input snapshot. The run commit is retained for navigation. Later unrelated commits do not invalidate the result. Run `python3 eval/check_retrieval_result_applicability.py` to check the current scope.
- Embedding model: `{provenance["embedding_model"]}`
- Reference threshold: `{provenance["reference_threshold"]}`
- Reference threshold source: `{provenance["reference_threshold_source"]}`
- Reference threshold mode: `{provenance["reference_threshold_mode"]}`

## Variant Types

Each original query is evaluated with deterministic wording variants:

- `original`
- `shortened`
- `paraphrased`
- `typo_punctuation_noise`
- `keyword_order_changed`

## Expected-Hit Contract

For each non-negative case, `expected_hit_at_5` is true only when at least one top-5 retrieval unit satisfies all applicable `entity_id`, slot, period, and expected-term constraints.

`negative_unsupported` cases are always recorded without an expected evidence hit. Semantic similarity or a single matching keyword is not sufficient.

## Results by Case Type

{markdown_table(
    [
        "case_type",
        "variant_count",
        "expected_hit_at_5_count",
        "expected_hit_at_5_rate",
        "above_reference_threshold_count",
        "above_reference_threshold_rate",
        "top1_changed_non_original_count",
        "top1_changed_non_original_rate",
    ],
    summary_rows,
)}

## Threshold Sweep

This sweep is not an optimization procedure. It shows how many query variants remain above several simple threshold values.

{markdown_table(
    ["threshold", "variants_above_threshold", "variants_above_threshold_rate"],
    sweep_summary_rows,
)}

The full threshold sweep by case type is stored in:

- `retail_ops/outputs/retrieval_query_threshold_sweep.csv`

## Interpretation Boundary

Supported cases should generally retain expected evidence in top-k under small wording changes.

Hard-negative, entity/period-mismatch, and ambiguous comparison cases may still remain semantically close to valid evidence. That behavior reinforces the current design: retrieval threshold is useful as one signal, but it cannot be treated as an answer-decision rule.

Unsupported cases should not become answerable merely because wording changes.

`top1_changed_non_original_rate` is descriptive of the current corpus and embedding runtime; it is not evidence of model improvement. The experiment records the model name but does not fingerprint the local Ollama model binary.

Final answer behavior should still depend on entity, period, slot, source-path, and interpretation-boundary checks.
"""

    summary_path = output_dir / "retrieval_query_robustness_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    print("[PASS] Query wording-variation stress test completed.")
    print(f"Wrote: {robustness_path.relative_to(ROOT)}")
    print(f"Wrote: {sweep_path.relative_to(ROOT)}")
    print(f"Wrote: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
