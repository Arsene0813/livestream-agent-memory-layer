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


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CASES_PATH = ROOT / "eval" / "retrieval_threshold_cases.json"
DEFAULT_OUTPUT_DIR = ROOT / "retail_ops" / "outputs"

CORPUS_JSON_FILES = [
    ROOT / "retail_ops" / "outputs" / "generated_retail_memory_facts.json",
    ROOT / "retail_ops" / "outputs" / "generated_demo2_retail_memory_facts.json",
]

CORPUS_TEXT_FILES = [
    ROOT / "retail_ops" / "data" / "DATA_DICTIONARY.md",
    ROOT / "retail_ops" / "data" / "demo2_source_notes.md",
]

REFERENCE_THRESHOLD_DEFAULT = 0.5707


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None and str(x) != ""]
    if isinstance(value, tuple):
        return [str(x) for x in value if x is not None and str(x) != ""]
    if isinstance(value, str):
        if not value.strip():
            return []
        return [value.strip()]
    return [str(value)]


def first_present(mapping: dict[str, Any], keys: list[str], default: Any = "") -> Any:
    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def extract_cases(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, list):
        cases = raw
    elif isinstance(raw, dict):
        cases = None
        for key in ["cases", "retrieval_threshold_cases", "items", "data"]:
            if isinstance(raw.get(key), list):
                cases = raw[key]
                break
        if cases is None:
            raise ValueError("Could not find a case list in retrieval_threshold_cases.json")
    else:
        raise ValueError("Unsupported retrieval threshold case format")

    normalized = []
    for i, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            continue

        query = first_present(
            case,
            ["query", "user_query", "question", "input_query", "prompt"],
            "",
        )
        if not str(query).strip():
            print(f"[WARN] Skipping case {i}: no query-like field found", file=sys.stderr)
            continue

        case_id = str(first_present(case, ["case_id", "id", "name"], f"case_{i:03d}"))
        case_type = str(first_present(case, ["case_type", "type", "label", "category", "group"], "unknown"))

        normalized.append(
            {
                "case_id": case_id,
                "case_type": case_type,
                "query": str(query).strip(),
                "expected_doc_ids": normalize_list(
                    first_present(case, ["expected_doc_ids", "expected_doc_id", "expected_fact_ids", "expected_fact_id"], [])
                ),
                "expected_slots": normalize_list(
                    first_present(case, ["expected_slots", "expected_slot", "expected_slot_names", "target_slot"], [])
                ),
                "expected_entities": normalize_list(
                    first_present(case, ["expected_entities", "expected_entity", "expected_entity_id", "target_entity", "target_entity_id"], [])
                ),
                "expected_source_paths": normalize_list(
                    first_present(case, ["expected_source_paths", "expected_source_path", "source_path"], [])
                ),
                "expected_keywords": normalize_list(
                    first_present(case, ["expected_keywords", "expected_terms", "must_contain", "evidence_keywords"], [])
                ),
                "raw": case,
            }
        )

    return normalized


def flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(flatten_text(x) for x in value)
    if isinstance(value, dict):
        parts = []
        for key in sorted(value.keys()):
            parts.append(f"{key}: {flatten_text(value[key])}")
        return " ".join(parts)
    return str(value)


def fact_records_from_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing corpus file: {path.relative_to(ROOT)}", file=sys.stderr)
        return []

    raw = load_json(path)
    if isinstance(raw, list):
        facts = raw
    elif isinstance(raw, dict):
        facts = None
        for key in ["facts", "memory_facts", "items", "data"]:
            if isinstance(raw.get(key), list):
                facts = raw[key]
                break
        if facts is None:
            facts = [raw]
    else:
        return []

    docs = []
    for i, fact in enumerate(facts, start=1):
        if not isinstance(fact, dict):
            continue

        entity = str(first_present(fact, ["entity_id", "entity", "store_id"], ""))
        slot = str(first_present(fact, ["slot", "slot_name", "fact_slot"], ""))
        source_path = str(first_present(fact, ["source_path", "source", "file_path"], path.as_posix()))
        fact_id = str(first_present(fact, ["fact_id", "id", "doc_id"], ""))

        if not fact_id:
            fact_id = f"{path.relative_to(ROOT)}#{i}:{entity}:{slot}"

        text = flatten_text(fact)

        docs.append(
            {
                "doc_id": fact_id,
                "doc_type": "memory_fact",
                "source_path": source_path,
                "entity": entity,
                "slot": slot,
                "text": text,
            }
        )

    return docs


def chunk_text(text: str, chunk_size: int = 1800, overlap: int = 200) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + chunk_size)
        chunks.append(clean[start:end])
        if end == len(clean):
            break
        start = max(0, end - overlap)
    return chunks


def docs_from_text_file(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        print(f"[WARN] Missing corpus text file: {path.relative_to(ROOT)}", file=sys.stderr)
        return []

    text = path.read_text(encoding="utf-8")
    docs = []
    for i, chunk in enumerate(chunk_text(text), start=1):
        docs.append(
            {
                "doc_id": f"{path.relative_to(ROOT)}#chunk_{i}",
                "doc_type": "text_chunk",
                "source_path": str(path.relative_to(ROOT)),
                "entity": "",
                "slot": "document_context",
                "text": chunk,
            }
        )
    return docs


def build_corpus() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []

    for path in CORPUS_JSON_FILES:
        docs.extend(fact_records_from_json(path))

    for path in CORPUS_TEXT_FILES:
        docs.extend(docs_from_text_file(path))

    # Deduplicate by doc_id.
    seen = set()
    deduped = []
    for doc in docs:
        if doc["doc_id"] in seen:
            continue
        seen.add(doc["doc_id"])
        deduped.append(doc)

    return deduped


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


def expected_hit(case: dict[str, Any], docs: list[dict[str, Any]]) -> bool:
    expected_doc_ids = set(case["expected_doc_ids"])
    expected_slots = set(case["expected_slots"])
    expected_entities = set(case["expected_entities"])
    expected_source_paths = set(case["expected_source_paths"])
    expected_keywords = [kw.lower() for kw in case["expected_keywords"]]

    for doc in docs:
        doc_id = str(doc.get("doc_id", ""))
        slot = str(doc.get("slot", ""))
        entity = str(doc.get("entity", ""))
        source_path = str(doc.get("source_path", ""))
        text = str(doc.get("text", "")).lower()

        if expected_doc_ids and doc_id in expected_doc_ids:
            return True

        if expected_source_paths:
            if source_path in expected_source_paths or any(p in source_path for p in expected_source_paths):
                return True

        if expected_keywords and any(kw.lower() in text for kw in expected_keywords):
            return True

        if expected_slots and expected_entities:
            if slot in expected_slots and entity in expected_entities:
                return True

        elif expected_slots:
            if slot in expected_slots:
                return True

        elif expected_entities:
            if entity in expected_entities:
                return True

    return False


def pct(n: int, d: int) -> float:
    if d == 0:
        return 0.0
    return round(n / d * 100.0, 2)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
    parser.add_argument("--reference-threshold", type=float, default=REFERENCE_THRESHOLD_DEFAULT)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    cases_path = Path(args.cases)
    output_dir = Path(args.output_dir)

    if not cases_path.exists():
        raise SystemExit(f"[FAIL] Cases file not found: {cases_path}")

    raw_cases = load_json(cases_path)
    cases = extract_cases(raw_cases)
    if not cases:
        raise SystemExit("[FAIL] No usable cases found.")

    docs = build_corpus()
    if not docs:
        raise SystemExit("[FAIL] No corpus documents found.")

    print(f"[INFO] Loaded cases: {len(cases)}")
    print(f"[INFO] Built corpus docs: {len(docs)}")
    print(f"[INFO] Embedding corpus with model={args.model}")

    doc_embeddings = []
    for i, doc in enumerate(docs, start=1):
        if i % 25 == 0 or i == 1 or i == len(docs):
            print(f"[INFO] Embedding doc {i}/{len(docs)}")
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

            hit_at_k = expected_hit(case, top_docs)
            top5_doc_ids = [doc["doc_id"] for _, doc in top]
            top5_slots = [doc["slot"] for _, doc in top]
            top5_entities = [doc["entity"] for _, doc in top]

            row = {
                "case_id": case["case_id"],
                "variant_id": f"{case['case_id']}::{variant_index:02d}_{variant_type}",
                "variant_type": variant_type,
                "case_type": case["case_type"],
                "original_query": case["query"],
                "variant_query": variant_query,
                "top1_score": round(top1_score, 6),
                "top1_doc_id": top1_doc["doc_id"],
                "top1_slot": top1_doc["slot"],
                "top1_entity": top1_doc["entity"],
                "top5_doc_ids": " | ".join(top5_doc_ids),
                "top5_slots": " | ".join(top5_slots),
                "top5_entities": " | ".join(top5_entities),
                "expected_hit_at_5": str(bool(hit_at_k)),
                "above_reference_threshold": str(bool(top1_score >= args.reference_threshold)),
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
- Embedding model: `{args.model}`
- Reference threshold: `{args.reference_threshold}`

## Variant Types

Each original query is evaluated with deterministic wording variants:

- `original`
- `shortened`
- `paraphrased`
- `typo_punctuation_noise`
- `keyword_order_changed`

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
