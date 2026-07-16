from __future__ import annotations

import csv
import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from retail_retrieval_corpus import (
        FACT_JSON_PATHS,
        FIELD_CONTRACT_PATHS,
        corpus_provenance,
        load_retail_retrieval_documents as load_documents,
    )
except ModuleNotFoundError:
    from eval.retail_retrieval_corpus import (
        FACT_JSON_PATHS,
        FIELD_CONTRACT_PATHS,
        corpus_provenance,
        load_retail_retrieval_documents as load_documents,
    )

try:
    from retrieval_case_validation import validate_retrieval_cases
except ModuleNotFoundError:
    from eval.retrieval_case_validation import validate_retrieval_cases



ROOT = Path(".")
CASES_PATH = ROOT / "eval/retrieval_threshold_cases.json"


DETAIL_CSV_PATH = ROOT / "retail_ops/outputs/retrieval_score_distribution.csv"
SUMMARY_MD_PATH = ROOT / "retail_ops/outputs/retrieval_threshold_summary.md"
PNG_PATH = ROOT / "retail_ops/outputs/retrieval_score_distribution.png"

OLLAMA_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "bge-m3"
TOP_K = 5


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def embed(text: str, retries: int = 2) -> list[float]:
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
            embedding = data.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise RuntimeError(f"Ollama returned no embedding for model {EMBED_MODEL}")
            return [float(x) for x in embedding]
        except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1)

    raise RuntimeError(
        "Could not get embeddings from Ollama. Make sure Ollama is running and bge-m3 is pulled.\n"
        "Try:\n"
        "  docker compose up -d ollama\n"
        "  docker exec -it oc_ollama ollama pull bge-m3\n"
        f"Last error: {last_error}"
    )


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def text_contains_all(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(str(term).lower() in lowered for term in terms)


def period_term_match(doc: dict[str, Any], expected_period_terms: list[str]) -> bool:
    if not expected_period_terms:
        return True

    period_text = " ".join([
        doc.get("period_label", ""),
        doc.get("period_start", ""),
        doc.get("period_end", ""),
        doc.get("text", ""),
    ]).lower()

    return any(str(term).lower() in period_text for term in expected_period_terms)


def expected_match(case: dict[str, Any], doc: dict[str, Any]) -> bool:
    if case["case_type"] == "negative_unsupported":
        return False

    expected_entity = case.get("expected_entity")
    expected_slot = case.get("expected_slot")
    expected_terms = case.get("expected_terms", [])
    expected_period_terms = case.get("expected_period_terms", [])

    entity_ok = True if not expected_entity else doc["entity_id"] == expected_entity
    slot_ok = True if not expected_slot else doc["slot"] == expected_slot
    period_ok = period_term_match(doc, expected_period_terms)
    terms_ok = text_contains_all(doc["text"], expected_terms)

    return bool(entity_ok and slot_ok and period_ok and terms_ok)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p
    lower = math.floor(k)
    upper = math.ceil(k)
    if lower == upper:
        return sorted_values[int(k)]
    return sorted_values[lower] * (upper - k) + sorted_values[upper] * (k - lower)


def make_plot(rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required to generate retrieval_score_distribution.png. "
            "Install it with: pip install matplotlib"
        ) from exc

    top1_rows = [row for row in rows if int(row["rank"]) == 1]
    case_types = [
        "positive_supported",
        "negative_unsupported",
        "hard_negative_boundary",
        "entity_period_mismatch",
        "ambiguous_comparison",
    ]

    grouped = {case_type: [] for case_type in case_types}
    for row in top1_rows:
        grouped.setdefault(str(row["case_type"]), []).append(float(row["score"]))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x_positions = list(range(1, len(case_types) + 1))
    data = [grouped.get(case_type, []) for case_type in case_types]

    ax.boxplot(data, positions=x_positions, widths=0.5, showfliers=False)

    for idx, values in enumerate(data, start=1):
        for point_idx, score in enumerate(values):
            jitter = ((point_idx % 5) - 2) * 0.035
            ax.plot(idx + jitter, score, marker="o", linestyle="None", markersize=4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(case_types, rotation=25, ha="right")
    ax.set_ylabel("Top-1 retrieval cosine score")
    ax.set_title("Prototype retrieval score distribution by case type")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=180)
    plt.close(fig)


def main() -> int:
    raw_cases = load_json(CASES_PATH)
    cases = validate_retrieval_cases(
        raw_cases,
        source=str(CASES_PATH),
    )

    docs = load_documents()
    provenance = corpus_provenance(docs, EMBED_MODEL)

    print(f"Loaded {len(cases)} retrieval threshold cases.")
    print(f"Loaded {len(docs)} retrieval documents.")
    print(f"Embedding model: {EMBED_MODEL}")
    print("Embedding documents...")

    doc_embeddings = [embed(doc["text"]) for doc in docs]

    rows: list[dict[str, Any]] = []
    top1_by_type: dict[str, list[float]] = defaultdict(list)
    margin_by_type: dict[str, list[float]] = defaultdict(list)
    hit_by_type: dict[str, list[int]] = defaultdict(list)

    print("Scoring cases...")

    for case in cases:
        query_embedding = embed(case["query"])
        scored = []

        for doc, doc_embedding in zip(docs, doc_embeddings):
            scored.append((cosine(query_embedding, doc_embedding), doc))

        scored.sort(key=lambda item: item[0], reverse=True)
        top_k = scored[:TOP_K]

        top1 = top_k[0][0] if top_k else 0.0
        top2 = top_k[1][0] if len(top_k) > 1 else 0.0
        margin = top1 - top2

        top1_by_type[case["case_type"]].append(top1)
        margin_by_type[case["case_type"]].append(margin)

        expected_hit = 0

        for rank, (score, doc) in enumerate(top_k, start=1):
            is_match = expected_match(case, doc)
            expected_hit = max(expected_hit, int(is_match))

            entity_match = (
                "" if not case.get("expected_entity")
                else str(doc["entity_id"] == case.get("expected_entity"))
            )
            slot_match = (
                "" if not case.get("expected_slot")
                else str(doc["slot"] == case.get("expected_slot"))
            )
            period_match = (
                "" if not case.get("expected_period_terms")
                else str(period_term_match(doc, case.get("expected_period_terms", [])))
            )

            rows.append({
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "corpus_document_count": provenance["corpus_document_count"],
                "corpus_sha256": provenance["corpus_sha256"],
                "embedding_model": provenance["embedding_model"],
                "corpus_builder": provenance["corpus_builder"],
                "generated_from_commit": provenance["generated_from_commit"],
                "query": case["query"],
                "rank": rank,
                "score": round(float(score), 6),
                "top1_minus_top2_margin": round(float(margin), 6),
                "retrieved_doc_id": doc["doc_id"],
                "retrieved_doc_type": doc["doc_type"],
                "retrieved_entity": doc["entity_id"],
                "retrieved_slot": doc["slot"],
                "retrieved_period_label": doc["period_label"],
                "retrieved_period_start": doc["period_start"],
                "retrieved_period_end": doc["period_end"],
                "retrieved_source_file": doc["source_file"],
                "retrieved_source_path": doc["source_path"],
                "expected_entity": case.get("expected_entity") or "",
                "expected_slot": case.get("expected_slot") or "",
                "expected_period_terms": "; ".join(case.get("expected_period_terms", [])),
                "entity_match": entity_match,
                "slot_match": slot_match,
                "period_match": period_match,
                "is_expected_match": int(is_match),
            })

        hit_by_type[case["case_type"]].append(expected_hit)

    DETAIL_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with DETAIL_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    make_plot(rows)

    lines = [
        "# Retrieval Threshold Inspection Summary",
        "",
        "This file summarizes prototype retrieval score distributions over file-backed retail evidence.",
        "",
        "The retrieval corpus combines generated Demo 1 and Demo 2 retail memory facts with selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.",
        "",
        "The current project does not connect to the live Meituan backend. The evidence used here comes from manually structured Meituan-style backend data and generated local memory facts.",
        "",
        "The purpose is to inspect retrieval-threshold behavior. It is not a production-level threshold validation, a broad LLM benchmark, or proof that retrieved evidence is sufficient for an operating conclusion.",
        "",
        "## Corpus",
        "",
        f"- Retrieval documents loaded: {provenance['corpus_document_count']}",
        f"- Corpus SHA-256: `{provenance['corpus_sha256']}`",
        f"- Corpus builder: `{provenance['corpus_builder']}`",
        f"- Generated from commit: `{provenance['generated_from_commit']}`",
        f"- Retrieval threshold cases: {len(cases)}",
        f"- Embedding model: `{provenance['embedding_model']}` via local Ollama",
        "- Generated memory fact sources:",
    ]

    for path in FACT_JSON_PATHS:
        lines.append(f"  - `{path}`")

    lines.append("- Field-contract sources:")
    for path in FIELD_CONTRACT_PATHS:
        lines.append(f"  - `{path}`")

    lines.extend([
        "",
        "## Case Groups",
        "",
        "| Case type | Purpose |",
        "|---|---|",
        "| positive_supported | Queries with expected supporting evidence in the current generated retail facts or field-contract notes. |",
        "| negative_unsupported | Queries that should not have enough evidence in the current corpus. |",
        "| hard_negative_boundary | Queries that may retrieve related facts but still require refusal or qualification. |",
        "| entity_period_mismatch | Queries that mention an entity, period, or demo scope not supported by the retrieved fact. |",
        "| ambiguous_comparison | Broad comparison queries where multiple records may be relevant. |",
        "",
        "## Score Distribution by Case Type",
        "",
        "| Case type | Cases | Top-1 min | Top-1 p25 | Top-1 median | Top-1 p75 | Top-1 max | Median margin | Expected hit@5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for case_type in sorted(top1_by_type):
        values = top1_by_type[case_type]
        margins = margin_by_type[case_type]
        hits = hit_by_type[case_type]

        lines.append(
            f"| {case_type} | {len(values)} | "
            f"{min(values):.4f} | {percentile(values, 0.25):.4f} | "
            f"{percentile(values, 0.50):.4f} | {percentile(values, 0.75):.4f} | "
            f"{max(values):.4f} | {percentile(margins, 0.50):.4f} | "
            f"{sum(hits)}/{len(hits)} |"
        )

    positive_values = top1_by_type.get("positive_supported", [])
    negative_values = top1_by_type.get("negative_unsupported", [])

    if positive_values and negative_values:
        pos_p25 = percentile(positive_values, 0.25)
        neg_p75 = percentile(negative_values, 0.75)
        candidate = (pos_p25 + neg_p75) / 2
        candidate_note = (
            f"An exploratory reference threshold from this small corpus is around `{candidate:.4f}`, "
            f"midway between the positive-supported p25 score `{pos_p25:.4f}` and the "
            f"negative-unsupported p75 score `{neg_p75:.4f}`. This is an inspection "
            "reference, not a production cutoff or an answer-decision rule."
        )
    else:
        candidate_note = "A reference threshold should be interpreted only after inspecting the generated score table."

    lines.extend([
        "",
        "## Threshold Interpretation",
        "",
        "A reference threshold can be inspected for the trade-off between unsupported retrieval noise and supported evidence retention.",
        "",
        candidate_note,
        "",
        "High scores in hard-negative boundary cases are expected. They show that a semantically related fact can be retrieved even when the correct answer should still refuse or qualify the requested conclusion.",
        "",
        "For that reason, retrieval thresholding must be paired with entity, period, slot, source-path, and answer-boundary checks.",
        "",
        "Entity/period mismatch cases should not be treated as answerable merely because a related store or metric is retrieved.",
        "",
        "Ambiguous comparison cases should be narrowed by metric and operating question before the system makes a comparison.",
        "",
        "Because the current corpus is small, this score distribution is used for inspection of retrieval behavior and failure modes rather than production-level threshold validation.",
        "",
        "## Outputs",
        "",
        f"- Detail CSV: `{DETAIL_CSV_PATH}`",
        f"- Score distribution plot: `{PNG_PATH}`",
    ])

    SUMMARY_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {DETAIL_CSV_PATH}")
    print(f"Wrote {SUMMARY_MD_PATH}")
    print(f"Wrote {PNG_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
