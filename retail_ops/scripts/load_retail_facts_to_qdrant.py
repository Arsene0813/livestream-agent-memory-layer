import argparse
import asyncio
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import httpx


DEFAULT_FACTS_PATH = "retail_ops/outputs/generated_retail_memory_facts.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "http://localhost:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
QDRANT_KB_COLLECTION = os.getenv("QDRANT_KB_COLLECTION", "mem_kb")


def normalize_entity_id(entity_id: str | None) -> str:
    return (entity_id or "").strip().lower().replace(" ", "_")


def stable_point_id(fact: dict[str, Any]) -> str:
    raw = "|".join([
        str(fact.get("kind", "")),
        str(fact.get("type", "")),
        str(fact.get("entity_id", "")),
        str(fact.get("slot", "")),
        str(fact.get("period_label", "")),
        str(fact.get("period_start", "")),
        str(fact.get("period_end", "")),
    ])
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


def build_retrieval_text(fact: dict[str, Any]) -> str:
    parts = [
        "Domain: retail_ops",
        f"Entity: {fact.get('entity_id')}",
        f"Slot: {fact.get('slot')}",
        f"Type: {fact.get('type')}",
        f"Period: {fact.get('period_label') or fact.get('period_month') or ''}",
        f"Value: {fact.get('value')}",
    ]

    observed = fact.get("observed_values")
    if observed:
        parts.append(f"Observed values: {json.dumps(observed, ensure_ascii=False)}")

    threshold = fact.get("threshold")
    if threshold:
        parts.append(f"Threshold: {json.dumps(threshold, ensure_ascii=False)}")

    calculation = fact.get("calculation")
    if calculation:
        parts.append(f"Calculation: {calculation}")

    limitations = fact.get("limitations") or []
    if limitations:
        parts.append("Limitations: " + "; ".join(str(x) for x in limitations))

    source_path = fact.get("source_path")
    if source_path:
        parts.append(f"Source path: {source_path}")

    return "\n".join(parts)


async def embed_text(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
        )
        r.raise_for_status()
        data = r.json()

    if "embeddings" in data:
        return data["embeddings"][0]
    if "embedding" in data:
        return data["embedding"]

    raise RuntimeError(f"Unexpected Ollama embedding response keys: {list(data.keys())}")


async def ensure_collection(vector_size: int) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}")

        if r.status_code == 200:
            return

        if r.status_code != 404:
            r.raise_for_status()

        body = {
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        }

        r = await client.put(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}",
            json=body,
        )
        r.raise_for_status()


async def upsert_points(points: list[dict[str, Any]]) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.put(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points",
            json={"points": points},
        )
        r.raise_for_status()


def build_payload(fact: dict[str, Any], retrieval_text: str) -> dict[str, Any]:
    entity_id = str(fact.get("entity_id", "")).strip()

    payload = dict(fact)
    payload.update({
        "domain": "retail_ops",
        "memory_layer": "retail_ops_demo",
        "entity_id": entity_id,
        "entity_id_norm": normalize_entity_id(entity_id),
        "retrieval_text": retrieval_text,
        "is_active": bool(fact.get("is_active", True)),
    })
    return payload


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--facts", default=DEFAULT_FACTS_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    facts_path = Path(args.facts)
    if not facts_path.exists():
        raise FileNotFoundError(f"Missing facts file: {facts_path}")

    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    if not isinstance(facts, list):
        raise TypeError("Retail facts JSON must be a list of fact objects.")

    points: list[dict[str, Any]] = []
    loaded = 0

    for fact in facts:
        if not fact.get("is_active", True):
            continue

        retrieval_text = build_retrieval_text(fact)
        vector = await embed_text(retrieval_text)

        if loaded == 0 and not points:
            await ensure_collection(len(vector))

        payload = build_payload(fact, retrieval_text)

        points.append({
            "id": stable_point_id(fact),
            "vector": vector,
            "payload": payload,
        })
        loaded += 1

        if len(points) >= args.batch_size:
            await upsert_points(points)
            points.clear()

    if points:
        await upsert_points(points)

    print(f"Loaded {loaded} active retail facts into Qdrant collection: {QDRANT_KB_COLLECTION}")


if __name__ == "__main__":
    asyncio.run(main())
