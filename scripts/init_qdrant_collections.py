"""Initialize Qdrant collections used by the local memory prototype."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

OLLAMA_BASE_URL = os.getenv("INIT_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
QDRANT_BASE_URL = os.getenv("INIT_QDRANT_BASE_URL", "http://127.0.0.1:6333").rstrip("/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

COLLECTIONS = [
    os.getenv("QDRANT_CHAT_COLLECTION", "mem_chat"),
    os.getenv("QDRANT_KB_COLLECTION", "mem_kb"),
]


def request_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body) if body else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        try:
            parsed = json.loads(body) if body else {}
        except json.JSONDecodeError:
            parsed = {"error": body}
        return exc.code, parsed


def get_embedding_size() -> int:
    status, data = request_json(
        "POST",
        f"{OLLAMA_BASE_URL}/api/embed",
        {"model": EMBED_MODEL, "input": "qdrant collection dimension check"},
    )

    if status >= 400:
        raise RuntimeError(f"Ollama embedding request failed: HTTP {status} {data}")

    embeddings = data.get("embeddings") or []

    if not embeddings or not isinstance(embeddings[0], list):
        raise RuntimeError(f"Unexpected Ollama embedding response: {data}")

    return len(embeddings[0])


def collection_exists(name: str) -> bool:
    status, _ = request_json("GET", f"{QDRANT_BASE_URL}/collections/{name}")
    return status == 200


def create_collection(name: str, vector_size: int) -> None:
    if collection_exists(name):
        print(f"[SKIP] Qdrant collection already exists: {name}")
        return

    payload = {
        "vectors": {
            "size": vector_size,
            "distance": "Cosine",
        }
    }

    status, data = request_json("PUT", f"{QDRANT_BASE_URL}/collections/{name}", payload)

    if status >= 400:
        raise RuntimeError(f"Failed to create collection {name}: HTTP {status} {data}")

    print(f"[OK] Created Qdrant collection: {name} ({vector_size} dimensions)")


def main() -> int:
    vector_size = get_embedding_size()
    print(f"[OK] Embedding model ready: {EMBED_MODEL} ({vector_size} dimensions)")

    for collection_name in COLLECTIONS:
        create_collection(collection_name, vector_size)

    print("[OK] Qdrant initialization complete.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
