# version test
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import httpx
import os
import time
import uuid
import json
import asyncio

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")

QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "http://qdrant:6333")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")
QDRANT_KB_COLLECTION = os.getenv("QDRANT_KB_COLLECTION", "mem_kb")
QDRANT_CHAT_COLLECTION = os.getenv("QDRANT_CHAT_COLLECTION", "mem_chat")
KB_TOP_K = int(os.getenv("KB_TOP_K", "3"))
CHAT_TOP_K = int(os.getenv("CHAT_TOP_K", "5"))
CHAT_SCORE_THRESHOLD = float(os.getenv("CHAT_SCORE_THRESHOLD", "0.75"))
KB_SCORE_THRESHOLD = float(os.getenv("KB_SCORE_THRESHOLD", "0.75"))
CHAT_BASE_THRESHOLD = float(os.getenv("CHAT_BASE_THRESHOLD", "0.60"))
CHAT_GAP_THRESHOLD  = float(os.getenv("CHAT_GAP_THRESHOLD",  "0.05"))
DEBUG_ROUTES_ENABLED = os.getenv("DEBUG_ROUTES_ENABLED", "true").lower() == "true"

LIVESTREAM_FACT_TYPES = {
    "product_price",
    "promo",
    "stock_status",
    "shipping_policy",
    "product_feature",
}

def get_livestream_score_threshold(fact_type: str | None) -> float:
    if fact_type == "product_feature":
        return 0.60
    return 0.65

app = FastAPI(title="Agent API", version="0.1.0")

class ChatReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)
    system: str | None = Field(default=None, max_length=4000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

async def ollama_embed(text: str) -> list[float]:
    last_error = None

    for _ in range(3):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": EMBED_MODEL, "input": text},
                )
                r.raise_for_status()
                data = r.json()
                return data["embeddings"][0]

        except Exception as e:
            last_error = e
            await asyncio.sleep(1)

    raise last_error

async def qdrant_upsert_chat(
    session_id: str,
    role: str,
    text: str,
    vector: list[float],
    fact_type: str | None = None,
):

    point_id = str(uuid.uuid4())
    payload = {
        "type": "chat",
        "session_id": session_id,
        "role": role,
        "text": text,
        "fact_type": fact_type,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    body = {"points": [{"id": point_id, "vector": vector, "payload": payload}]}

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.put(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_CHAT_COLLECTION}/points",
            json=body,
        )
        r.raise_for_status()
        return r.json()

async def qdrant_search_chat(session_id: str, query_vector: list[float], limit: int):
    body = {
        "query": query_vector,
        "limit": limit,
        "with_payload": True,
        "filter": {"must": [{"key": "session_id", "match": {"value": session_id}}]},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_CHAT_COLLECTION}/points/query",
            json=body,
        )
        r.raise_for_status()
        return r.json().get("result", {}).get("points", [])


async def qdrant_scroll_chat(session_id: str, limit: int = 20):
    body = {
        "limit": limit,
        "with_payload": True,
        "with_vectors": False,
        "filter": {"must": [{"key": "session_id", "match": {"value": session_id}}]},
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_CHAT_COLLECTION}/points/scroll",
            json=body,
        )
        r.raise_for_status()
        return r.json().get("result", {}).get("points", [])
async def qdrant_delete_chat_facts_by_session_and_type(session_id: str, fact_type: str):
    body = {
        "filter": {
            "must": [
                {"key": "session_id", "match": {"value": session_id}},
                {"key": "role", "match": {"value": "user"}},
                {"key": "fact_type", "match": {"value": fact_type}},
            ]
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_CHAT_COLLECTION}/points/delete",
            json=body,
        )
        r.raise_for_status()
        return r.json()

async def get_last_user_message(session_id: str):
    points = await qdrant_scroll_chat(session_id, limit=50)
    user_points = []
    for p in points:
        payload = p.get("payload") or {}
        if payload.get("role") == "user":
            user_points.append(payload)

    if not user_points:
        return None

    user_points.sort(key=lambda x: x.get("ts") or "")
    return user_points[-1].get("text")

def should_store_user_memory(text: str) -> bool:
    t = (text or "").strip()

    if not t:
        return False

    # 太短，通常没有存储价值
    if len(t) < 2:
        return False

    # 纯数字，通常没有存储价值
    if t.isdigit():
        return False

    # 纯符号/标点，通常没有存储价值
    if all(not ch.isalnum() for ch in t):
        return False

    return True

async def should_store_user_memory_llm(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False

    prompt = (
        "You are a memory gate for a long-term memory assistant.\n"
        "Decide whether the following message contains information worth storing as memory.\n\n"
        "Store durable personal facts, preferences, identity, stable relationships, and operationally meaningful domain facts.\n"
        "For livestream or commerce-related messages, store facts such as product prices, promotions, stock status, shipping policies, and product features.\n"
        "Do NOT store questions, vague statements, or temporary requests unless they clearly contain a stable or actionable fact.\n\n"
        f"User message: {t}\n\n"
        "Answer YES or NO."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    answer = data.get("message", {}).get("content", "").strip().upper()
    return answer.startswith("YES")

def build_extract_fact_prompt(text: str) -> str:
    t = (text or "").strip()

    return (
        "You are an information extraction system.\n"
        "Extract one stable or operationally meaningful fact from the following message if possible.\n"
        "Valid fact types include: preference, identity, relationship, location, pet_name, "
        "product_price, promo, stock_status, shipping_policy, product_feature.\n"
        "Only extract if the message clearly contains a fact that should be stored as memory.\n"
        "For livestream or commerce-related messages, examples include product prices, promotions, "
        "stock status, shipping policies, and product features.\n"
        "Examples:\n"
        'User message: 这款隐形眼镜价格是99元\n'
        'Output: {"type":"product_price","value":"99元","source_text":"这款隐形眼镜价格是99元"}\n'
        'User message: 本场活动满199减30\n'
        'Output: {"type":"promo","value":"满199减30","source_text":"本场活动满199减30"}\n'
        'User message: 护理液目前现货\n'
        'Output: {"type":"stock_status","value":"现货","source_text":"护理液目前现货"}\n'
        'User message: 本商品支持次日达\n'
        'Output: {"type":"shipping_policy","value":"次日达","source_text":"本商品支持次日达"}\n'
        'User message: 这款产品主打高透氧\n'
        'Output: {"type":"product_feature","value":"高透氧","source_text":"这款产品主打高透氧"}\n'
        "If no suitable fact is present, answer exactly: NONE\n\n"
        "Return JSON only, with this schema:\n"
        '{"type":"...", "value":"...", "source_text":"..."}\n\n'
        f"User message: {t}"
    )

async def extract_structured_fact(text: str):
    t = (text or "").strip()
    if not t:
        return None

    prompt = build_extract_fact_prompt(t)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    content = data.get("message", {}).get("content", "").strip()

    if content.upper() == "NONE":
        return None

    fact = try_parse_json_object(content)
    if fact is None:
        return None

    required_keys = {"type", "value", "source_text"}
    if not required_keys.issubset(fact):
        return None

    fact["type"] = normalize_fact_type(str(fact.get("type")).strip())
    fact["value"] = str(fact.get("value")).strip()
    fact["source_text"] = str(fact.get("source_text")).strip()

    if not fact["type"] or not fact["value"] or not fact["source_text"]:
        return None

    return fact

def try_parse_json_object(text: str) -> dict | None:
    if not text:
        return None

    raw = text.strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            raw = "\n".join(lines[1:-1]).strip()

    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    return None

def is_fact_fresh(source_ts: str | None, freshness_days: int | None) -> bool:
    if not source_ts or freshness_days is None:
        return True

    try:
        source_time = time.strptime(source_ts, "%Y-%m-%dT%H:%M:%SZ")
        source_epoch = time.mktime(source_time)
        now_epoch = time.time()
        age_days = (now_epoch - source_epoch) / 86400
        return age_days <= freshness_days
    except Exception:
        return False

def normalize_fact_type(fact_type: str | None) -> str | None:
    if fact_type is None:
        return None

    t = fact_type.strip().lower()
    t = t.replace(" ", "_")

    mapping = {
        "pet_name": "pet_name",
        "location": "location",
        "preference": "preference",
        "identity": "identity",
        "relationship": "relationship",
        "product_price": "product_price",
        "promo": "promo",
        "stock_status": "stock_status",
        "shipping_policy": "shipping_policy",
        "product_feature": "product_feature",
    }

    return mapping.get(t, t)

async def kb_delete_facts_by_session_and_type(session_id: str, fact_type: str):
    now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    body = {
        "payload": {
            "is_active": False,
            "deactivated_ts": now_ts,
        },
        "filter": {
            "must": [
                {"key": "session_id", "match": {"value": session_id}},
                {"key": "type", "match": {"value": fact_type}},
                {"key": "is_active", "match": {"value": True}},
            ]
        }
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points/payload",
            json=body,
        )
        r.raise_for_status()
        return r.json()

async def kb_upsert_fact(session_id: str, fact: dict, vector: list[float] | None = None):
    fact["type"] = normalize_fact_type(fact.get("type"))
    if fact.get("type") is not None:
        await kb_delete_facts_by_session_and_type(session_id, fact["type"])


    fact_text = json.dumps(fact, ensure_ascii=False)
    embed_text = fact.get("source_text") or fact_text

    if vector is not None:
        vec = vector
    else:
        vec = await ollama_embed(embed_text)

    point_id = str(uuid.uuid4())
    now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    fact_type = fact.get("type")

    if fact_type == "promo":
        freshness_days = 1
    elif fact_type == "stock_status":
        freshness_days = 3
    elif fact_type == "product_price":
        freshness_days = 7
    else:
        freshness_days = 30

    body = {
        "points": [
            {
                "id": point_id,
                "vector": vec,
                "payload": {
                    "kind": "structured_fact",
                    "session_id": session_id,
                    "type": fact.get("type"),
                    "value": fact.get("value"),
                    "source_text": fact.get("source_text"),
                    "text": fact_text,
                    "ts": now_ts,
                    "source_ts": now_ts,
                    "last_seen_ts": now_ts,
                    "freshness_days": freshness_days,
                    "is_active": True,
                }
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.put(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points",
            json=body,
        )
        r.raise_for_status()
        return {"id": point_id, "result": r.json()}

async def kb_touch_fact(point_id: str):
    now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    body = {
        "payload": {
            "last_seen_ts": now_ts
        },
        "points": [point_id],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points/payload",
            json=body,
        )
        r.raise_for_status()
        return r.json()

async def store_user_memory_if_needed(
    session_id: str,
    message: str,
    qvec: list[float],
    current_fact: dict | None,
):
    fact_extracted = False
    kb_upsert_ok = False
    kb_upsert_error = None
    memory_llm_ok = None
    skipped_reason = None

    if not (message.strip() and should_store_user_memory(message)):
        return {
            "memory_llm_ok": memory_llm_ok,
            "fact_extracted": fact_extracted,
            "kb_upsert_ok": kb_upsert_ok,
            "kb_upsert_error": kb_upsert_error,
            "skipped_reason": skipped_reason,
        }

    memory_llm_ok = await should_store_user_memory_llm(message)
    if not memory_llm_ok:
        return {
            "memory_llm_ok": memory_llm_ok,
            "fact_extracted": fact_extracted,
            "kb_upsert_ok": kb_upsert_ok,
            "kb_upsert_error": kb_upsert_error,
            "skipped_reason": skipped_reason,
        }

    last_user_text = await get_last_user_message(session_id)
    if last_user_text == message:
        skipped_reason = "duplicate_message"
        return {
            "memory_llm_ok": memory_llm_ok,
            "fact_extracted": fact_extracted,
            "kb_upsert_ok": kb_upsert_ok,
            "kb_upsert_error": kb_upsert_error,
            "skipped_reason": skipped_reason,
        }

    fact = current_fact
    if fact is not None:
        fact_extracted = True
        fact_type = normalize_fact_type(fact.get("type"))
        if fact_type is not None:
            await qdrant_delete_chat_facts_by_session_and_type(session_id, fact_type)

        await qdrant_upsert_chat(
            session_id,
            "user",
            message,
            qvec,
            fact_type=fact_type,
        )

        try:
            await kb_upsert_fact(session_id, fact, qvec)
            kb_upsert_ok = True
        except Exception as e:
            kb_upsert_error = str(e)
    else:
        await qdrant_upsert_chat(session_id, "user", message, qvec)

    return {
        "memory_llm_ok": memory_llm_ok,
        "fact_extracted": fact_extracted,
        "kb_upsert_ok": kb_upsert_ok,
        "kb_upsert_error": kb_upsert_error,
        "skipped_reason": skipped_reason,
    }

@app.get("/health")
async def health():
    return {"ok": True, "model": OLLAMA_MODEL}

@app.post("/chat")
async def chat(req: ChatReq):
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": []
    }
    if req.system:
        payload["messages"].append({"role": "system", "content": req.system})
    payload["messages"].append({"role": "user", "content": req.message})

    if req.temperature is not None:
        payload["options"] = {"temperature": req.temperature}

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    # Ollama /api/chat returns {"message": {"role": "...", "content": "..."}, ...}
    return {
        "model": OLLAMA_MODEL,
        "reply": data.get("message", {}).get("content", ""),
        "raw": data,
    }


@app.get("/test")
async def test():
    return {"ok": True}

@app.post("/chat_mem")
async def chat_mem(req: ChatReq):
    session_id = req.session_id or "demo"
    current_fact = None
    current_fact_type = None

    # 1) embedding 用户输入（只算向量，不写库）
    qvec = await ollama_embed(req.message)

    # 2) 先检索历史记忆（注意：此时还没把本次 user 写进去，所以不会“命中自己”）
    hits = await qdrant_search_chat(session_id, qvec, CHAT_TOP_K)
    candidate_hits = []
    for h in hits:
        p = h.get("payload") or {}
        txt = (p.get("text") or "").strip()
        role = p.get("role")

        if not txt:
            continue
        if role != "user":
            continue
        if not should_store_user_memory(txt):
            continue

        candidate_hits.append(h)

    if should_store_user_memory(req.message):
        try:
            current_fact = await extract_structured_fact(req.message)
            if current_fact is not None:
                current_fact_type = normalize_fact_type(current_fact.get("type"))
        except Exception:
            current_fact = None
            current_fact_type = None

    if current_fact_type is not None:
        filtered_candidates = []
        for h in candidate_hits:
            p = h.get("payload") or {}
            old_fact_type = normalize_fact_type(p.get("fact_type"))
            if old_fact_type == current_fact_type:
                continue
            filtered_candidates.append(h)
        candidate_hits = filtered_candidates

    # ---- Dynamic gating (research-y) ----
    top1 = candidate_hits[0].get("score") if len(candidate_hits) >= 1 else None
    top2 = candidate_hits[1].get("score") if len(candidate_hits) >= 2 else None

    top1_score = top1
    top2_score = top2

    score_th = CHAT_BASE_THRESHOLD
    gap_ok = False
    th_ok = False

    if top1 is not None:
        th_ok = top1 >= CHAT_BASE_THRESHOLD

        if top2 is None:
            gap_ok = True
        elif top1 >= 0.92:
            gap_ok = True
        else:
            gap_ok = (top1 - top2) >= CHAT_GAP_THRESHOLD

    memory_allowed = bool(th_ok and gap_ok)

    if current_fact_type is not None and len(candidate_hits) == 0:
        memory_allowed = True
    # ------------------------------------

    # 3) 如果不允许用记忆：只写入一次 user（可选，但推荐），然后直接拒绝
    if not memory_allowed:
        memres = await store_user_memory_if_needed(
            session_id=session_id,
            message=req.message,
            qvec=qvec,
            current_fact=current_fact,
        )

        return {
            "session_id": session_id,
            "model": OLLAMA_MODEL,
            "reply": "...",
            "refusal": True,
            "sources": [],
            "fact_debug": {
                "memory_llm_ok": memres["memory_llm_ok"],
                "extracted": memres["fact_extracted"],
                "kb_upsert_ok": memres["kb_upsert_ok"],
                "kb_upsert_error": memres["kb_upsert_error"],
                "skipped_reason": memres["skipped_reason"],
            },
            "retrieval": {
                "query": req.message,
                "top_k": CHAT_TOP_K,
                "base_threshold": CHAT_BASE_THRESHOLD,
                "gap_threshold": CHAT_GAP_THRESHOLD,
                "top1_score": top1_score,
                "top2_score": top2_score,
                "score_threshold": score_th,
                "dynamic_threshold": dynamic_th,
                "th_ok": th_ok,
                "gap_ok": gap_ok,
                "hit_count_raw": len(hits),
                "hit_count_candidates": len(candidate_hits),
                "hit_count_used": 0,
            },
        }


    # 4) 通过 gating：把 hits 里的 payload 做成可追溯 sources + 引用编号
    contexts = []  # 给 LLM 的文本块，带 [1][2]...
    sources = []   # 返回给用户的可追溯信息
    cite_no = 0

    for h in candidate_hits:
        score = h.get("score")
        point_id = h.get("id")
        p = h.get("payload") or {}

        txt = p.get("text") or ""
        role = p.get("role")
        ts = p.get("ts")

        if txt.strip() == "":
            continue
        if role != "user":
            continue
        if not should_store_user_memory(txt):
            continue

        cite_no += 1
        contexts.append(f"[{cite_no}] {txt}")

        sources.append({
            "cite": cite_no,
            "point_id": point_id,
            "score": score,
            "collection": QDRANT_CHAT_COLLECTION,
            "role": role,
            "ts": ts,
            "text": txt,
        })

    # 5) 拼 system prompt
    system_text = req.system or "你是一个有长期记忆的助手。"
    if contexts:
        memory_block = (
            "以下是与当前问题相关的历史记忆（可能不完全准确）。"
            "如果使用了其中信息，请用 [编号] 标注引用：\n"
            + "\n".join(contexts[:CHAT_TOP_K])
        )
        system_text = system_text + "\n\n" + memory_block

    # 6) 调用 LLM
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": req.message},
        ],
    }
    if req.temperature is not None:
        payload["options"] = {"temperature": req.temperature}

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    reply = data.get("message", {}).get("content", "")

    # 7) 最后统一写入记忆：user + assistant（都只写一次）
    memres = await store_user_memory_if_needed(
        session_id=session_id,
        message=req.message,
        qvec=qvec,
        current_fact=current_fact,
    )

    #if reply.strip():
    #    avec = await ollama_embed(reply)
    #    await qdrant_upsert_chat(session_id, "assistant", reply, avec)

    return {
        "session_id": session_id,
        "model": OLLAMA_MODEL,
        "reply": reply,
        "refusal": False,
        "fact_debug": {
            "memory_llm_ok": memres["memory_llm_ok"],
            "extracted": memres["fact_extracted"],
            "kb_upsert_ok": memres["kb_upsert_ok"],
            "kb_upsert_error": memres["kb_upsert_error"],
            "skipped_reason": memres["skipped_reason"],
        },
        "sources": sources[:CHAT_TOP_K],
        "retrieval": {
            "query": req.message,
            "top_k": CHAT_TOP_K,
            "base_threshold": CHAT_BASE_THRESHOLD,
            "gap_threshold": CHAT_GAP_THRESHOLD,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "th_ok": th_ok,
            "gap_ok": gap_ok,
            "score_threshold": score_th,
            "hit_count_raw": len(hits),
            "hit_count_candidates": len(candidate_hits),
            "hit_count_used": len(sources[:CHAT_TOP_K]),
        }
    }


@app.post("/chat_mem_strict")
async def chat_mem_strict(req: ChatReq):
    """
    Legacy endpoint.
    Kept for comparison/debugging with the old fixed-threshold retrieval behavior.
    New development should happen in /chat_mem.
    """
    session_id = req.session_id or "demo"

    # NOTE:
    # This endpoint intentionally preserves the older fixed-threshold behavior.
    # Do not add new memory/fact features here unless we explicitly decide to revive it.

    # 1) embedding 用户输入
    qvec = await ollama_embed(req.message)

    # 2) 检索历史记忆
    hits = await qdrant_search_chat(session_id, qvec, CHAT_TOP_K)

    # 3) 阈值过滤
    memories = []
    kept = []  # 用于返回调试信息：point_id/score
    for h in hits:
        score = h.get("score")
        p = h.get("payload") or {}
        t = p.get("text")
        pid = h.get("id")

        if t and (score is not None) and (score >= CHAT_SCORE_THRESHOLD):
            memories.append(t)
            kept.append({"point_id": pid, "score": score})

    # 4) HARD REFUSAL：没有任何通过阈值的记忆 -> 不调用 LLM
    if not memories:
        return {
            "session_id": session_id,
            "model": OLLAMA_MODEL,
            "reply": "...",
            "refusal": True,
            "memory_used": [],
            "retrieval": {
                "query": req.message,
                "top_k": CHAT_TOP_K,
                "score_threshold": CHAT_SCORE_THRESHOLD,
                "hit_count_raw": len(hits),
                "hit_count_filtered": 0,
            }
        }

    # 5) 拼 system prompt
    system_text = req.system or "你是一个有长期记忆的助手。"
    memory_block = "以下是与当前问题相关的历史记忆（可能不完全准确）：\n" + "\n".join(
        [f"- {m}" for m in memories[:CHAT_TOP_K]]
    )
    system_text = system_text + "\n\n" + memory_block

    # 6) 调用 LLM
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": req.message},
        ],
    }
    if req.temperature is not None:
        payload["options"] = {"temperature": req.temperature}

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    reply = data.get("message", {}).get("content", "")

    # 7) 写入记忆：user + assistant（和你现在的 chat_mem 一样）
    if req.message.strip():
        await qdrant_upsert_chat(session_id, "user", req.message, qvec)

    if reply.strip():
        avec = await ollama_embed(reply)
        await qdrant_upsert_chat(session_id, "assistant", reply, avec)

    return {
        "session_id": session_id,
        "model": OLLAMA_MODEL,
        "reply": reply,
        "refusal": False,
        "memory_used": memories[:CHAT_TOP_K],
        "retrieval": {
            "query": req.message,
            "top_k": CHAT_TOP_K,
            "score_threshold": CHAT_SCORE_THRESHOLD,
            "hit_count_raw": len(hits),
            "hit_count_filtered": len(memories[:CHAT_TOP_K]),
            "kept": kept,
        }
}

@app.get("/debug/chat_mem/{session_id}")
async def debug_chat_mem(session_id: str, limit: int = 20):
    if not DEBUG_ROUTES_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    points = await qdrant_scroll_chat(session_id, limit=limit)
    items = []
    for p in points:
        payload = p.get("payload") or {}
        items.append({
            "point_id": p.get("id"),
            "role": payload.get("role"),
            "text": payload.get("text"),
            "fact_type": payload.get("fact_type"),
            "ts": payload.get("ts"),
        })
    return {
        "session_id": session_id,
        "collection": QDRANT_CHAT_COLLECTION,
        "count": len(items),
        "items": items,
    }
class DebugChatSearchReq(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)

class DebugFactOverwriteReq(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)

class DebugExtractFactReq(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)

@app.post("/debug/chat_mem_search")
async def debug_chat_mem_search(req: DebugChatSearchReq):
    if not DEBUG_ROUTES_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    top_k = req.top_k or CHAT_TOP_K
    qvec = await ollama_embed(req.query)
    hits = await qdrant_search_chat(req.session_id, qvec, top_k)

    out = []
    for h in hits:
        p = h.get("payload") or {}
        out.append({
            "score": h.get("score"),
            "point_id": h.get("id"),
            "role": p.get("role"),
            "fact_type": p.get("fact_type"),
            "text": p.get("text"),
            "ts": p.get("ts"),
        })

    return {
        "session_id": req.session_id,
        "top_k": top_k,
        "hits": out,
    }

@app.post("/debug/chat_mem_trace")
async def debug_chat_mem_trace(req: ChatReq):
    if not DEBUG_ROUTES_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    session_id = req.session_id or "demo"

    qvec = await ollama_embed(req.message)
    hits = await qdrant_search_chat(session_id, qvec, CHAT_TOP_K)

    candidate_hits = []
    for h in hits:
        p = h.get("payload") or {}
        txt = (p.get("text") or "").strip()
        role = p.get("role")

        if not txt:
            continue
        if role != "user":
            continue
        if not should_store_user_memory(txt):
            continue

        candidate_hits.append({
            "score": h.get("score"),
            "point_id": h.get("id"),
            "role": role,
            "fact_type": p.get("fact_type"),
            "text": txt,
            "ts": p.get("ts"),
        })

    current_fact = None
    current_fact_type = None
    if should_store_user_memory(req.message):
        try:
            current_fact = await extract_structured_fact(req.message)
            if current_fact is not None:
                current_fact_type = normalize_fact_type(current_fact.get("type"))
        except Exception:
            current_fact = None
            current_fact_type = None

    filtered_candidates = candidate_hits
    if current_fact_type is not None:
        filtered_candidates = []
        for h in candidate_hits:
            old_fact_type = normalize_fact_type(h.get("fact_type"))
            if old_fact_type == current_fact_type:
                continue
            filtered_candidates.append(h)

    top1 = filtered_candidates[0].get("score") if len(filtered_candidates) >= 1 else None
    top2 = filtered_candidates[1].get("score") if len(filtered_candidates) >= 2 else None

    score_th = CHAT_BASE_THRESHOLD
    gap_ok = False
    th_ok = False

    if top1 is not None:
        th_ok = top1 >= CHAT_BASE_THRESHOLD

        if top2 is None:
            gap_ok = True
        elif top1 >= 0.92:
            gap_ok = True
        else:
            gap_ok = (top1 - top2) >= CHAT_GAP_THRESHOLD

    memory_allowed = bool(th_ok and gap_ok)

    if current_fact_type is not None and len(filtered_candidates) == 0:
        memory_allowed = True

    return {
        "session_id": session_id,
        "query": req.message,
        "should_store_by_rule": should_store_user_memory(req.message),
        "current_fact": current_fact,
        "current_fact_type": current_fact_type,
        "retrieval": {
            "top_k": CHAT_TOP_K,
            "base_threshold": CHAT_BASE_THRESHOLD,
            "gap_threshold": CHAT_GAP_THRESHOLD,
            "hit_count_raw": len(hits),
            "hit_count_candidates": len(candidate_hits),
            "hit_count_after_type_filter": len(filtered_candidates),
            "top1_score": top1,
            "top2_score": top2,
            "score_threshold": score_th,
            "th_ok": th_ok,
            "gap_ok": gap_ok,
            "memory_allowed": memory_allowed,
        },
        "candidate_hits": candidate_hits,
        "filtered_candidates": filtered_candidates,
    }


@app.post("/debug/extract_fact")
async def debug_extract_fact(req: DebugExtractFactReq):
    t = (req.message or "").strip()

    prompt = build_extract_fact_prompt(t)

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    raw_content = data.get("message", {}).get("content", "").strip()
    parsed = try_parse_json_object(raw_content)

    normalized = None
    if isinstance(parsed, dict):
        required_keys = {"type", "value", "source_text"}
        if required_keys.issubset(parsed):
            normalized = {
                "type": normalize_fact_type(str(parsed.get("type")).strip()),
                "value": str(parsed.get("value")).strip(),
                "source_text": str(parsed.get("source_text")).strip(),
            }

    return {
        "message": req.message,
        "raw_content": raw_content,
        "parsed": parsed,
        "normalized": normalized,
    }

class ChatLivestreamKBReq(BaseModel):
    session_id: str | None = Field(default=None, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    system: str | None = Field(default=None, max_length=4000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

@app.post("/debug/select_livestream_fact")
async def debug_select_livestream_fact(req: ChatLivestreamKBReq):
    if not DEBUG_ROUTES_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    top_k = req.top_k if req.top_k else KB_TOP_K

    best = await select_best_livestream_fact(
        message=req.message,
        session_id=req.session_id,
        top_k=top_k,
        score_threshold=req.score_threshold,
    )

    if best is None:
        return {
            "message": req.message,
            "session_id": req.session_id,
            "selected_fact_type": None,
            "score_threshold": req.score_threshold if req.score_threshold is not None else 0.65,
            "selected_hit": None,
        }

    hit = best.get("hit") or {}
    payload = hit.get("payload") or {}

    return {
        "message": req.message,
        "session_id": req.session_id,
        "selected_fact_type": best.get("fact_type"),
        "score_threshold": best.get("score_threshold"),
        "selected_hit": {
            "point_id": hit.get("id"),
            "score": hit.get("score"),
            "type": normalize_fact_type(payload.get("type")),
            "value": payload.get("value"),
            "source_text": payload.get("source_text"),
            "source_ts": payload.get("source_ts"),
            "last_seen_ts": payload.get("last_seen_ts"),
            "freshness_days": payload.get("freshness_days"),
            "is_active": payload.get("is_active"),
            "text": payload.get("text"),
        },
    }

@app.post("/debug/fact_overwrite_trace")
async def debug_fact_overwrite_trace(req: DebugFactOverwriteReq):
    if not DEBUG_ROUTES_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    should_store = should_store_user_memory(req.message)

    current_fact = None
    current_fact_type = None
    if should_store:
        try:
            current_fact = await extract_structured_fact(req.message)
            if current_fact is not None:
                current_fact_type = normalize_fact_type(current_fact.get("type"))
        except Exception:
            current_fact = None
            current_fact_type = None

    existing_same_type_facts = []
    if current_fact_type is not None:
        hits = await kb_search(req.message, KB_TOP_K, req.session_id)

        for h in hits:
            payload = h.get("payload") or {}
            fact_type = normalize_fact_type(payload.get("type"))

            if fact_type != current_fact_type:
                continue

            if payload.get("is_active") is False:
                continue

            existing_same_type_facts.append({
                "point_id": h.get("id"),
                "score": h.get("score"),
                "type": fact_type,
                "value": payload.get("value"),
                "source_text": payload.get("source_text"),
                "source_ts": payload.get("source_ts"),
                "last_seen_ts": payload.get("last_seen_ts"),
                "freshness_days": payload.get("freshness_days"),
                "is_active": payload.get("is_active"),
                "text": payload.get("text"),
            })

    return {
        "session_id": req.session_id,
        "message": req.message,
        "should_store_by_rule": should_store,
        "current_fact": current_fact,
        "current_fact_type": current_fact_type,
        "existing_same_type_facts": existing_same_type_facts,
        "overwrite_target_count": len(existing_same_type_facts),
    }

class ChatKBReq(BaseModel):
    session_id: str | None = Field(default=None, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    system: str | None = Field(default=None, max_length=4000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    fact_type: str | None = Field(default=None, max_length=64)

class DebugKBSearchReq(BaseModel):
    session_id: str | None = Field(default=None, max_length=128)
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    fact_type: str | None = Field(default=None, max_length=64)

class KBUpsertReq(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    id: str | None = Field(default=None, max_length=128)

async def kb_search(query: str, top_k: int, session_id: str | None = None):
    # 1. 生成 embedding
    qvec = await ollama_embed(query)

    # 2. 搜索 Qdrant
    body = {
        "query": qvec,
        "limit": top_k,
        "with_payload": True
    }

    if session_id is not None:
        body["filter"] = {
            "must": [
                {"key": "session_id", "match": {"value": session_id}}
            ]
        }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points/query",
            json=body
        )
        r.raise_for_status()
        return r.json().get("result", {}).get("points", [])

async def kb_upsert(text: str, point_id: str | None = None):
    vec = await ollama_embed(text)
    if point_id is None:
        point_id = str(uuid.uuid4())

    body = {
        "points": [
            {
                "id": point_id,
                "vector": vec,
                "payload": {"text": text}
            }
        ]
    }

    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.put(
            f"{QDRANT_BASE_URL}/collections/{QDRANT_KB_COLLECTION}/points",
            json=body,
        )
        r.raise_for_status()
        return {"id": point_id, "result": r.json()}

async def select_best_livestream_fact(
    message: str,
    session_id: str | None,
    top_k: int,
    score_threshold: float | None = None,
):
    best = None
    hits = await kb_search(message, top_k, session_id)

    for h in hits:
        payload = h.get("payload") or {}
        txt = payload.get("text") or ""
        score = h.get("score")
        fact_type = normalize_fact_type(payload.get("type"))
        is_active = payload.get("is_active")
        source_ts = payload.get("source_ts")
        freshness_days = payload.get("freshness_days")

        if fact_type not in LIVESTREAM_FACT_TYPES:
            continue
        if txt.strip() == "":
            continue
        if is_active is False:
            continue
        if not is_fact_fresh(source_ts, freshness_days):
            continue

        if score_threshold is not None:
            current_threshold = score_threshold
        else:
            current_threshold = get_livestream_score_threshold(fact_type)

        if score is None or score < current_threshold:
            continue

        candidate = {
            "fact_type": fact_type,
            "score": score,
            "hit": h,
            "score_threshold": current_threshold,
        }

        if best is None or score > best["score"]:
            best = candidate

    return best

@app.post("/kb_upsert")
async def kb_upsert_api(req: KBUpsertReq):
    return await kb_upsert(req.text, req.id)

@app.post("/debug/kb_search")
async def debug_kb_search(req: DebugKBSearchReq):
    if not DEBUG_ROUTES_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")

    top_k = req.top_k or KB_TOP_K
    wanted_type = normalize_fact_type(req.fact_type)

    hits = await kb_search(req.query, top_k, req.session_id)

    out = []
    for h in hits:
        payload = h.get("payload") or {}

        fact_type = normalize_fact_type(payload.get("type"))
        if wanted_type is not None and fact_type != wanted_type:
            continue

        out.append({
            "score": h.get("score"),
            "point_id": h.get("id"),
            "kind": payload.get("kind"),
            "type": fact_type,
            "value": payload.get("value"),
            "source_text": payload.get("source_text"),
            "source_ts": payload.get("source_ts"),
            "last_seen_ts": payload.get("last_seen_ts"),
            "freshness_days": payload.get("freshness_days"),
            "is_active": payload.get("is_active"),
            "deactivated_ts": payload.get("deactivated_ts"),
            "text": payload.get("text"),
        })

    return {
        "session_id": req.session_id,
        "query": req.query,
        "top_k": top_k,
        "fact_type": wanted_type,
        "hits": out,
    }

@app.post("/chat_kb")
async def chat_kb(req: ChatKBReq):
    top_k = req.top_k if req.top_k else KB_TOP_K
    wanted_type = normalize_fact_type(req.fact_type)

    if req.score_threshold is not None:
        score_th = req.score_threshold
    elif wanted_type in LIVESTREAM_FACT_TYPES:
        score_th = get_livestream_score_threshold(wanted_type)
    else:
        score_th = KB_SCORE_THRESHOLD

    hits = await kb_search(req.message, top_k, req.session_id)
    raw_hit_count = len(hits)

    contexts = []
    sources = []  # 这里我们改名叫 sources（返回给用户的，可追溯）
    filtered_out = []
    cite_no = 0   # 引用编号：1,2,3...（只给“通过阈值”的命中编号，避免中间断号）

    for h in hits:
        point_id = h.get("id")  # ✅ 真实 Qdrant point id（uuid 或数字）
        payload = h.get("payload") or {}
        txt = payload.get("text") or ""
        score = h.get("score")
        kind = payload.get("kind")
        fact_type = normalize_fact_type(payload.get("type"))
        fact_value = payload.get("value")
        source_text = payload.get("source_text")
        source_ts = payload.get("source_ts")
        last_seen_ts = payload.get("last_seen_ts")
        freshness_days = payload.get("freshness_days")
        is_active = payload.get("is_active")

        if is_active is False:
            filtered_out.append({
                "point_id": point_id,
                "type": fact_type,
                "score": score,
                "reason": "inactive",
                "source_text": source_text,
                "is_active": is_active,
            })
            continue

        if not is_fact_fresh(source_ts, freshness_days):
            filtered_out.append({
                "point_id": point_id,
                "type": fact_type,
                "score": score,
                "reason": "stale",
                "source_text": source_text,
                "source_ts": source_ts,
                "freshness_days": freshness_days,
            })
            continue

        # 兼容旧 structured_fact：payload 里没有 value/source_text，只把 JSON 放在 text 里
        if kind == "structured_fact" and txt.strip().startswith("{"):
            try:
                legacy = json.loads(txt)
                kind = "structured_fact_legacy"
                fact_type = normalize_fact_type(legacy.get("type"))
                fact_value = legacy.get("value")
                source_text = legacy.get("source_text")
            except Exception:
                pass

        if wanted_type is not None and fact_type != wanted_type:
            filtered_out.append({
                "point_id": point_id,
                "type": fact_type,
                "score": score,
                "reason": "type_mismatch",
                "source_text": source_text,
            })
            continue

        if (txt is None) or (txt.strip() == ""):
            filtered_out.append({
                "point_id": point_id,
                "type": fact_type,
                "score": score,
                "reason": "empty_text",
                "source_text": source_text,
            })
            continue

        if (score is None) or (score < score_th):
            filtered_out.append({
                "point_id": point_id,
                "type": fact_type,
                "score": score,
                "reason": "below_score_threshold",
                "source_text": source_text,
                "score_threshold": score_th,
            })
            continue

        await kb_touch_fact(point_id)

        cite_no += 1
        contexts.append(f"[{cite_no}] {txt}")
        sources.append({
            "cite": cite_no,
            "point_id": point_id,
            "score": score,
            "collection": QDRANT_KB_COLLECTION,
            "kind": kind,
            "type": fact_type,
            "value": fact_value,
            "source_text": source_text,
            "source_ts": source_ts,
            "last_seen_ts": last_seen_ts,
            "freshness_days": freshness_days,
            "is_active": is_active,
            "text": txt
        })

    filtered_hit_count = len(sources)

    context_block = "\n".join(contexts) if contexts else "(知识库未命中)"
    # HARD REFUSAL: after threshold filtering, no usable KB context -> do not call LLM

    if not sources:
        return {
            "model": OLLAMA_MODEL,
            "reply": "...",
            "sources": [],
            "refusal": True,
            "retrieval": {
                "query": req.message,
                "top_k": top_k,
                "fact_type": wanted_type,
                "session_id": req.session_id,
                "hit_count_raw": raw_hit_count,
                "hit_count_filtered": 0,
                "filtered_out": filtered_out,
            }
        }

    system_prompt = (
        "你是一个检索增强助手。请基于知识库回答问题，并在使用知识时用 [编号] 标注引用。\n\n"
        f"【知识库】\n{context_block}\n\n"
        "【用户问题】"
    )

    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ]
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    return {
        "model": OLLAMA_MODEL,
        "reply": data.get("message", {}).get("content", ""),
        "sources": sources,
        "refusal": False,
        "retrieval": {
            "query": req.message,
            "top_k": top_k,
            "score_threshold": score_th,
            "fact_type": wanted_type,
            "session_id": req.session_id,
            "hit_count_raw": raw_hit_count,
            "hit_count_filtered": filtered_hit_count,
            "filtered_out": filtered_out,
        }
    }


@app.post("/chat_livestream_kb")
async def chat_livestream_kb(req: ChatLivestreamKBReq):
    top_k = req.top_k if req.top_k else KB_TOP_K

    best = await select_best_livestream_fact(
        message=req.message,
        session_id=req.session_id,
        top_k=top_k,
        score_threshold=req.score_threshold,
    )

    default_score_threshold = req.score_threshold if req.score_threshold is not None else 0.65

    if best is None:
        return {
            "model": OLLAMA_MODEL,
            "reply": "我目前没有检索到足够可靠的直播知识来回答这个问题。",
            "sources": [],
            "refusal": True,
            "routed_fact_type": None,
            "routing": {
                "candidate_fact_types": sorted(LIVESTREAM_FACT_TYPES),
                "selected_score": None,
                "selected_point_id": None,
            },
            "retrieval": {
                "query": req.message,
                "top_k": top_k,
                "score_threshold": default_score_threshold,
                "session_id": req.session_id,
                "selected_fact_type": None,
            },
        }

    routed_fact_type = best.get("fact_type")
    selected_hit = best.get("hit") or {}

    delegated_req = ChatKBReq(
        session_id=req.session_id,
        message=req.message,
        system=req.system,
        temperature=req.temperature,
        top_k=top_k,
        score_threshold=req.score_threshold if req.score_threshold is not None else best.get("score_threshold"),
        fact_type=routed_fact_type,
    )

    result = await chat_kb(delegated_req)
    result["refusal"] = False
    result["routed_fact_type"] = routed_fact_type
    result["routing"] = {
        "candidate_fact_types": sorted(LIVESTREAM_FACT_TYPES),
        "selected_score": best.get("score"),
        "selected_point_id": selected_hit.get("id"),
    }

    return result
