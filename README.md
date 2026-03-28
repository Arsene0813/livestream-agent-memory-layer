# From an LLM-Powered Livestream System to a Policy-Driven, Lifecycle-Aware Agent Memory Layer

An iterative project that extends an earlier LLM-powered livestream system into a policy-driven memory and retrieval layer for a more agent-like architecture.

This project began as the next iteration of an earlier livestream system. Instead of focusing only on fluent generation, it now moves toward structured fact extraction, policy-driven typed memory, entity-aware overwrite control, lifecycle-aware retrieval, and traceable decision behavior.

At its current stage, the system can handle livestream commerce knowledge such as product price, promotions, stock status, shipping policy, and product features. It supports typed fact storage, product-level entity separation, slot-based overwrite, non-fact filtering, and traceable retrieval behavior rather than relying on ungrounded generation.

## Current Scope

Currently, the project includes:

- structured fact extraction from interaction
- policy-driven typed memory through a centralized fact-policy registry
- slot-based overwrite control with soft deactivation
- entity-aware storage based on `entity_id + slot` instead of type-only overwrite
- product-level memory separation through lightweight `product_ref` extraction
- lifecycle metadata such as timestamps, freshness windows, active-state flags, and reuse metadata
- non-fact filtering so greetings and lightweight chat do not enter memory
- a hybrid memory path in which chat-vector recall is still used for conversational retrieval, while structured facts are extracted and written into the typed knowledge layer when applicable
- automatic routing across five livestream fact types:
  - product price
  - promotions
  - stock status
  - shipping policy
  - product features
- traceable retrieval outputs and inspectable fallback behavior
- a small evaluation setup for routing and fallback behavior in the current livestream knowledge layer
- A legacy strict-threshold chat-memory endpoint is still kept for comparison and debugging, but it is not the primary interaction path of the current memory layer.

## Why This Project

My earlier livestream system could already support product explanation and customer-facing dialogue, but I gradually realized that fluent generation alone was not enough to make such a system reliable in practice. The main limitation was not language generation itself, but the lack of a clear way to manage memory over time: newer information had no principled mechanism for replacing older information, outdated content could still remain retrievable, and it was often difficult to explain why a particular past item of information had been used in a response.

This project emerged from that limitation. Its purpose is to make an LLM-based interaction layer more dependable and easier to inspect by introducing structured memory, retrieval gating, overwrite logic, and lifecycle-aware updates.

## System Evolution

- **Local LLM backend**  
  The project began as a simple local model-serving setup for conversational use in a livestream-oriented setting. The initial goal was to make local generation work reliably enough to support product explanation and customer-facing interaction.

- **Raw vector chat memory**  
  The next step was to add vector-based recall so that past exchanges could be retrieved instead of being discarded entirely. This made the system more context-aware, but it also exposed an important limitation: retrieval alone did not make memory reliable. Past content could be surfaced based on similarity without any clear standard for whether it should still matter.

- **Retrieval gating and traceability**  
  To address that limitation, I introduced retrieval gating and fallback or refusal conditions so that memory would only be used when there was enough evidence that it was genuinely relevant. I also added traceable evidence in retrieval outputs so that memory usage could be inspected afterward rather than remaining hidden inside model behavior.

- **Structured facts and typed memory**  
  Once retrieval became more controlled, the next issue became clearer: storing past text was still not the same as representing knowledge in a stable form. The system therefore moved from unstructured recall toward structured fact extraction and typed memory, allowing different categories of information to be stored and handled more consistently.

- **Update policies and overwrite control**  
  After typed memory was in place, I added overwrite logic so that newer facts could replace older active knowledge in a controlled way rather than letting conflicting information accumulate indefinitely. Older facts were preserved through soft deactivation instead of being deleted without trace, which made memory updates easier to inspect and reason about.

- **Lifecycle-aware retrieval**  
  The project then evolved into a lifecycle-aware memory layer. Stored facts were no longer treated as timeless entries, but as knowledge objects with timestamps, active-state flags, freshness windows, and reuse metadata. This allowed retrieval to prefer currently valid knowledge while filtering out inactive or stale facts when appropriate.

- **Livestream commerce knowledge routing**  
  The next iteration extended the same memory architecture into livestream commerce knowledge. Structured facts such as product price, promotions, stock status, shipping policy, and product features could be written into typed memory and retrieved later through the same lifecycle-aware logic. On top of that, I added automatic fact-type routing so that the system could handle livestream questions without requiring the caller to manually specify which category of knowledge should be searched.

- **Policy-driven memory behavior**  
  As the number of fact types increased, hard-coded rules for overwrite, freshness, routing thresholds, scope, and storage semantics became difficult to maintain. To address this, the system moved toward a centralized fact-policy registry, making storage and retrieval behavior easier to extend and reason about.

- **Product-level entity separation**  
  Once livestream facts became more structured, it was no longer sufficient to store all product knowledge under a single default product context. The system therefore introduced lightweight product reference extraction so that facts could be attached to specific product entities when possible, while still falling back to a default product scope when no explicit reference was available.

- **Single-call extraction and non-fact filtering**  
  The earlier pipeline separated memory gating from fact extraction, which increased latency and could produce inconsistent behavior. The current version consolidates this flow so that extraction itself determines whether a message yields a storable fact, while obvious non-fact messages such as greetings are filtered out before memory write.

- **Explainable fallback and small-scale evaluation**  
  At the current stage, the system can not only answer from stored knowledge, but also fall back safely when available knowledge is stale, inactive, unsupported, or insufficiently reliable. Retrieval outputs expose routing decisions and filtered reasons, making failure cases easier to inspect rather than leaving them ambiguous. To make the current behavior easier to verify, I also added a small evaluation setup covering successful routing cases as well as fallback cases such as stale or unsupported queries.

## Current Capabilities

Currently, the system can:

- extract structured facts from interaction instead of relying only on raw conversational history
- store facts through a centralized fact-policy registry rather than scattered type-specific rules
- attach each fact to `scope`, `slot`, and `entity_id` for more explicit storage semantics
- apply overwrite at the `entity_id + slot` level for single-value facts such as product price
- preserve multi-value facts such as product features without collapsing them into a single active value
- preserve older superseded records through soft deactivation instead of hard deletion
- attach lifecycle metadata such as timestamps, freshness windows, active-state flags, and last-seen signals
- separate product memory across entities through lightweight `product_ref` extraction
- filter non-fact messages such as greetings from entering memory
- return traceable supporting memory items instead of relying on hidden recall
- refuse or fall back safely when no sufficiently reliable structured memory is available
- support a small evaluation setup covering routing and fallback behavior in the current livestream knowledge layer

## Example Workflows

### 1. Structured fact extraction from interaction

A livestream or product-related input can be converted into a structured fact instead of remaining only as raw chat history. For example, a message such as `A款价格是99元` can be stored as a typed fact like `product_price = 99元`, making it available for later retrieval and update handling.

### 2. Traceable typed retrieval

When the system answers a follow-up query, it can retrieve the relevant stored fact and surface the supporting memory entry rather than relying on hidden recall. For example, after storing `A款价格是99元`, a later query such as `A款多少钱` can be answered with traceable evidence showing which fact was used.

### 3. Overwrite and soft deactivation

If newer information of the same type appears, the system can update the active fact without discarding the older record entirely. For example, if `A款价格是99元` is later followed by `A款价格是89元`, the newer price can become active while the older one is preserved through soft deactivation for traceability.

### 4. Product-level separation

The system can separate facts across products instead of forcing all livestream knowledge into a single default product context. For example, `A款价格是99元` and `B款价格是199元` can be stored under different product entities, so that later retrieval and overwrite behavior remains product-specific.

### 5. Livestream commerce query routing

For livestream commerce queries, the system does not rely on a separate intent classifier. Instead, it performs semantic retrieval over eligible knowledge candidates, applies fact-type-specific validity checks and routing thresholds, and uses the highest-scoring eligible fact type as the routed type.

In other words, livestream fact-type routing is implemented through retrieval-time score competition rather than a standalone classification step.

Current examples include:

- `这款多少钱` → product price
- `今天有什么优惠` → promotions
- `现在有货吗` → stock status
- `多久能发货` → shipping policy
- `这款有什么特点` → product features

### 6. Lifecycle-aware fallback and refusal

The system can also fall back or refuse when stored knowledge should no longer be used. This includes cases where a fact is stale, inactive, unsupported, or otherwise not reliable enough to support an answer. For example, an outdated promotion may still remain in storage for history tracking, but it can be filtered out or refused at answer time rather than being treated as current knowledge.

## Evaluation

The repository includes a small evaluation setup for the current livestream knowledge layer.

The evaluation is intentionally limited in scope. Its purpose is not to claim broad benchmark coverage, but to verify that the current routing and fallback behavior matches the intended design for the livestream scenarios implemented so far.

At the moment, the evaluation covers seven scenarios in total:

- five successful routing cases:
  - product price
  - promotions
  - stock status
  - shipping policy
  - product features
- two fallback or refusal cases:
  - stale promotion knowledge
  - unsupported or no-match queries

The evaluation files are:

- `eval_livestream_cases.json` - evaluation cases and expected outcomes
- `eval_livestream.py` - a lightweight script that runs the cases against the local API and reports pass/fail results

A typical run checks whether the system:

- routes a query to the expected fact type
- produces a non-refusal answer when reliable knowledge is available
- falls back or refuses when knowledge is stale or insufficiently reliable
- returns responses that contain the expected key information

## Running the Project

This project is intended to run locally with Docker Compose.

Start the services with:

```bash
docker compose up -d
```

The local setup includes the API service, Ollama for local model inference and embeddings, and Qdrant for vector storage and typed memory retrieval.

You can check that the API is running with:

```bash
curl http://127.0.0.1:8000/health
```

If you update the API code, rebuild the API service with:

```bash
docker compose up -d --build api
```

## Running the Evaluation

The evaluation script is designed to run against the local API at `http://127.0.0.1:8000`.

Assuming the services are already running, you can copy the evaluation files into the API container and execute the script with:

```bash
docker compose cp eval_livestream.py api:/app/eval_livestream.py
docker compose cp eval_livestream_cases.json api:/app/eval_livestream_cases.json
docker compose exec api python /app/eval_livestream.py
```

## Repository Structure

- `api/main.py` - main FastAPI application, including memory logic, retrieval control, routing, and debug endpoints
- `api/Dockerfile` - Docker image definition for the API service
- `api/requirements.txt` - Python dependencies for the API service
- `docker-compose.yml` - local multi-service setup for the API and supporting services
- `eval_livestream_cases.json` - small evaluation set for livestream routing and fallback/refusal cases
- `eval_livestream.py` - simple evaluation runner for the livestream cases
- `README.md` - project overview, current capabilities, evaluation notes, and usage instructions

## Notes

This repository reflects an ongoing iteration rather than a finished product. The project is still intentionally compact, and a number of design choices remain visible at the application level for ease of inspection and experimentation.

At the current stage, the implementation still keeps much of the logic in a single main service file. This makes iteration easier during development and keeps policy, overwrite, routing, and retrieval behavior easy to inspect, even though future cleanup would likely separate routing, memory policy, evaluation, and storage logic more clearly as the project grows.

## Next Steps

- expand the evaluation set beyond the current small livestream cases
- make fact-policy configuration easier to extend and maintain
- cover overwrite, multi-value coexistence, product-level separation, and non-fact filtering more explicitly in evaluation
- continue separating extraction, routing, storage, and evaluation logic out of the current main service file as the project grows
- explore richer profile-level and cross-session memory on top of the current typed fact layer



