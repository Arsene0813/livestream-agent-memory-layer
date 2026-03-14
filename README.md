# From an LLM-Powered Livestream System to a Lifecycle-Aware Agent Memory Layer

This project is the next iteration of my earlier LLM-powered livestream system. It evolves a fluent conversational layer into a more agent-like architecture by adding structured memory, retrieval control, update policies, and lifecycle-aware knowledge management.

## Why This Project

My earlier livestream system could already support product explanation and customer-facing dialogue, but I gradually realized that fluent generation alone was not enough to make such a system reliable in practice. The key limitation was not language generation itself, but the lack of a principled way to manage memory over time: newer information had no clear mechanism for replacing older information, outdated content could still remain retrievable, and it was often difficult to explain why a particular past item of information had been used in a response.

This project grew out of that limitation. Its purpose is to make an LLM-powered interaction layer more dependable, more explainable, and better aligned with real-world operational use by introducing structured memory, retrieval gating, overwrite logic, and lifecycle-aware updates.

## System Evolution

- **Local LLM backend:** started with a simple local model-serving setup that supported conversational use in a livestream-oriented setting.
- **Raw vector chat memory:** added vector-based recall so that past exchanges could be retrieved instead of being discarded entirely.
- **Retrieval gating and traceability:** introduced selective retrieval, refusal conditions, and traceable evidence so that memory would only be used when sufficiently relevant and could be inspected afterward.
- **Structured facts and typed memory:** moved from unstructured text recall toward structured fact extraction and typed memory so that different categories of information could be handled more consistently.
- **Update policies and overwrite control:** added mechanisms for replacing older same-type facts with newer ones in a controlled way rather than allowing conflicting information to accumulate.
- **Lifecycle-aware retrieval:** extended memory into a managed knowledge layer with timestamps, active-state signals, freshness windows, and reuse tracking.

- ## Current Capabilities

- Extracts structured facts from interactions instead of relying only on unstructured text recall.
- Organizes stored knowledge into typed memory so that different categories of information can be handled more consistently.
- Applies update rules when newer information supersedes older same-type facts.
- Distinguishes between active and inactive facts during retrieval rather than treating all stored memory as equally valid.
- Attaches lifecycle metadata such as timestamps, freshness signals, and reuse history to stored knowledge.
- Surfaces traceable retrieval evidence so that memory usage can be inspected rather than remaining hidden inside model behavior.

- ## Example Workflows

### 1. Structured fact extraction
A user message such as `我住在北京` can be converted from plain conversational text into a structured fact like `location = 北京`, which is then stored as typed memory rather than remaining only as raw chat history.

### 2. Traceable retrieval
When the system answers a follow-up query such as `我住在哪`, it can retrieve the relevant stored fact and surface traceable evidence showing which memory item influenced the response.

### 3. Overwrite and lifecycle handling
If a newer fact of the same type appears, such as `我住在上海` followed later by `我住在北京`, the system can update the active fact in a controlled way, preserve the older one through soft deactivation, and keep lifecycle metadata to reflect how knowledge changes over time.

## Next Steps

- Add richer evaluation workflows to compare retrieval behavior, overwrite decisions, and lifecycle-aware memory usage across different scenarios.
- Adapt the same memory architecture to livestream commerce knowledge, such as product facts, promotional updates, and operational information that changes over time.
- Explore higher-level profile synthesis built on top of structured memory, while keeping retrieval traceable and update policies explicit.
