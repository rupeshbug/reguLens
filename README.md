# ReguLens

ReguLens is a retrieval-augmented generation (RAG) system focused on U.S. SEC Climate‑Related Disclosure rules. It is designed to answer regulatory and investor‑facing questions using **only** official SEC source documents, with strong focus to prevent hallucination.

The system currently supports:
- The **2022 SEC Climate‑Related Disclosure Proposed Rule**
- The **2024 SEC Climate‑Related Disclosure Final Rule**

ReguLens prioritizes accuracy, traceability, and compliance‑safe reasoning over generic conversational behavior.

---

## What ReguLens Does

- Retrieves relevant regulatory passages using **hybrid search** (dense + sparse)
- Reranks results using a **cross‑encoder** for semantic precision
- Generates answers using a large language model constrained by retrieved context
- Prefers **final rules over proposed rules** when both are available
- Explicitly refuses to answer when context is insufficient

This makes ReguLens suitable for:
- Regulatory analysis
- Compliance and legal research
- Investor disclosures and explanations
- Policy comparison and interpretation (within source limits)

---

## Key Capabilities

- **Hybrid Retrieval**: Combines semantic embeddings with lexical matching
- **RRF Fusion**: Robustly merges multiple retrievers
- **Cross‑Encoder Reranking**: Improves answer relevance and coherence
- **Strict Grounding**: No external knowledge or assumptions
- **Source Awareness**: Tracks document version and section metadata

---

## Models Used

- Dense embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Sparse retrieval: `naver/splade-cocondenser-ensembledistil`
- Reranker: Sentence‑Transformers cross‑encoder
- Vector database: Qdrant
- LLM: Groq‑hosted `llama‑3.3‑70b‑versatile`

---

## Design Philosophy

ReguLens intentionally avoids heavy RAG orchestration frameworks in favor of:
- Transparency
- Debuggability
- Full control over retrieval and prompting

This makes the system easier to audit, extend, and productionize.

---
