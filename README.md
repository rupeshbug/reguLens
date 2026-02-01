# SECPolicyLens
> Built as a portfolio project to demonstrate production-aware RAG system design for regulatory and policy analysis.

SECPolicyLens is a retrieval-augmented generation (RAG) system focused on U.S. SEC Climate‑Related Disclosure rules. It is designed to answer regulatory and investor-facing questions using only official SEC source documents, with a strong emphasis on grounding, traceability, and hallucination prevention.

A core capability of SECPolicyLens is its ability to analyze, explain, and contrast different versions of the same regulation, with particular emphasis on comparing:
- The **2022 SEC Climate‑Related Disclosure Proposed Rule**
- The **2024 SEC Climate‑Related Disclosure Final Rule**

The system prioritizes accuracy, traceability, and compliance-safe reasoning, with explicit safeguards to prevent hallucination or unsupported interpretation.

---

## Live Demo

**Web App**: https://sec-policy-lens.vercel.app/  

> Note: The live deployment uses the fast retrieval path optimized for latency and free-tier infrastructure of Render.  
> Full hybrid retrieval is available in offline / evaluation mode.

---

## What SECPolicyLens Does

- Retrieves relevant regulatory passages using **metadata-aware vector search**
- Supports **version-filtered retrieval** (2022 vs 2024)
- Generates answers strictly grounded in retrieved SEC text
- Prefers **final rules over proposed rules** when both are available
- Explicitly avoids external knowledge or unsupported interpretation
- Returns structured source metadata (document, version, section)

SECPolicyLens is suitable for:

- Regulatory interpretation and compliance analysis  
- Comparison between proposed and finalized SEC rules  
- Investor-facing regulatory explanations  
- Legal and policy research workflows  

---

## Retrieval Architecture Overview

SECPolicyLens implements **two retrieval paths**, intentionally separated to balance **latency, cost, and retrieval quality**.

---

### 1. Fast Retrieval Path (Production Default)

Used in the deployed API on free-tier infrastructure.

- **Dense-only semantic retrieval**
- Sentence-Transformer embeddings
- Metadata-based filtering (rule version, section)
- No sparse retrieval, no cross-encoder reranking
- Optimized for low latency and API responsiveness

This configuration ensures:
- Stable performance under strict timeout limits
- Predictable latency for real-time API usage
- Safe deployment on free-tier hosting (e.g. Render)

---

### 2. Full Retrieval Path (Offline / Evaluation)

Used for experiments, ablations, and retrieval quality validation.

- **Hybrid retrieval** (dense + sparse)
- SPLADE sparse vectors for lexical precision
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking for semantic refinement
- Query decomposition for improved recall

This configuration prioritizes **retrieval quality over latency** and is intentionally **not used in the production API** due to computational cost.

---

## Key Capabilities

- **Version-Aware Retrieval**  
  Every chunk is indexed with document version, section, and title metadata, enabling strict version filtering and comparison.

- **Chunk-Based Indexing**  
  Regulatory text is chunked into semantically coherent units to balance retrieval precision and contextual completeness.

- **Hybrid Retrieval (Evaluation Mode)**  
  Dense embeddings capture semantic similarity, while sparse vectors preserve exact regulatory phrasing.

- **RRF Fusion**  
  Reciprocal Rank Fusion combines multiple retrieval signals into a robust candidate set.

- **Cross-Encoder Reranking (Evaluation Mode)**  
  A transformer-based reranker refines candidate passages to maximize contextual relevance.

- **Strict Grounding**  
  The language model is constrained to retrieved SEC text only. External knowledge is explicitly disallowed.

- **Source Awareness**  
  Every answer includes document, version, and section metadata for traceability.

---

## Models and Infrastructure

- **Dense embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Sparse retrieval (evaluation)**: `naver/splade-cocondenser-ensembledistil`
- **Reranker (evaluation)**: Sentence-Transformers cross-encoder
- **Vector database**: Qdrant
- **LLM**: Groq-hosted `llama-3.3-70b-versatile`
- **API framework**: FastAPI

---

## Retrieval Ablation and Validation

SECPolicyLens includes a simple but effective ablation test to validate **metadata-aware retrieval and version control**.

The same regulatory question is executed under three settings:

1. No version filter (mixed retrieval)
2. Retrieval restricted to the 2024 Final Rule
3. Retrieval restricted to the 2022 Proposed Rule

Observed behavior:

- Unfiltered retrieval produces blended answers spanning both rule versions
- Version-filtered retrieval strictly limits both sources and generated answers
- 2024-only responses reflect finalized regulatory intent
- 2022-only responses capture proposal-stage rationale and investor-protection arguments

This confirms that document metadata is actively used during retrieval and materially influences downstream generation.

---

### Query Decomposition Ablation

SECPolicyLens also evaluates the impact of query decomposition on retrieval quality.

The same question is executed with:

- Query decomposition disabled (single query)
- Query decomposition enabled (LLM-generated sub-questions)

Results:

- Without decomposition, answers may be conservative or incomplete when relevant information is distributed across sections
- With decomposition enabled, retrieval recall improves while remaining grounded
- A global reranking step ensures alignment with the original user intent

This validates that query decomposition improves recall **without compromising regulatory precision**.

---

## Design Philosophy

SECPolicyLens is intentionally **not a generic chatbot**.

It is designed as:
- A **controlled regulatory reasoning system**
- A demonstration of **production-aware RAG design**
- A balance between **retrieval quality and real-world deployment constraints**

The separation between **fast production retrieval** and **full evaluation retrieval** reflects real industry trade-offs rather than idealized architectures.
