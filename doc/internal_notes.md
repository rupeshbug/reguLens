# ReguLens – Internal Project Notes

This document captures the **story of ReguLens**: what problem we set out to solve, the concrete engineering decisions we made along the way, the tradeoffs we accepted, and how the system evolved from an evaluation-heavy RAG system into a production-deployable service. It is intended both as a long-term memory aid and as a narrative to clearly explain the project end-to-end to technical interviewers.

---

## 1. Problem Definition

Regulatory documents such as SEC rules are fundamentally different from typical knowledge sources. They are long, highly structured, legally precise, and versioned over time. Small wording changes can materially affect interpretation and compliance obligations.

In early experiments, generic LLMs showed predictable failure modes:
- Hallucinating regulatory intent
- Mixing proposed rules with finalized rules
- Answering confidently even when evidence was missing

The core problem, therefore, was **not language generation**, but **trustworthy retrieval and grounding**.

**Goal:** Build a system that answers regulatory questions *only* from official SEC climate-related disclosure documents, with strong guarantees around correctness, traceability, version awareness, and refusal when evidence is insufficient.

---

## 2. Why Retrieval-Augmented Generation (RAG)

Pure LLM answers are unacceptable for compliance and regulatory use cases. Even strong models tend to generalize, interpolate, or guess when faced with incomplete or ambiguous information.

RAG was chosen because it:
- Grounds every answer in source text
- Makes the system evidence-limited by design
- Allows explicit detection of missing or insufficient context
- Enables traceability to specific regulatory passages

In ReguLens, RAG is treated as a **control and safety mechanism**, not a convenience layer.

---

## 3. Ingestion Strategy: Chunking & Metadata (What Happens Before Search)

A major early realization was that **retrieval quality is largely determined at ingestion time**. Poor chunking or missing metadata cannot be “fixed” downstream.

### Structural Normalization & Cross-Referencing

SEC rule documents frequently reference other parts of the regulation using informal citations (e.g., “see infra Section I.C.2”) without machine-readable structure.

To address this, ReguLens introduces a normalized structural layer during ingestion:

- A custom `structure.json` maps conceptual sections and subsections
- Cross-references are resolved against this normalized structure
- Chunks are associated with logical regulatory sections, not just raw text offsets

This enables:
- Reliable retrieval across conceptually linked passages
- Version-aware comparison (2022 vs 2024) at the section level
- Clear traceability when explaining answers

### Chunking Strategy

SEC climate disclosure rules are not cleanly structured documents. They contain long narrative paragraphs, conceptual reasoning, frequent cross-references, and inconsistent structural markers.

ReguLens therefore avoids naive fixed-size chunking.

Instead, the ingestion pipeline uses **section-aware semantic chunking**, built on top of:
- Paragraph-aligned boundaries to preserve legal reasoning
- A soft maximum length (≈1500 characters)
- A tolerance window (≈200 characters) to avoid breaking paragraphs unnecessarily

This approach:
- Preserves regulatory arguments as interpretable units
- Avoids splitting legal reasoning mid-thought
- Produces fewer but higher-quality chunks

This design significantly improved retrieval precision and reduced downstream noise.

### Metadata

Each chunk is enriched with structured metadata:
- Rule version (2022 Proposed vs 2024 Final)
- Section / subsection identifiers
- Document source

Metadata is not decorative — it is **actively used at query time** for filtering, prioritization, and ablation experiments.

---

## 4. Retrieval Stage: Dense vs Sparse vs Hybrid

No single retrieval method is sufficient for regulatory text.

### Dense Retrieval (Embeddings)

Dense retrieval uses vector embeddings to capture semantic similarity.

**Strengths:**
- Handles paraphrased or conceptual questions well
- Strong recall for “why” and “intent” questions

**Weaknesses:**
- Can miss exact legal phrasing
- May retrieve semantically similar but legally irrelevant text

### Sparse Retrieval (SPLADE)

**SPLADE** (Sparse Lexical and Expansion Model) is a neural sparse retriever that expands queries into weighted lexical terms while retaining sparse representations.

**Strengths:**
- Preserves exact legal terminology
- Strong performance on keyword-heavy regulatory queries

**Weaknesses:**
- Limited semantic generalization
- Higher computational and latency cost

### Hybrid Retrieval

Hybrid retrieval combines dense and sparse retrievers to capture both:
- Conceptual meaning
- Exact regulatory language

This configuration performs best during **offline evaluation and ablation**, but introduces latency and infrastructure overhead.

---

## 5. Reciprocal Rank Fusion (RRF)

When multiple retrievers are used, their results must be merged.

**RRF (Reciprocal Rank Fusion)** combines ranked lists by summing the reciprocal of each document’s rank across retrievers.

Why RRF:
- Does not rely on score calibration
- Is robust when retrievers disagree
- Simple, interpretable, and widely used in IR systems

RRF allows ReguLens to merge dense and sparse retrieval results into a stable candidate set without fragile heuristics.

---

## 6. Reranking with Cross-Encoders

Initial retrieval produces *candidates*, not guaranteed answer-relevant passages.

A **cross-encoder reranker** jointly encodes the query and passage to produce a relevance score.

This reranking step:
- Pushes truly relevant chunks to the top
- Filters semantically close but contextually irrelevant text
- Significantly improves answer coherence

This step is computationally expensive and therefore not always enabled in production.

---

## 7. Two-Path Architecture: Evaluation vs Production

A key design decision was explicitly separating **evaluation correctness** from **production latency**.

### Full Retrieval Path (Evaluation / Offline)

Used for:
- Ablation studies
- Architecture validation
- Retrieval quality analysis

Includes:
- Dense + SPLADE sparse retrieval
- RRF fusion
- Query decomposition (multi-query expansion)
- Cross-encoder reranking

This path maximizes recall and precision but is latency-heavy.

### Fast Retrieval Path (Production)

For deployment on free-tier infrastructure (e.g., Render), ReguLens uses a **dense-only retrieval configuration**:

- Single dense retriever
- Metadata-aware filtering
- No query decomposition
- No cross-encoder reranking

This reduces:
- Cold-start latency
- Memory usage
- External model calls

The fast path is exposed as the default production API, while the full path remains available for internal evaluation.

This explicit split reflects real-world engineering tradeoffs rather than theoretical optimality.

---

## 8. API & Deployment Design

ReguLens exposes a single production entry point that supports both retrieval modes:
mode = "fast" | "full"

- `fast`: production-safe, low-latency
- `full`: evaluation-grade, correctness-maximized

The service is deployed on Render free-tier infrastructure. Observed behavior:
- First request experiences cold-start latency (~2 minutes)
- Subsequent requests stabilize to ~1–2 seconds

This confirms correct system behavior under constrained infrastructure.

---

## 9. Prompt Design Philosophy

Prompting is treated as a **control surface**, not a creativity tool.

The system prompt enforces:
- Strict domain scope (SEC climate rules only)
- Preference for final rules over proposed rules
- Explicit refusal when evidence is insufficient

The user prompt:
- Injects retrieved passages verbatim
- Forbids external knowledge
- Allows synthesis only when supported

The model is constrained to behave like a **compliance analyst**, not a conversational chatbot.

---

## 10. Key Challenges Faced

Challenges that materially influenced design:
- Mixing of proposed and final rule content
- Over-retrieval of semantically similar sections
- LLM overconfidence under thin context
- Latency constraints on free-tier deployment

These directly led to:
- Metadata-aware filtering
- Explicit fast vs full retrieval paths
- Strong refusal logic

---

## 11. Key Takeaway

ReguLens is not a demo RAG chatbot.

It is a **compliance-grade retrieval and reasoning system** designed with:
- Explicit constraints
- Auditable decisions
- Clear separation between evaluation and production

In ReguLens, refusing to answer is not a failure — it is a core feature.

---

## Ablation Test: Version-Aware Retrieval

To validate metadata-aware retrieval, the same query was executed under three conditions:
1. No version filter
2. 2024 Final Rule only
3. 2022 Proposed Rule only

Results showed:
- Unfiltered retrieval produced blended answers
- Version-restricted retrieval strictly constrained evidence
- Generated answers reflected the correct regulatory intent for each version

This confirms that metadata filtering causally affects both retrieval and reasoning.

**Note:**  
When query decomposition is enabled, ReguLens applies a final global cross-encoder reranking step to ensure that the final context aligns with the original user intent rather than individual sub-queries.

---

## How I’d Explain This in 90 Seconds

ReguLens is a compliance-focused RAG system built specifically for SEC climate-related disclosure rules.

The key insight is that regulatory QA fails not because of generation, but because of retrieval and grounding. Generic LLMs hallucinate, mix rule versions, and answer even when evidence is missing — all unacceptable for compliance use cases.

I designed ReguLens around that constraint. During ingestion, documents are semantically chunked and enriched with strict metadata like rule version and section IDs. Retrieval is metadata-aware by design.

For evaluation, the system uses hybrid retrieval — dense embeddings plus SPLADE sparse search — fused with Reciprocal Rank Fusion and refined using a cross-encoder reranker. This maximizes correctness and lets me run ablation studies, like comparing 2022 proposed vs 2024 final rules.

For production, I explicitly separated the path. On free-tier infrastructure, ReguLens runs a dense-only, low-latency retrieval pipeline with the same metadata guarantees, trading off reranking for speed and stability.

The LLM is tightly constrained: it can only answer from retrieved context, prefers final rules, and explicitly refuses when evidence is insufficient.

The result is not a demo chatbot, but a production-minded, auditable regulatory reasoning system where refusal is a feature, not a bug.
