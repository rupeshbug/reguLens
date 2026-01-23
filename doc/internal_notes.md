# ReguLens – Internal Project Notes

This document captures the **story of ReguLens**: what problem we set out to solve, the concrete engineering decisions we made along the way, the tradeoffs we accepted, and how the system is evolving. It is meant both as a long‑term memory aid and as a narrative to clearly explain the project end‑to‑end.

---

## 1. Problem Definition

Regulatory documents such as SEC rules are fundamentally different from typical knowledge sources. They are long, highly structured, legally precise, and versioned over time. Small wording changes can materially affect interpretation.

In early experiments, generic LLMs showed predictable failure modes:
- Hallucinating regulatory intent
- Mixing proposed rules with finalized rules
- Answering confidently even when evidence was missing

The core problem, therefore, was **not language generation**, but **trustworthy retrieval and grounding**.

**Goal:** Build a system that answers regulatory questions *only* from official SEC climate‑related disclosure documents, with strong guarantees around correctness, traceability, and refusal when evidence is insufficient.

---

## 2. Why Retrieval‑Augmented Generation (RAG)

Pure LLM answers are unacceptable for compliance and regulatory use cases. Even strong models tend to generalize, interpolate, or guess when faced with incomplete information.

RAG was chosen because it:
- Grounds every answer in source text
- Makes the system evidence‑limited by design
- Allows explicit detection of missing context

---

## 3. Ingestion Strategy: Chunking & Metadata (What Happens Before Search)

A major early challenge was realizing that **retrieval quality is largely determined at ingestion time**.

### Structural Normalization & Cross-Referencing
The source documents frequently reference other parts of the rule using informal citations (e.g., “see infra Section I.C.2”) without machine-readable structure.

To address this, ReguLens introduces a normalized structural layer during ingestion:

- A custom structure.json maps conceptual sections and subsections
- Cross-references are resolved against this normalized structure
- Chunks are associated with logical sections, not just raw text offsets

This enables:

- More reliable retrieval across conceptually linked passages
- Future version-aware comparison (2022 vs 2024) at the section level
- Clearer traceability when explaining answers

### Chunking
SEC climate disclosure rules are not cleanly structured documents.
They contain long narrative paragraphs, conceptual sections, frequent cross-references (e.g., “See infra Section I.C.2”), and inconsistent structural markers.

Because of this, ReguLens does not rely on naive fixed-size chunking or literal section headers.

Instead, the ingestion pipeline performs section-aware semantic chunking, built on top of:

- Paragraph-aware chunk boundaries to preserve legal and conceptual coherence
- A soft maximum length constraint (≈1500 characters)
- A tolerance window (≈200 characters) to avoid unnecessary fragmentation when a paragraph slightly exceeds the limit

This approach:

- Preserves regulatory reasoning ensures passages remain interpretable in isolation
- Avoids breaking sentences or legal arguments mid-thought
- Produces chunks that align with how the SEC argues, not how the PDF is formatted

The result is fewer but higher-quality chunks, which directly improves retrieval precision and downstream generation.

### Metadata
Each chunk is enriched with structured metadata, including:
- Rule version (2022 proposed vs 2024 final)
- Section / subsection identifiers
- Document source

This metadata enables:
- Fast filtering during retrieval
- Version‑aware prioritization
- Cleaner, more auditable answers

These ingestion decisions significantly improved both **accuracy and retrieval latency**.

---

## 4. Retrieval Stage: Why Hybrid Search

No single retrieval method was sufficient for SEC‑style text.

### Dense Retrieval (Embeddings)
Dense search captures semantic similarity and handles paraphrased or conceptual questions well. However, it struggles with exact legal phrasing and specific terminology.

### Sparse Retrieval (SPLADE)
Sparse retrieval preserves regulatory and legal language, making it strong for exact keyword and phrase matching. Its weakness lies in semantic generalization.

### Why Hybrid
Real regulatory questions often mix both styles, for example:
- Conceptual intent ("why does the SEC require…")
- Precise legal wording ("disclosure requirements", "material risk")

Using both dense and sparse retrieval provides robustness across these dimensions.

---

## 5. Why Reciprocal Rank Fusion (RRF)

Once multiple retrievers are involved, their results must be combined reliably.

RRF was chosen because it:
- Does not depend on score calibration
- Is robust when retrievers disagree
- Is simple, transparent, and production‑proven

RRF allows ReguLens to merge dense and sparse results into a single high‑quality candidate set without introducing fragile heuristics.

---

## 6. Reranking: Why Cross‑Encoders

Initial retrieval only identifies *candidates*. It does not guarantee that the top results are the most answer‑relevant.

A cross‑encoder reranker is used to:
- Jointly score query–passage pairs
- Push truly relevant chunks to the top
- Filter out semantically close but contextually irrelevant text

This reranking step proved critical before passing context to the LLM and noticeably improved answer coherence.

---

## 7. Prompt Design Philosophy

Prompting is treated as a **control mechanism**, not a creativity tool.

The system prompt enforces:
- Strict domain scope (SEC climate rules only)
- Preference for final rules over proposed rules
- Explicit refusal when evidence is insufficient

The user‑level prompt:
- Injects retrieved passages verbatim
- Forbids external knowledge
- Encourages synthesis *only when supported*

The model is intentionally constrained to behave more like a compliance analyst than a conversational assistant.

---

## 8. Key Challenges Faced

Some challenges that materially influenced the design:
- Mixing of proposed vs final rule content in early retrieval
- Over‑retrieval of semantically similar but irrelevant sections
- LLM tendency to over‑generalize when context was thin

These issues directly led to:
- Metadata‑aware filtering
- Strong reranking
- Explicit refusal logic

---

## 9. Key Takeaway

ReguLens is not a demo RAG system.

It is a **compliance‑grade retrieval and reasoning pipeline** built with:
- Intentional constraints
- Auditable logic
- Clear separation of ingestion, retrieval, reranking, and generation

In ReguLens, refusing to answer is not a weakness — it is a core feature.

### Ablation Test: Version-Aware Retrieval

To evaluate the effect of metadata filtering on retrieval and answer generation, we conducted an ablation test using the same query under three conditions: (1) no version filter, (2) retrieval restricted to the 2024 final rule, and (3) retrieval restricted to the 2022 proposed rule. The goal was to verify whether constraining retrieval by document version causally affects the evidence retrieved and the resulting answer.

The results show clear behavioral differences across conditions. Without filtering, the system produces a blended response drawing from both the proposed and final rule rationales. When restricted to the 2024 final rule, the answer reflects the SEC’s finalized position, emphasizing investor pricing efficiency, materiality, and final regulatory justifications. When restricted to the 2022 proposed rule, the answer shifts toward consultation-stage reasoning, focusing on inconsistent voluntary disclosures and the need for investor protection through standardized reporting.

In the final run, retrieved sources fully aligned with the specified version constraints, confirming that metadata-aware retrieval is correctly enforced. This demonstrates that retrieval configuration meaningfully influences both evidence selection and model reasoning, validating the effectiveness of version-aware RAG and highlighting the importance of retriever design in regulatory QA systems.
