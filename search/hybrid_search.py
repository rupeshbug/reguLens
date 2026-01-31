import os

USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"

if USE_LOCAL_MODELS:
    import torch
    from qdrant_client.models import (
        Prefetch,
        FusionQuery,
        Fusion,
        Filter,
        FieldCondition,
        MatchValue,
    )

from ingest.reranker import CrossEncoderReranker

from search.runtime import (
    get_dense_model,
    get_splade,
    get_qdrant,
    get_cross_encoder_reranker,
)

# =========================
# config
# =========================

COLLECTION_NAME = "regulens"

TOP_K = 10
RERANK_TOP_K = 7
FINAL_TOP_N = 3


# =========================
# SPLADE (LOCAL ONLY)
# =========================

def compute_splade_query(text: str):
    if not USE_LOCAL_MODELS:
        raise RuntimeError("SPLADE is disabled in production")

    tokenizer, model, device = get_splade()

    with torch.no_grad():
        tokens = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        output = model(**tokens)
        logits = output.logits

        relu_log = torch.log1p(torch.relu(logits))
        weighted = relu_log * tokens.attention_mask.unsqueeze(-1)

        vec, _ = torch.max(weighted, dim=1)
        vec = vec.squeeze()

        nonzero = vec.nonzero(as_tuple=False).squeeze().cpu()
        values = vec[nonzero].cpu()

        return {
            "indices": nonzero.tolist(),
            "values": values.tolist(),
        }


# =========================
# reranking
# =========================

def rerank_results(query: str, points, rerank_k: int):
    reranker = get_cross_encoder_reranker()

    candidates = points[:rerank_k]
    passages = [
        p.payload["text"]
        for p in candidates
        if p.payload and "text" in p.payload
    ]

    rerank_scores = reranker.rerank(query, passages)

    scored = list(zip(rerank_scores, candidates))
    scored.sort(key=lambda x: x[0], reverse=True)

    return [point for _, point in scored]


# =========================
# hybrid search (LOCAL ONLY)
# =========================

def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    rerank_k: int = RERANK_TOP_K,
    version_filter: str | None = None,
):
    if not USE_LOCAL_MODELS:
        raise RuntimeError("Hybrid search is disabled in production")

    dense_model = get_dense_model()
    client = get_qdrant()

    qdrant_filter = None
    if version_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="version",
                    match=MatchValue(value=version_filter),
                )
            ]
        )

    dense_query = dense_model.encode(
        query,
        normalize_embeddings=True,
    ).tolist()

    sparse_query = compute_splade_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                using="dense",
                query=dense_query,
                limit=top_k,
                filter=qdrant_filter,
            ),
            Prefetch(
                using="sparse",
                query=sparse_query,
                limit=top_k,
                filter=qdrant_filter,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=top_k,
    )

    if not response.points:
        return []

    reranked = rerank_results(query, response.points, rerank_k)

    return reranked[:FINAL_TOP_N]
