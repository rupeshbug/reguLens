from functools import lru_cache
import os

USE_LOCAL_MODELS = os.getenv("USE_LOCAL_MODELS", "false").lower() == "true"


# -------------------------
# Lightweight / always-on
# -------------------------

@lru_cache
def get_qdrant():
    from qdrant_client import QdrantClient

    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


@lru_cache
def get_llm_client():
    from groq import Groq

    return Groq(api_key=os.getenv("GROQ_API_KEY"))


@lru_cache
def get_decomposer():
    from search.query_decomposition import QueryDecomposer

    return QueryDecomposer()


@lru_cache
def get_dense_model():
    """
    Local-only dense model.
    In prod, this should never be called.
    """
    if not USE_LOCAL_MODELS:
        raise RuntimeError("Dense model disabled in production")

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )


@lru_cache
def get_splade():
    """
    Local-only SPLADE model.
    """
    if not USE_LOCAL_MODELS:
        raise RuntimeError("SPLADE disabled in production")

    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        "naver/splade-cocondenser-ensembledistil"
    )
    model = AutoModelForMaskedLM.from_pretrained(
        "naver/splade-cocondenser-ensembledistil"
    ).to(device)

    model.eval()
    return tokenizer, model, device


@lru_cache
def get_cross_encoder_reranker():
    from ingest.reranker import CrossEncoderReranker
    return CrossEncoderReranker()
