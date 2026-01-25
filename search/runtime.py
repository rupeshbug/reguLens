# search/runtime.py
from functools import lru_cache
import os
from groq import Groq
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM
from qdrant_client import QdrantClient

from search.query_decomposition import QueryDecomposer
from sentence_transformers import  SentenceTransformer
from qdrant_client import QdrantClient
from ingest.reranker import CrossEncoderReranker


@lru_cache
def get_decomposer():
    return QueryDecomposer()


@lru_cache
def get_llm_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

@lru_cache
def get_dense_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )


@lru_cache
def get_splade():
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
def get_qdrant():
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )


@lru_cache
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache
def get_cross_encoder_reranker():
    return CrossEncoderReranker()
