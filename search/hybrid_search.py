import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM

from qdrant_client import QdrantClient
from qdrant_client.models import Prefetch, FusionQuery, Fusion

from ingest.reranker import CrossEncoderReranker

# config

COLLECTION_NAME = "regulens"

TOP_K = 10            # retrieve more for reranking
RERANK_TOP_K = 7      # rerank top N
FINAL_TOP_N = 3       # final context chunks

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPLADE_MODEL_ID = "naver/splade-cocondenser-ensembledistil"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# load models

print("[INFO] Loading models...")

dense_model = SentenceTransformer(DENSE_MODEL_NAME, device=DEVICE)

splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_ID)
splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_ID).to(DEVICE)
splade_model.eval()

reranker = CrossEncoderReranker()  

client = QdrantClient(url="http://localhost:6333")



# SPLADE query encoder

@torch.no_grad()
def compute_splade_query(text: str):
    tokens = splade_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(DEVICE)

    output = splade_model(**tokens)
    logits = output.logits

    relu_log = torch.log1p(torch.relu(logits))
    weighted = relu_log * tokens.attention_mask.unsqueeze(-1)

    vec, _ = torch.max(weighted, dim=1)
    vec = vec.squeeze()

    nonzero = vec.nonzero(as_tuple=False).squeeze().cpu()
    values = vec[nonzero].cpu()

    return {
        "indices": nonzero.tolist(),
        "values": values.tolist()
    }


# reranking
def rerank_results(query: str, points, rerank_k: int):
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



# hybrid search

def hybrid_search(
    query: str,
    top_k: int = TOP_K,
    rerank_k: int = RERANK_TOP_K ,
    return_payload: bool = True,
):

    dense_query = dense_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    sparse_query = compute_splade_query(query)

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                using="dense",
                query=dense_query,
                limit=top_k,
            ),
            Prefetch(
                using="sparse",
                query=sparse_query,
                limit=top_k,
            ),
        ],
        query = FusionQuery(fusion=Fusion.RRF),
        limit = top_k,
    )

    if not response.points:
        return []

    reranked = rerank_results(query, response.points, rerank_k)

    return reranked[:FINAL_TOP_N]


# entry point

if __name__ == "__main__":
    hybrid_search(
        "Why does the SEC believe additional climate-related disclosure "
        "requirements are necessary for investors?"
    )
    