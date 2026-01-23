import json
import uuid
from pathlib import Path
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForMaskedLM


# config

COLLECTION_NAME = "regulens"
CHUNKS_DIR = Path("data/chunks")

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPLADE_MODEL_ID = "naver/splade-cocondenser-ensembledistil"

BATCH_SIZE = 32


# load models

print("[INFO] Loading dense embedding model...")
dense_model = SentenceTransformer(DENSE_MODEL_NAME)

print("[INFO] Loading SPLADE model...")
splade_tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_ID)
splade_model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_ID)
splade_model.eval()

# SPLADE model

@torch.no_grad()
def compute_splade_sparse_vector(text: str):
    """
    Returns sparse vector in Qdrant format:
    { "indices": [...], "values": [...] }
    """
    tokens = splade_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = splade_model(**tokens)
    logits = output.logits

    # SPLADE transformation
    relu_log = torch.log1p(torch.relu(logits))
    weighted = relu_log * tokens.attention_mask.unsqueeze(-1)

    # max over sequence length
    vec, _ = torch.max(weighted, dim=1)
    vec = vec.squeeze()

    # keep only non-zero entries
    nonzero_indices = vec.nonzero(as_tuple=False).squeeze().cpu()
    nonzero_values = vec[nonzero_indices].cpu()

    return {
        "indices": nonzero_indices.tolist(),
        "values": nonzero_values.tolist()
    }

# ingestion

def ingest_chunks(json_file: Path):
    print(f"\n=== Ingesting {json_file.name} ===")

    with open(json_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[INFO] Loaded {len(chunks)} chunks")

    client = QdrantClient(url="http://localhost:6333")

    texts = [c["text"] for c in chunks]

    print("[INFO] Generating dense embeddings...")
    dense_vectors = dense_model.encode(
        texts,
        batch_size = BATCH_SIZE,
        show_progress_bar = True,
        normalize_embeddings = True
    )

    points = []

    print("[INFO] Generating SPLADE sparse vectors + preparing points...")
    for chunk, dense_vec in tqdm(zip(chunks, dense_vectors), total=len(chunks)):
        sparse_vec = compute_splade_sparse_vector(chunk["text"])

        point = PointStruct(
            id = uuid.uuid4().hex,
            vector = {
                "dense": dense_vec.tolist(),
                "sparse": sparse_vec
            },
            payload = {
                "document_id": chunk["document_id"],
                "version": chunk["version"],
                "section_id": chunk["section_id"],
                "section_path": chunk["section_path"],
                "title": chunk["title"],
                "text": chunk["text"]
            }
        )
        points.append(point)

    print(f"[INFO] Upserting {len(points)} points...")
    client.upsert(
        collection_name = COLLECTION_NAME,
        points = points
    )

    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"[DONE] Collection now contains {count} points")


if __name__ == "__main__":
    ingest_chunks(CHUNKS_DIR / "2022_proposed_chunks.json")
    ingest_chunks(CHUNKS_DIR / "2024_final_chunks.json")
