import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    SparseVectorParams,
    HnswConfigDiff,
    PayloadSchemaType
)
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "regulens"

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

def create_collection():
    # Delete if exists (safe for local dev)
    if COLLECTION_NAME in [c.name for c in client.get_collections().collections]:
        print(f"[INFO] Collection '{COLLECTION_NAME}' already exists. Deleting...")
        client.delete_collection(COLLECTION_NAME)

    print("[INFO] Creating collection...")

    client.create_collection(
        collection_name = COLLECTION_NAME,
        vectors_config = {
            "dense": VectorParams(
                size = 384,
                distance = Distance.COSINE,
                hnsw_config = HnswConfigDiff(
                    m = 16,
                    ef_construct = 128
                )
            )
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams()
        }
    )

    print("[INFO] Creating payload indexes...")

    # Metadata filters for pre-filtering -> faster retrieval
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="version",
        field_schema=PayloadSchemaType.KEYWORD
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="section_id",
        field_schema=PayloadSchemaType.KEYWORD
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="document_id",
        field_schema=PayloadSchemaType.KEYWORD
    )

    print("[DONE] Collection created successfully.")

if __name__ == "__main__":
    create_collection()
