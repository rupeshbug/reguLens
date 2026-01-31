from typing import Dict
from search.fast_dense_search import fast_dense_search
from search.rag_answer import build_user_prompt, SYSTEM_PROMPT
from search.runtime import get_llm_client

MODEL_NAME = "llama-3.3-70b-versatile"


def answer_query_fast(
    query: str,
    version_filter: str | None = None
) -> Dict:
    """
    Fast, production-safe RAG path.
    """

    results = fast_dense_search(
        query=query,
        version_filter=version_filter,
        top_k=5
    )

    if not results:
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "sources": []
        }

    contexts = []
    sources = []

    for r in results:
        payload = r.payload
        text = payload.get("text", "").strip()

        if not text:
            continue

        contexts.append({
            "doc": payload.get("document_id"),
            "version": payload.get("version"),
            "section": payload.get("section_id"),
            "text": text
        })

        sources.append({
            "doc": payload.get("document_id"),
            "version": payload.get("version"),
            "section": payload.get("section_id")
        })

    if not contexts:
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "sources": []
        }

    # Prefer final rule
    contexts.sort(
        key=lambda c: (c["version"] == "2024_final"),
        reverse=True
    )

    user_prompt = build_user_prompt(query, contexts)

    client = get_llm_client()

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    return {
        "answer": completion.choices[0].message.content.strip(),
        "sources": sources
    }
