import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from typing import List, Dict
from dotenv import load_dotenv

from groq import Groq

from search.hybrid_search import hybrid_search

load_dotenv()

# model initialization using Groq
client = Groq(
    api_key = os.getenv("GROQ_API_KEY")
)

MODEL_NAME = "llama-3.3-70b-versatile"


# prompt template

SYSTEM_PROMPT = """
    You are an expert assistant specializing in U.S. SEC regulations and
    financial disclosure requirements.

    You answer questions strictly using the provided regulatory context from:
    - The 2022 SEC Climate-Related Disclosure Proposed Rule
    - The 2024 SEC Climate-Related Disclosure Final Rule

    You must NOT rely on external knowledge or make assumptions beyond the context.

    Behavior rules:
    - If the user greets you (e.g., "hi", "hello"), respond politely and briefly.
    - If the user asks a question unrelated to SEC climate-related disclosure
    regulations, clearly state that you can only answer questions within this scope.
    - If the provided context is insufficient to answer the question, explicitly say so.

    Answering guidelines:
    - Prefer the 2024 Final Rule over the 2022 Proposed Rule when both are available.
    - You may combine information across sections or documents to improve coherence.
    - Do not introduce new interpretations or policy opinions.
    - Maintain a neutral, professional, compliance-safe tone suitable for legal,
    regulatory, or investor-facing analysis.
    - When helpful, you may briefly indicate whether an explanation reflects the
    SEC’s proposed (2022) or final (2024) position, without formal citations.
"""


def build_user_prompt(query: str, contexts: List[Dict]) -> str:
    """
    Build the user prompt with retrieved context.
    """

    context_blocks = []

    for i, ctx in enumerate(contexts, start=1):
        block = f"""
            [Context {i}]
            Document: {ctx['doc']}
            Version: {ctx['version']}
            Section: {ctx['section']}
            Text:
            {ctx['text']}
    """
        context_blocks.append(block.strip())

    context_str = "\n\n".join(context_blocks)

    user_prompt = f"""
        Answer the following question using ONLY the context below.

        Question:
        {query}

        Context:
        {context_str}

        Answer:
"""

    return user_prompt.strip()


# main rag function
def answer_query(
    query: str,
    top_k: int = 10,
    rerank_k: int = 3
) -> Dict:
    """
    End-to-end RAG answer generation.
    """

    # 1. hybrid retrieval + reranking
    results = hybrid_search(
        query = query,
        top_k = top_k,
        rerank_k = rerank_k,
        return_payload = True
    )

    if not results:
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "sources": []
        }

    # prepare context for prompt
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
        
    # Prefer Final Rule context first
    contexts.sort(
        key=lambda c: (c["version"] == "2024"),
        reverse=True
    )

    user_prompt = build_user_prompt(query, contexts)

    # Call Groq LLM
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
    )

    answer = completion.choices[0].message.content.strip()

    # 5. Return structured response
    return {
        "answer": answer,
        "sources": sources
    }


# testing
if __name__ == "__main__":
    query = (
        "Why does the SEC believe additional climate-related disclosure requirements are necessary for investors?"
    )

    response = answer_query(query)

    print("\n[ANSWER]\n")
    print(response["answer"])

    print("\n[SOURCES]\n")
    for s in response["sources"]:
        print(
            f"- {s['doc']} ({s['version']}) | Section: {s['section']}"
        )
