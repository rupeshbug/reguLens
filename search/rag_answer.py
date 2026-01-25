import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from typing import List, Dict
from dotenv import load_dotenv

from groq import Groq

from search.hybrid_search import hybrid_search
from search.global_rerank import global_rerank

from search.runtime import get_decomposer, get_llm_client

from dotenv import load_dotenv


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
    - When the question asks for differences or changes, explicitly compare the 2022 Proposed Rule and the 2024 Final Rule.
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
    rerank_k: int = 3,
    version_filter: str | None = None,
    decompose: bool = True,
    global_rerank_enabled: bool = True
) -> Dict:
    """
    End-to-end RAG answer generation.
    """
    
    # decompose query
    queries = [query]

    if decompose:
        decomposer = get_decomposer()
        queries = decomposer.decompose(query)

    # run retrieval for each sub-query
    all_results = []

    for q in queries:
        retrieved = hybrid_search(
            query=q,
            top_k=top_k,
            rerank_k=rerank_k,
            return_payload=True,
            version_filter=version_filter
        )
        all_results.extend(retrieved)

    # deduplicate by point id
    seen_ids = set()
    results = []
    for r in all_results:
        if r.id not in seen_ids:
            seen_ids.add(r.id)
            results.append(r)

    if not results:
        return {
            "answer": "The provided documents do not contain sufficient information to answer this question.",
            "sources": []
        }
        
    if global_rerank_enabled and len(results) > 1:
        results = global_rerank(
            query=query,          
            candidates=results,
            top_k=8              
        )    

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
    client = get_llm_client()
    
    completion = client.chat.completions.create(
        model = MODEL_NAME,
        temperature = 0.2,
        messages = [
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
    
