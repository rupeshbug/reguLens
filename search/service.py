from typing import Dict
from search.rag_answer_fast import answer_query_fast

def answer_regulatory_question(
    query: str,
    version: str | None = None,
) -> Dict:
    """
    Production entry point (Render free tier).

    - Dense-only retrieval
    - Low memory footprint
    - Low latency
    - Compliance-safe
    """
    return answer_query_fast(
        query=query,
        version_filter=version,
    )
