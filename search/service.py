from typing import Dict

from search.rag_answer import answer_query


def answer_regulatory_question(query: str, version: str | None = None) -> Dict:
    """
    Stable production entry point for regulatory Q&A.

    Uses the best-performing retrieval configuration by default:
    - Query decomposition
    - Hybrid retrieval
    - Global cross-encoder reranking
    - Version-aware context prioritization
    """

    return answer_query(
        query = query,
        version_filter = version,
        decompose =True,
        global_rerank_enabled = True
    )
