from typing import Dict
from search.rag_answer import answer_query
from search.rag_answer_fast import answer_query_fast


def answer_regulatory_question(
    query: str,
    version: str | None = None,
    mode: str = "fast",  # fast | full
) -> Dict:
    """
    Production entry point.

    fast:
      - dense-only retrieval
      - low latency
    full:
      - hybrid retrieval
      - query decomposition
      - reranking
    """

    if mode == "full":
        return answer_query(
            query=query,
            version_filter=version,
            decompose=True,
            global_rerank_enabled=True
        )

    return answer_query_fast(
        query=query,
        version_filter=version
    )
