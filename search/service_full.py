from typing import Dict
from search.rag_answer import answer_query

def answer_regulatory_question_full(
    query: str,
    version: str | None = None,
) -> Dict:
    return answer_query(
        query=query,
        version_filter=version,
        decompose=True,
        global_rerank_enabled=True,
    )
