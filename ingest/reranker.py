from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        """
        Cross-encoder reranker for (query, passage) pairs.
        """
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, passages: list[str]) -> list[float]:
        """
        Returns relevance scores aligned with passages order.
        """
        pairs = [(query, passage) for passage in passages]
        scores = self.model.predict(pairs)
        return scores.tolist()
