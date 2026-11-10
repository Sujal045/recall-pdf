"""Step 5: turn the vector store into a retriever: question in, relevant chunks out."""

from typing import Any, Literal

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from rag.embeddings import cosine_similarity

TOP_K = 4

SearchType = Literal["similarity", "mmr", "similarity_score_threshold"]


def get_retriever(
    store: Chroma,
    pdf_id: str,
    k: int = TOP_K,
    search_type: SearchType = "similarity",
    **search_kwargs: Any,
) -> VectorStoreRetriever:
    """A retriever that only searches the chunks of one PDF.

    Extra `search_kwargs` go to the search, e.g. `fetch_k`/`lambda_mult` for "mmr" or
    `score_threshold` for "similarity_score_threshold".
    """
    return store.as_retriever(
        search_type=search_type,
        search_kwargs={"k": k, "filter": {"pdf_id": pdf_id}, **search_kwargs},
    )


def similarity_scores(store: Chroma, question: str, docs: list) -> list[float]:
    """Cosine similarity of the question to each returned chunk (retrievers drop scores)."""
    if not docs:
        return []
    question_vector = store.embeddings.embed_query(question)
    stored = store.get(ids=[doc.id for doc in docs], include=["embeddings"])
    by_id = dict(zip(stored["ids"], stored["embeddings"]))
    return [cosine_similarity(question_vector, list(by_id[doc.id])) for doc in docs]
