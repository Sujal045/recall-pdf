"""Step 3: turn text into vectors (embeddings) with a local Ollama model."""

import math

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from ollama import ResponseError

EMBED_MODEL = "nomic-embed-text"

# nomic-embed-text was trained with these task prefixes; it retrieves better when the indexed
# text and the question are labelled. Index and query must use the same scheme.
DOCUMENT_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


class OllamaNotReady(RuntimeError):
    """Ollama is not running, or the embedding model has not been pulled."""


class NomicEmbeddings(Embeddings):
    """`nomic-embed-text` through Ollama, adding its task prefixes.

    Wraps `OllamaEmbeddings` instead of subclassing it because its `embed_query`
    calls `embed_documents`, which would add both prefixes to a question.
    """

    def __init__(self, base_url: str | None = None):
        self.model = EMBED_MODEL
        self._ollama = OllamaEmbeddings(model=self.model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed([DOCUMENT_PREFIX + text for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([QUERY_PREFIX + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._ollama.embed_documents(texts)
        except ConnectionError as err:
            raise OllamaNotReady(
                "Cannot reach Ollama. Start it with `sudo snap start ollama` "
                "(or `ollama serve`), then check with `ollama list`."
            ) from err
        except ResponseError as err:
            if err.status_code == 404:
                raise OllamaNotReady(
                    f"Model {self.model!r} is not pulled. Run: ollama pull {self.model}"
                ) from err
            raise


def get_embeddings(base_url: str | None = None) -> Embeddings:
    """The embedding model used for both indexing chunks and embedding questions."""
    return NomicEmbeddings(base_url=base_url)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(angle between a and b): 1 = same direction (same meaning), near 0 = unrelated."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))
