"""Step 4: store chunk vectors on disk in Chroma, so each PDF is embedded only once.

All PDFs share one collection. Every chunk carries a `pdf_id` (hash of the file's bytes) in
its metadata, so a search can be limited to one PDF with `filter={"pdf_id": ...}`.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.embeddings import EMBEDDING_RECORD, get_embeddings
from rag.loader import load_pdf
from rag.splitter import CHUNK_OVERLAP, CHUNK_SIZE, split_documents

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION = "recall_pdf"


class IncompatibleStore(RuntimeError):
    """The saved vectors were made with a different embedding model or prefixes."""


@dataclass
class IndexResult:
    pdf_id: str
    chunks: int
    embedded: int
    reused: int
    removed: int


def _client(persist_dir: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
    )


def open_store(persist_dir: Path = CHROMA_DIR) -> Chroma:
    """Open the saved collection, or create it (cosine distance) on first use."""
    client = _client(persist_dir)
    if COLLECTION in [c.name for c in client.list_collections()]:
        saved = client.get_collection(COLLECTION).metadata or {}
        changed = {k: (saved.get(k), v) for k, v in EMBEDDING_RECORD.items() if saved.get(k) != v}
        if changed:
            raise IncompatibleStore(
                f"{persist_dir} was built with different embedding settings "
                f"(saved, current): {changed}. Delete it to re-index: rm -rf {persist_dir}"
            )
        return Chroma(client=client, collection_name=COLLECTION, embedding_function=get_embeddings())

    return Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        collection_metadata=EMBEDDING_RECORD,
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


def delete_store(persist_dir: Path = CHROMA_DIR) -> None:
    """Drop the collection (all PDFs); the next `open_store` starts empty."""
    client = _client(persist_dir)
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)


def saved_embedding_record(store: Chroma) -> dict:
    return store._collection.metadata or {}


def pdf_id(path: str | Path) -> str:
    """Same bytes -> same id, even if the file is renamed or moved."""
    return hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest()[:16]


def chunk_id(chunk: Document) -> str:
    """Stable id: re-indexing the same chunk gives the same id, so nothing is duplicated."""
    meta = chunk.metadata
    text_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()[:8]
    return f"{meta['pdf_id']}-p{meta['page']}-s{meta['start_index']}-{text_hash}"


def index_pdf(
    store: Chroma,
    path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> IndexResult:
    """Load, split and store a PDF, embedding only chunks that are not stored yet.

    Chunks of this PDF that no longer exist (e.g. after changing chunk_size) are removed.
    """
    pid = pdf_id(path)
    chunks = split_documents(load_pdf(path), chunk_size, chunk_overlap)
    for chunk in chunks:
        chunk.metadata["pdf_id"] = pid
    ids = [chunk_id(c) for c in chunks]

    stored = set(store.get(where={"pdf_id": pid}, include=[])["ids"])
    new = [(c, i) for c, i in zip(chunks, ids) if i not in stored]
    stale = list(stored - set(ids))

    if stale:
        store.delete(ids=stale)
    if new:
        store.add_documents([c for c, _ in new], ids=[i for _, i in new])
    return IndexResult(pid, len(chunks), embedded=len(new), reused=len(chunks) - len(new),
                       removed=len(stale))
