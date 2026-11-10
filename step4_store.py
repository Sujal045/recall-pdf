"""Step 4: store the PDF's chunk vectors in Chroma on disk, then look inside.

Run it twice: the second run reuses the saved vectors instead of embedding again.
Usage: python step4_store.py <pdf> [--question TEXT] [--reset] [--chunk-size N] [--chunk-overlap N]
"""

import argparse
import sys
import time

from rag.embeddings import OllamaNotReady, cosine_similarity
from rag.splitter import CHUNK_OVERLAP, CHUNK_SIZE
from rag.vectorstore import (
    CHROMA_DIR,
    IncompatibleStore,
    delete_store,
    index_pdf,
    open_store,
    saved_embedding_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF into Chroma.")
    parser.add_argument("pdf", help="path to a PDF file")
    parser.add_argument("--question", default="How long does a worker bee live?")
    parser.add_argument("--reset", action="store_true", help="delete all stored vectors first")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    if args.reset:
        delete_store()
        print(f"Deleted the stored collection in {CHROMA_DIR}")

    try:
        store = open_store()
        started = time.perf_counter()
        result = index_pdf(store, args.pdf, args.chunk_size, args.chunk_overlap)
        elapsed = time.perf_counter() - started
        hits = store.similarity_search_with_score(args.question, k=3, filter={"pdf_id": result.pdf_id})
        question_vector = store.embeddings.embed_query(args.question)
    except (OllamaNotReady, IncompatibleStore) as err:
        sys.exit(str(err))

    print(f"=== Indexing {args.pdf} (pdf_id {result.pdf_id}) ===")
    print(f"{result.chunks} chunk(s): {result.embedded} embedded now, {result.reused} reused "
          f"from disk, {result.removed} stale removed ({elapsed:.2f}s)")
    print(f"Collection holds {len(store.get(include=[])['ids'])} item(s) in total, "
          f"saved in {CHROMA_DIR}")
    print(f"Embedding record saved with the collection: {saved_embedding_record(store)}")
    if not result.chunks:
        print("\nNo text in this PDF, so nothing was stored. Is it scanned? Run step1_load.py on it.")
        return

    print("\n=== What Chroma stores per item (first 2 of this PDF) ===")
    peek = store.get(where={"pdf_id": result.pdf_id}, limit=2,
                     include=["documents", "metadatas", "embeddings"])
    for item_id, text, meta, vector in zip(peek["ids"], peek["documents"], peek["metadatas"],
                                           peek["embeddings"]):
        print(f"\nid:        {item_id}")
        print(f"document:  {text[:100]!r}...")
        print(f"metadata:  {meta}")
        print(f"embedding: {len(vector)} numbers, starts {[round(float(x), 4) for x in vector[:4]]}")

    print(f"\n=== similarity_search_with_score({args.question!r}, k=3) ===")
    print("score = cosine distance = 1 - cosine similarity (lower is closer)")
    for doc, distance in hits:
        stored_vector = store.get(ids=[doc.id], include=["embeddings"])["embeddings"][0]
        similarity = cosine_similarity(question_vector, list(stored_vector))
        start = doc.page_content[:60].replace("\n", " ")
        print(f"distance {distance:.3f}  (step 3 cosine similarity {similarity:.3f})  "
              f"page {doc.metadata['page']}  {start!r}...")


if __name__ == "__main__":
    main()
