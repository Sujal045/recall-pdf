"""Step 5: ask the retriever questions and read which chunks come back.

The sample PDF's pages are short, so this step indexes it with 300-character chunks
(14 chunks) by default; with 1000 there are only 4 and k=4 would return all of them.
Usage: python step5_retrieve.py <pdf> [--question TEXT ...] [--chunk-size N] [--chunk-overlap N]
"""

import argparse
import logging
import sys

from langchain_core.runnables import Runnable

from rag.embeddings import OllamaNotReady
from rag.retriever import get_retriever, similarity_scores
from rag.vectorstore import IncompatibleStore, index_pdf, open_store

WORKER_LIFESPAN = "How long does a worker bee live?"
WAGGLE_DANCE = "Who decoded the waggle dance?"
BIGGEST_THREAT = "What is the biggest threat to honeybees?"
OFF_TOPIC = "What is the capital of France?"
DEFAULT_QUESTIONS = [WORKER_LIFESPAN, WAGGLE_DANCE, BIGGEST_THREAT, OFF_TOPIC]
THRESHOLD = 0.7

# Section 4 prints how many chunks pass the threshold; LangChain's log line would repeat it out of order.
logging.getLogger("langchain_core.vectorstores.base").setLevel(logging.ERROR)


def show(store, question, docs) -> None:
    print(f"\nQ: {question}")
    if not docs:
        print("   (no chunks returned)")
    for rank, (doc, score) in enumerate(zip(docs, similarity_scores(store, question, docs)), 1):
        text = doc.page_content[:70].replace("\n", " ")
        print(f"   {rank}. sim {score:.3f}  page {doc.metadata['page']}  "
              f"start {doc.metadata['start_index']:<4} {text!r}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve chunks of a PDF for questions.")
    parser.add_argument("pdf", help="path to a PDF file")
    parser.add_argument("--question", action="append", default=[],
                        help="extra question for sections 1 and 4 (repeatable)")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    args = parser.parse_args()
    questions = DEFAULT_QUESTIONS + args.question

    try:
        store = open_store()
        result = index_pdf(store, args.pdf, args.chunk_size, args.chunk_overlap)
        print(f"{args.pdf}: {result.chunks} chunks ({result.embedded} embedded now, "
              f"{result.reused} reused)")
        if not result.chunks:
            sys.exit("No text in this PDF. Is it scanned? Run step1_load.py on it.")
        pid = result.pdf_id

        retriever = get_retriever(store, pid)
        print("\n=== 1. retriever.invoke(question), similarity search, k=4 ===")
        for question in questions:
            show(store, question, retriever.invoke(question))

        print("\n=== 2. Effect of k ===")
        for k in (1, 4):
            print(f"\nk={k}:", end="")
            show(store, WORKER_LIFESPAN, get_retriever(store, pid, k=k).invoke(WORKER_LIFESPAN))

        print("\n=== 3. search_type 'similarity' vs 'mmr' (k=4, fetch_k=10, lambda_mult=0.5) ===")
        mmr = get_retriever(store, pid, search_type="mmr", fetch_k=10, lambda_mult=0.5)
        print("\nsimilarity:", end="")
        show(store, BIGGEST_THREAT, retriever.invoke(BIGGEST_THREAT))
        print("\nmmr:", end="")
        show(store, BIGGEST_THREAT, mmr.invoke(BIGGEST_THREAT))

        print(f"\n=== 4. Fixed score threshold {THRESHOLD} (similarity_score_threshold) ===")
        thresholded = get_retriever(store, pid, search_type="similarity_score_threshold",
                                    score_threshold=THRESHOLD)
        for question in questions:
            kept = thresholded.invoke(question)
            print(f"{len(kept)} of up to 4 chunks kept  {question!r}")

        print("\n=== 5. A retriever is a Runnable ===")
        print(f"isinstance(retriever, Runnable): {isinstance(retriever, Runnable)}")
        pair = [WORKER_LIFESPAN, WAGGLE_DANCE]
        for question, docs in zip(pair, retriever.batch(pair)):
            print(f"batch -> {len(docs)} docs, pages {[d.metadata['page'] for d in docs]}  {question!r}")
    except (OllamaNotReady, IncompatibleStore) as err:
        sys.exit(str(err))


if __name__ == "__main__":
    main()
