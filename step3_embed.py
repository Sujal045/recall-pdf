"""Step 3: embed the PDF's chunks and compare meanings with cosine similarity.

Needs Ollama running with nomic-embed-text pulled.
Usage: python step3_embed.py <pdf> [--chunk-size N] [--chunk-overlap N] [--question TEXT ...]
"""

import argparse
import sys

from rag.embeddings import OllamaNotReady, cosine_similarity, get_embeddings
from rag.loader import load_pdf
from rag.splitter import CHUNK_OVERLAP, CHUNK_SIZE, split_documents

SENTENCE_PAIRS = [
    ("The queen lays the eggs.", "The colony's eggs are produced by the queen bee."),
    ("The queen lays the eggs.", "Drones are male bees that cannot sting."),
    ("The queen lays the eggs.", "Interest rates rose again last quarter."),
]

DEFAULT_QUESTIONS = [
    "How do bees tell each other where food is?",
    "What is the biggest threat to honeybees?",
    "How long does a worker bee live?",
    "What is the capital of France?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed a PDF's chunks and compare them.")
    parser.add_argument("pdf", help="path to a PDF file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--question", action="append", default=[],
                        help="extra question to compare (repeatable)")
    args = parser.parse_args()

    chunks = split_documents(load_pdf(args.pdf), args.chunk_size, args.chunk_overlap)
    if not chunks:
        sys.exit("No text to embed. Is the PDF scanned? Run step1_load.py on it.")
    embeddings = get_embeddings()

    try:
        print(f"Embedding {len(chunks)} chunk(s) with {embeddings.model}...")
        chunk_vectors = embeddings.embed_documents([c.page_content for c in chunks])

        first = chunk_vectors[0]
        length = sum(x * x for x in first) ** 0.5
        print(f"Each chunk is now a vector of {len(first)} numbers (vector length {length:.3f}).")
        print(f"Chunk #0 starts: {[round(x, 4) for x in first[:8]]} ...\n")

        print("=== Sentence vs sentence ===")
        for a, b in SENTENCE_PAIRS:
            va, vb = embeddings.embed_documents([a, b])
            print(f"{cosine_similarity(va, vb):.3f}  {a!r} vs {b!r}")

        questions = DEFAULT_QUESTIONS + args.question
        question_vectors = [embeddings.embed_query(q) for q in questions]
    except OllamaNotReady as err:
        sys.exit(f"Ollama is not ready: {err}")

    print("\n=== Question vs each chunk (* = most similar chunk) ===")
    width = max(len(q) for q in questions)
    print(" " * width + "  " + " ".join(f"#{i} p{c.metadata['page']}".rjust(7) for i, c in enumerate(chunks)))
    for question, qv in zip(questions, question_vectors):
        scores = [cosine_similarity(qv, cv) for cv in chunk_vectors]
        best = scores.index(max(scores))
        cells = [f"{s:.3f}{'*' if i == best else ' '}".rjust(7) for i, s in enumerate(scores)]
        print(f"{question:<{width}}  " + " ".join(cells))


if __name__ == "__main__":
    main()
