"""Step 2: split the PDF's pages into chunks and look at where the splitter cuts.

Usage: python step2_split.py <pdf> [--chunk-size N] [--chunk-overlap N] [--show N]
"""

import argparse

from rag.loader import load_pdf
from rag.splitter import CHUNK_OVERLAP, CHUNK_SIZE, split_documents


def overlap_with_next(chunk, next_chunk) -> str:
    """Text the two chunks share, or "" if they come from different pages or don't overlap."""
    if chunk.metadata["page"] != next_chunk.metadata["page"]:
        return ""
    end = chunk.metadata["start_index"] + len(chunk.page_content)
    shared = end - next_chunk.metadata["start_index"]
    return chunk.page_content[-shared:] if shared > 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a PDF into chunks.")
    parser.add_argument("pdf", help="path to a PDF file")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP)
    parser.add_argument("--show", type=int, default=3, help="how many chunks to print in full")
    args = parser.parse_args()

    pages = load_pdf(args.pdf)
    chunks = split_documents(pages, args.chunk_size, args.chunk_overlap)

    lengths = [len(c.page_content) for c in chunks]
    print(f"{len(pages)} page(s) -> {len(chunks)} chunk(s) "
          f"(chunk_size={args.chunk_size}, chunk_overlap={args.chunk_overlap})")
    if not chunks:
        print("No text to split. Is the PDF scanned? Run step1_load.py on it.")
        return
    print(f"chunk length: min {min(lengths)}, max {max(lengths)}, "
          f"avg {sum(lengths) // len(lengths)} chars\n")

    print("=== All chunks ===")
    for i, chunk in enumerate(chunks):
        meta = chunk.metadata
        start = chunk.page_content[:50].replace("\n", " ")
        print(f"#{i:<3} page {meta['page']}  start_index {meta['start_index']:<5} "
              f"{len(chunk.page_content):>4} chars  {start!r}...")

    print(f"\n=== First {args.show} chunk(s) in full ===")
    for i, chunk in enumerate(chunks[: args.show]):
        print(f"\n--- chunk #{i} (page {chunk.metadata['page']}) ---")
        print(chunk.page_content)
        if i + 1 < len(chunks) and (shared := overlap_with_next(chunk, chunks[i + 1])):
            print(f"\n[overlap: chunk #{i + 1} starts by repeating {len(shared)} chars: {shared!r}]")


if __name__ == "__main__":
    main()
