"""Step 1: load a PDF and look at the Documents LangChain gives back.

Usage: python step1_load.py <pdf> [--pages N] [--chars N]
"""

import argparse

from rag.loader import empty_pages, load_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a PDF into LangChain Documents.")
    parser.add_argument("pdf", help="path to a PDF file")
    parser.add_argument("--pages", type=int, default=2, help="how many pages to preview (default 2)")
    parser.add_argument("--chars", type=int, default=300, help="characters per preview (default 300)")
    args = parser.parse_args()

    docs = load_pdf(args.pdf)
    print(f"Loaded {len(docs)} page(s) from {args.pdf}\n")

    print("=== Metadata per page ===")
    for doc in docs:
        print(f"page {doc.metadata['page']} ({len(doc.page_content)} chars): {doc.metadata}")

    print(f"\n=== First {args.chars} characters of the first {args.pages} page(s) ===")
    for doc in docs[: args.pages]:
        print(f"\n--- page_label {doc.metadata.get('page_label')} ---")
        print(doc.page_content[: args.chars])

    if blank := empty_pages(docs):
        print(f"\nWARNING: no extractable text on page(s) {blank}. "
              "They are probably scanned images and need OCR.")


if __name__ == "__main__":
    main()
