"""Step 1: load a PDF into LangChain Documents, one per page."""

import warnings
from pathlib import Path

from langchain_core.documents import Document

with warnings.catch_warnings():
    # langchain-community warns on import that it is being sunset; PyPDFLoader still works.
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.document_loaders import PyPDFLoader


def load_pdf(path: str | Path) -> list[Document]:
    """Return one Document per page of the PDF at `path`.

    Each Document has `page_content` (the page's extracted text) and `metadata`
    (source path, 0-based `page`, printed `page_label`, `total_pages`, plus PDF info).
    """
    pdf_path = Path(path).expanduser()
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a .pdf file: {pdf_path}")

    return PyPDFLoader(str(pdf_path)).load()


def empty_pages(docs: list[Document], min_chars: int = 20) -> list[int]:
    """Return 0-based page numbers with (almost) no text, usually scanned images that need OCR."""
    return [doc.metadata["page"] for doc in docs if len(doc.page_content.strip()) < min_chars]
