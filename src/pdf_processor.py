"""PDF text extraction and configurable text chunking."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import fitz  # PyMuPDF


@dataclass
class Chunk:
    """A text chunk extracted from a PDF document."""

    text: str
    source_page: int          # 1-indexed page number
    chunk_index: int          # global index across the whole document
    start_char: int = 0       # character offset in the full document text
    metadata: dict = field(default_factory=dict)


def extract_text_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Extract text from a PDF file, returning (page_number, text) pairs.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw bytes of the uploaded PDF file.

    Returns
    -------
    list[tuple[int, str]]
        List of (1-indexed page number, page text) tuples.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[tuple[int, str]] = []
    for page_num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append((page_num, text))
    doc.close()
    return pages


def _clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    # Collapse multiple spaces/tabs into one space
    text = re.sub(r"[^\S\n]+", " ", text)
    # Collapse 3+ newlines into 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    pages: list[tuple[int, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split page text into overlapping chunks.

    Parameters
    ----------
    pages : list[tuple[int, str]]
        Output of :func:`extract_text_from_pdf`.
    chunk_size : int
        Maximum number of characters per chunk.
    chunk_overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[Chunk]
        Ordered list of :class:`Chunk` objects.
    """
    # Concatenate all pages, tracking page boundaries
    full_text = ""
    page_boundaries: list[tuple[int, int, int]] = []  # (page_num, start, end)
    for page_num, text in pages:
        cleaned = _clean_text(text)
        start = len(full_text)
        full_text += cleaned + "\n"
        end = len(full_text)
        page_boundaries.append((page_num, start, end))

    if not full_text.strip():
        return []

    # Sliding-window chunking
    chunks: list[Chunk] = []
    stride = max(chunk_size - chunk_overlap, 1)
    idx = 0
    chunk_index = 0

    while idx < len(full_text):
        end = min(idx + chunk_size, len(full_text))
        chunk_text_str = full_text[idx:end].strip()

        if not chunk_text_str:
            idx += stride
            continue

        # Determine which page this chunk belongs to (by midpoint)
        midpoint = idx + len(chunk_text_str) // 2
        source_page = _find_page(midpoint, page_boundaries)

        chunks.append(
            Chunk(
                text=chunk_text_str,
                source_page=source_page,
                chunk_index=chunk_index,
                start_char=idx,
            )
        )
        chunk_index += 1
        idx += stride

    return chunks


def _find_page(char_offset: int, boundaries: list[tuple[int, int, int]]) -> int:
    """Find the page number for a given character offset."""
    for page_num, start, end in boundaries:
        if start <= char_offset < end:
            return page_num
    # Default to the last page
    return boundaries[-1][0] if boundaries else 1


def process_pdf(
    pdf_bytes: bytes,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Full pipeline: PDF bytes → extracted text → cleaned chunks.

    This is the main entry point for the Streamlit app.
    """
    pages = extract_text_from_pdf(pdf_bytes)
    return chunk_text(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
