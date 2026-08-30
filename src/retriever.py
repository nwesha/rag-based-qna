"""FAISS-based top-k chunk retrieval."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.embeddings import encode_query
from src.pdf_processor import Chunk
from src.config import DEFAULT_TOP_K


@dataclass
class RetrievedChunk:
    """A chunk returned by retrieval, enriched with its similarity score."""

    text: str
    score: float              # cosine similarity (0–1 for normalized vectors)
    source_page: int
    chunk_index: int
    rank: int                 # 1-indexed retrieval rank


def retrieve(
    query: str,
    model: SentenceTransformer,
    index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    top_k: int = DEFAULT_TOP_K,
) -> tuple[list[RetrievedChunk], float]:
    """Retrieve the top-k most relevant chunks for a query.

    Parameters
    ----------
    query : str
        The user's question.
    model : SentenceTransformer
        Loaded embedding model (shared with index building).
    index : faiss.IndexFlatIP
        Pre-built FAISS index.
    chunks : list[Chunk]
        The original chunk objects (aligned with index rows).
    top_k : int
        Number of chunks to retrieve.

    Returns
    -------
    tuple[list[RetrievedChunk], float]
        (retrieved_chunks, latency_seconds)
    """
    start = time.perf_counter()

    query_vec = encode_query(model, query)
    # FAISS search returns (distances, indices), both shape (1, top_k)
    scores, indices = index.search(query_vec, min(top_k, len(chunks)))

    results: list[RetrievedChunk] = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx == -1:
            continue  # fewer results than top_k
        chunk = chunks[idx]
        results.append(
            RetrievedChunk(
                text=chunk.text,
                score=float(score),
                source_page=chunk.source_page,
                chunk_index=chunk.chunk_index,
                rank=rank,
            )
        )

    latency = time.perf_counter() - start
    return results, latency
