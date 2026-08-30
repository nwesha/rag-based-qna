"""SentenceTransformer embedding and FAISS index construction."""

from __future__ import annotations

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL
from src.pdf_processor import Chunk


def load_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Load and return a SentenceTransformer model.

    In the Streamlit app this is wrapped with ``@st.cache_resource``
    so the model is loaded only once.
    """
    return SentenceTransformer(model_name)


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[Chunk],
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Encode chunk texts into dense vectors.

    Parameters
    ----------
    model : SentenceTransformer
        The loaded embedding model.
    chunks : list[Chunk]
        Chunks produced by :mod:`src.pdf_processor`.
    batch_size : int
        Batch size for encoding.
    show_progress : bool
        Whether to show a progress bar (useful in CLI, not in Streamlit).

    Returns
    -------
    np.ndarray
        Matrix of shape ``(len(chunks), embedding_dim)`` with L2-normalized vectors.
    """
    texts = [c.text for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # L2-normalize for cosine via inner product
    )
    return np.asarray(embeddings, dtype=np.float32)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from L2-normalized embeddings.

    Since the embeddings are L2-normalized, inner product == cosine similarity.
    This is dramatically faster than sklearn cosine_similarity over all chunks.

    Parameters
    ----------
    embeddings : np.ndarray
        Shape ``(n_chunks, dim)``, L2-normalized.

    Returns
    -------
    faiss.IndexFlatIP
        A FAISS index ready for querying.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def encode_query(model: SentenceTransformer, query: str) -> np.ndarray:
    """Encode a single query string into a normalized vector.

    Returns
    -------
    np.ndarray
        Shape ``(1, dim)`` suitable for FAISS search.
    """
    vec = model.encode([query], normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)
