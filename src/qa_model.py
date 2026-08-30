"""Extractive QA using Hugging Face Transformers with model caching."""

from __future__ import annotations

import time
from dataclasses import dataclass

from transformers import pipeline, Pipeline

from src.config import QA_MODELS, DEFAULT_QA_MODEL_KEY
from src.retriever import RetrievedChunk


@dataclass
class AnswerResult:
    """Container for a QA result with full provenance."""

    answer: str
    confidence: float         # model confidence score (0–1)
    source_chunks: list[RetrievedChunk]   # chunks used as context
    retrieval_latency_ms: float
    qa_latency_ms: float

    @property
    def total_latency_ms(self) -> float:
        return self.retrieval_latency_ms + self.qa_latency_ms


def load_qa_pipeline(model_key: str = DEFAULT_QA_MODEL_KEY) -> Pipeline:
    """Load a HuggingFace QA pipeline.

    In the Streamlit app this is wrapped with ``@st.cache_resource``
    so the model is loaded only once per model key.

    Parameters
    ----------
    model_key : str
        Key from :data:`src.config.QA_MODELS`.

    Returns
    -------
    Pipeline
        A ``question-answering`` pipeline.
    """
    model_name = QA_MODELS[model_key]
    return pipeline("question-answering", model=model_name)


def answer_question(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
    qa_pipe: Pipeline,
    retrieval_latency_s: float = 0.0,
) -> AnswerResult:
    """Run extractive QA over the retrieved context.

    Parameters
    ----------
    question : str
        The user's question.
    retrieved_chunks : list[RetrievedChunk]
        Chunks returned by the retriever.
    qa_pipe : Pipeline
        Pre-loaded QA pipeline.
    retrieval_latency_s : float
        Retrieval latency in seconds (for reporting).

    Returns
    -------
    AnswerResult
        Answer with confidence, source chunks, and timing info.
    """
    # Build context by joining retrieved chunk texts (highest-similarity first)
    context = " ".join(chunk.text for chunk in retrieved_chunks)

    start = time.perf_counter()
    result = qa_pipe(question=question, context=context)
    qa_latency = time.perf_counter() - start

    return AnswerResult(
        answer=result["answer"],
        confidence=result["score"],
        source_chunks=retrieved_chunks,
        retrieval_latency_ms=retrieval_latency_s * 1000,
        qa_latency_ms=qa_latency * 1000,
    )
