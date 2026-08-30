"""RAG-based PDF Question Answering System — Streamlit Application."""

import streamlit as st

from src.config import (
    EMBEDDING_MODEL,
    QA_MODELS,
    DEFAULT_QA_MODEL_KEY,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_TOP_K,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
    CHUNK_SIZE_STEP,
    MIN_CHUNK_OVERLAP,
    MAX_CHUNK_OVERLAP,
    CHUNK_OVERLAP_STEP,
    MIN_TOP_K,
    MAX_TOP_K,
)
from src.pdf_processor import process_pdf
from src.embeddings import load_embedding_model, embed_chunks, build_faiss_index
from src.retriever import retrieve
from src.qa_model import load_qa_pipeline, answer_question


# ── Cached model loaders (loaded once, shared across reruns) ──────────────

@st.cache_resource(show_spinner="Loading embedding model…")
def _get_embedding_model():
    return load_embedding_model(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Loading QA model…")
def _get_qa_pipeline(model_key: str):
    return load_qa_pipeline(model_key)


# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG QnA — PDF Question Answering",
    page_icon="📄",
    layout="wide",
)

st.title("📄 RAG-based PDF Question Answering")
st.caption(
    "Upload a PDF → extract & chunk text → embed with SentenceTransformers "
    "→ retrieve with FAISS → answer with BERT/DistilBERT"
)

# ── Sidebar controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📤 Upload PDF")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document to ask questions about its content.",
    )

    st.divider()
    st.subheader("🔧 Chunking")
    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=MIN_CHUNK_SIZE,
        max_value=MAX_CHUNK_SIZE,
        value=DEFAULT_CHUNK_SIZE,
        step=CHUNK_SIZE_STEP,
        help="Maximum number of characters per text chunk.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=MIN_CHUNK_OVERLAP,
        max_value=MAX_CHUNK_OVERLAP,
        value=DEFAULT_CHUNK_OVERLAP,
        step=CHUNK_OVERLAP_STEP,
        help="Number of overlapping characters between consecutive chunks.",
    )

    st.divider()
    st.subheader("🔍 Retrieval")
    top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=MIN_TOP_K,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        help="Number of most-similar chunks retrieved for each query.",
    )

    st.divider()
    st.subheader("🤖 QA Model")
    qa_model_key = st.selectbox(
        "Select QA model",
        options=list(QA_MODELS.keys()),
        index=list(QA_MODELS.keys()).index(DEFAULT_QA_MODEL_KEY),
        help="BERT-Large is more accurate but slower; DistilBERT is faster.",
    )

# ── Session state keys ───────────────────────────────────────────────────

_STATE_KEYS = ("chunks", "faiss_index", "pdf_name", "chunk_cfg")


def _chunking_changed() -> bool:
    """Check if chunking parameters have changed since last processing."""
    current_cfg = (chunk_size, chunk_overlap)
    return st.session_state.get("chunk_cfg") != current_cfg


# ── PDF processing ────────────────────────────────────────────────────────

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    pdf_name = uploaded_file.name
    need_reprocess = (
        st.session_state.get("pdf_name") != pdf_name
        or _chunking_changed()
    )

    if need_reprocess:
        with st.spinner(f"Processing **{pdf_name}**…"):
            chunks = process_pdf(pdf_bytes, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            embed_model = _get_embedding_model()
            embeddings = embed_chunks(embed_model, chunks, show_progress=False)
            faiss_index = build_faiss_index(embeddings)

            # Store in session state
            st.session_state["chunks"] = chunks
            st.session_state["faiss_index"] = faiss_index
            st.session_state["pdf_name"] = pdf_name
            st.session_state["chunk_cfg"] = (chunk_size, chunk_overlap)

        st.sidebar.success(f"✅ {len(chunks)} chunks indexed from {pdf_name}")

    # ── Info bar ──────────────────────────────────────────────────────────
    chunks = st.session_state["chunks"]
    col1, col2, col3 = st.columns(3)
    col1.metric("📑 Chunks", len(chunks))
    col2.metric("📏 Chunk Size", f"{chunk_size} chars")
    col3.metric("🔝 Top-K", top_k)

    # ── Question input ────────────────────────────────────────────────────
    st.divider()
    question = st.text_input(
        "❓ Ask a question about your PDF",
        placeholder="e.g. What is the basic unit of life?",
    )

    if question:
        embed_model = _get_embedding_model()
        faiss_index = st.session_state["faiss_index"]

        # Retrieve
        retrieved, ret_latency = retrieve(
            query=question,
            model=embed_model,
            index=faiss_index,
            chunks=chunks,
            top_k=top_k,
        )

        # Answer
        qa_pipe = _get_qa_pipeline(qa_model_key)
        result = answer_question(
            question=question,
            retrieved_chunks=retrieved,
            qa_pipe=qa_pipe,
            retrieval_latency_s=ret_latency,
        )

        # ── Answer card ──────────────────────────────────────────────────
        st.divider()
        st.subheader("💡 Answer")

        answer_col, metrics_col = st.columns([3, 1])
        with answer_col:
            st.markdown(f"### {result.answer}")
        with metrics_col:
            st.metric("Confidence", f"{result.confidence:.1%}")
            st.caption(f"Retrieval: {result.retrieval_latency_ms:.0f} ms")
            st.caption(f"QA: {result.qa_latency_ms:.0f} ms")
            st.caption(f"Total: {result.total_latency_ms:.0f} ms")

        # ── Source chunks ─────────────────────────────────────────────────
        st.divider()
        st.subheader("📚 Retrieved Source Chunks")
        st.caption(
            f"Top-{top_k} chunks retrieved by FAISS (cosine similarity). "
            "These chunks were used as context for the QA model."
        )

        for rc in result.source_chunks:
            with st.expander(
                f"**Chunk #{rc.chunk_index}** — Page {rc.source_page} "
                f"(similarity: {rc.score:.4f})",
                expanded=(rc.rank == 1),
            ):
                st.markdown(rc.text)

else:
    # ── Landing page ──────────────────────────────────────────────────────
    st.info(
        "👈 Upload a PDF in the sidebar to get started.\n\n"
        "**How it works:**\n"
        "1. Your PDF is split into overlapping text chunks\n"
        "2. Each chunk is embedded with SentenceTransformers\n"
        "3. A FAISS index enables fast similarity search\n"
        "4. Your question retrieves the most relevant chunks\n"
        "5. A BERT/DistilBERT model extracts the answer from those chunks"
    )
