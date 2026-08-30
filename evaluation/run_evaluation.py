"""Retrieval evaluation: Recall@k across chunk-size top-k configurations.

Run from the project root:
    python -m evaluation.run_evaluation

Outputs:
    evaluation/results/recall_results.csv   - per-question results
    evaluation/results/recall_summary.csv   - aggregated Recall@k per config
    evaluation/results/recall_heatmap.png   - visual heatmap
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is on sys.path when running as a module
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDING_MODEL,
    EVAL_CHUNK_SIZES,
    EVAL_TOP_K_VALUES,
    EVAL_DIR,
    EVAL_RESULTS_DIR,
)
from src.embeddings import load_embedding_model, embed_chunks, build_faiss_index
from src.retriever import retrieve
from src.pdf_processor import Chunk


# Helpers
def load_eval_dataset(path: Path | None = None) -> list[dict]:
    """Load the evaluation dataset from JSON."""
    path = path or (EVAL_DIR / "eval_dataset.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text_from_file(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """Chunk raw text (for the .txt sample data) into Chunk objects."""
    chunks: list[Chunk] = []
    stride = max(chunk_size - chunk_overlap, 1)
    idx = 0
    chunk_index = 0
    while idx < len(text):
        end = min(idx + chunk_size, len(text))
        chunk_text = text[idx:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source_page=0,
                chunk_index=chunk_index,
                start_char=idx,
            ))
            chunk_index += 1
        idx += stride
    return chunks


def is_relevant(chunk_text: str, keywords: list[str]) -> bool:
    """Check if a chunk is relevant by verifying keyword presence.

    A chunk is considered relevant if it contains ALL of the specified keywords
    (case-insensitive).
    """
    lower = chunk_text.lower()
    return all(kw.lower() in lower for kw in keywords)


def compute_recall_at_k(
    retrieved_chunks: list,
    keywords: list[str],
    k: int,
) -> bool:
    """Check if at least one relevant chunk appears in the top-k results."""
    for chunk in retrieved_chunks[:k]:
        if is_relevant(chunk.text, keywords):
            return True
    return False


# Main evaluation loop

def run_evaluation():
    """Run the full evaluation grid and save results."""
    print("=" * 70)
    print("RAG Retrieval Evaluation")
    print("=" * 70)

    # Load data
    txt_path = PROJECT_ROOT / "ConceptsofBiology-WEB.txt"
    if not txt_path.exists():
        txt_path = PROJECT_ROOT / "sample_data" / "ConceptsofBiology-WEB.txt"
    if not txt_path.exists():
        print(f"ERROR: Cannot find textbook at {txt_path}")
        sys.exit(1)

    print(f"Loading textbook from: {txt_path}")
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    print(f"  -> {len(text):,} characters")

    eval_data = load_eval_dataset()
    print(f"Loaded {len(eval_data)} evaluation questions")

    # Load embedding model once
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = load_embedding_model(EMBEDDING_MODEL)
    print("  -> Model loaded")

    all_results: list[dict] = []
    summary_rows: list[dict] = []

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    max_k = max(EVAL_TOP_K_VALUES)

    for chunk_size in EVAL_CHUNK_SIZES:
        overlap = min(50, chunk_size // 5)
        print(f"\n{'-' * 60}")
        print(f"Chunking: size={chunk_size}, overlap={overlap}")
        chunks = chunk_text_from_file(text, chunk_size, overlap)
        print(f"  -> {len(chunks)} chunks")

        print("  Embedding chunks...")
        t0 = time.perf_counter()
        embeddings = embed_chunks(model, chunks, batch_size=128, show_progress=False)
        embed_time = time.perf_counter() - t0
        print(f"  -> Embedded in {embed_time:.1f}s")

        index = build_faiss_index(embeddings)

        # 1. Retrieve top max_k results once per question
        retrieved_per_question = []
        for item in eval_data:
            retrieved, _ = retrieve(
                query=item["question"],
                model=model,
                index=index,
                chunks=chunks,
                top_k=max_k,
            )
            retrieved_per_question.append((item, retrieved))

        # 2. Evaluate Recall@k for each k in EVAL_TOP_K_VALUES
        for k in EVAL_TOP_K_VALUES:
            hits = 0
            for item, retrieved in retrieved_per_question:
                kws = item["relevant_chunk_keywords"]
                is_hit = compute_recall_at_k(retrieved, kws, k=k)
                if is_hit:
                    hits += 1

                all_results.append({
                    "chunk_size": chunk_size,
                    "k": k,
                    "question_id": item["id"],
                    "question": item["question"],
                    "difficulty": item["difficulty"],
                    "topic": item["topic"],
                    "hit": int(is_hit),
                })

            recall_k = hits / len(eval_data)
            summary_rows.append({
                "chunk_size": chunk_size,
                "k": k,
                "num_chunks": len(chunks),
                "recall": recall_k,
            })
            print(f"    Recall@{k} = {recall_k:.2%}")

    # Save results
    df_all = pd.DataFrame(all_results)
    df_summary = pd.DataFrame(summary_rows)

    csv_all = EVAL_RESULTS_DIR / "recall_results.csv"
    csv_summary = EVAL_RESULTS_DIR / "recall_summary.csv"
    df_all.to_csv(csv_all, index=False)
    df_summary.to_csv(csv_summary, index=False)
    print(f"\n Saved per-question results to: {csv_all}")
    print(f" Saved summary to: {csv_summary}")

    # Best configuration
    best_idx = df_summary["recall"].idxmax()
    best = df_summary.iloc[best_idx]
    print(f"\n{'=' * 70}")
    print(" Best Configuration:")
    print(f"   chunk_size = {int(best['chunk_size'])}")
    print(f"   Top-K (k)  = {int(best['k'])}")
    print(f"   Recall@{int(best['k'])} = {best['recall']:.2%}")
    print(f"{'=' * 70}")

    # Heatmap
    _plot_heatmaps(df_summary)

    return df_summary


def _plot_heatmaps(df_summary: pd.DataFrame) -> None:
    """Generate a single Recall@k heatmap (Chunk Size vs Top-K)."""
    fig, ax = plt.subplots(figsize=(8, 6))

    pivot = df_summary.pivot(
        index="chunk_size", columns="k", values="recall"
    )

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0%",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Recall@k"},
    )

    ax.set_title("Retrieval Recall@k — Chunk Size vs Top-K", fontsize=14, fontweight="bold")
    ax.set_xlabel("Top-K (k)")
    ax.set_ylabel("Chunk Size (chars)")

    plt.tight_layout()
    out = EVAL_RESULTS_DIR / "recall_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" Saved heatmap to: {out}")


if __name__ == "__main__":
    run_evaluation()

