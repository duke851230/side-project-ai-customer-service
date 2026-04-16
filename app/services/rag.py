from __future__ import annotations

import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import EMBEDDING_MODEL, RAG_INDEX_DIR
from app.core.timing import elapsed_ms

logger = logging.getLogger("uvicorn.error")


@lru_cache(maxsize=1)
def get_rag_resources() -> tuple[faiss.Index, list[dict[str, str]], SentenceTransformer]:
    """載入 FAISS 索引、metadata 與 embedding 模型（程序內快取）。"""
    index_dir = Path(RAG_INDEX_DIR)
    metadata_path = index_dir / "metadata.json"
    index_path = index_dir / "faiss.index"

    if not metadata_path.exists() or not index_path.exists():
        raise FileNotFoundError(f"RAG index not found under: {index_dir}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    return index, metadata, embedder


def preload_rag_resources() -> None:
    """啟動時預載入 RAG 資源，降低首請求冷啟動延遲。"""
    t0 = time.perf_counter()
    try:
        get_rag_resources()
        logger.info("[startup] rag_preload_ms=%.1f", elapsed_ms(t0))
    except Exception:
        logger.exception("[startup] rag_preload_failed_ms=%.1f", elapsed_ms(t0))


def retrieve_context(question: str, top_k: int, request_id: str | None = None) -> list[dict[str, Any]]:
    """從 FAISS 取回 top-k 片段，分數使用內積（近似 cosine）。"""
    index, metadata, embedder = get_rag_resources()
    t_encode = time.perf_counter()
    query_vec = embedder.encode([question], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")
    encode_ms = elapsed_ms(t_encode)

    t_search = time.perf_counter()
    scores, indices = index.search(query_vec, top_k)
    search_ms = elapsed_ms(t_search)
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = metadata[idx]
        results.append(
            {
                "score": float(score),
                "source": item.get("source", ""),
                "id": item.get("id", f"chunk-{idx}"),
                "text": item.get("text", ""),
            }
        )
    logger.info(
        "[req=%s] rag_encode_ms=%.1f rag_search_ms=%.1f rag_hits=%d top_score=%.3f",
        request_id,
        encode_ms,
        search_ms,
        len(results),
        (results[0]["score"] if results else -1.0),
    )
    return results
