from __future__ import annotations

import json
import logging
import re
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
FOLLOWUP_HINT_PATTERN = re.compile(
    r"(它|他|它們|他們|這個|那個|那可以|那怎麼)",
    re.IGNORECASE,
)


def _is_followup_like(text: str) -> bool:
    clean = text.strip()
    if not clean:
        return False
    return bool(FOLLOWUP_HINT_PATTERN.search(clean))


def _find_anchor_topic(user_turns: list[str]) -> str:
    """找最近一個「明確主題句」（非追問句）作為檢索錨點。"""
    # 由近到遠找第一個非追問句，避免連續追問時主題漂移。
    for turn in reversed(user_turns):
        if not _is_followup_like(turn):
            return turn
    # 若全是追問句，退回最近一句。
    return user_turns[-1] if user_turns else ""


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


def rewrite_query_with_history(message: str, history: list[dict[str, str]] | None = None) -> str:
    """將短追問改寫為可檢索查詢，補上最近主題，降低代詞造成的召回下降。"""
    clean_message = message.strip()
    if not clean_message or not history:
        return clean_message

    user_turns = [turn["text"].strip() for turn in history if turn.get("role") == "user" and turn.get("text")]
    if not user_turns:
        return clean_message

    # 排除與當前輸入完全相同的尾端 turn，避免重覆污染錨點挑選。
    while user_turns and user_turns[-1] == clean_message:
        user_turns.pop()
    if not user_turns:
        return clean_message

    # 僅在追問句時套用改寫，避免污染明確新問題。
    if not _is_followup_like(clean_message):
        return clean_message

    anchor_topic = _find_anchor_topic(user_turns)
    if not anchor_topic:
        return clean_message

    # 同時補上最近兩句 user 語境，降低單一錨點缺詞的風險。
    recent_context = " / ".join(user_turns[-2:])[:180]
    anchor_topic = anchor_topic[:120]
    rewritten = (
        f"{clean_message}\n"
        f"主題錨點：{anchor_topic}\n"
        f"近期語境：{recent_context}"
    )
    return rewritten.strip()
