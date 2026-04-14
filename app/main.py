from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Local AI Customer Support API")
ORDER_API_BASE_URL = os.getenv("ORDER_API_BASE_URL", "http://localhost:9000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "./data/index")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.45"))
INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.7"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
FALLBACK_MESSAGE = "目前文件中沒有相關資訊"
ORDER_ID_REGEX = re.compile(r"\bORDER-\d+\b", re.IGNORECASE)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    citations: list[str] = Field(default_factory=list)


def route_intent(message: str) -> tuple[str, float]:
    """根據使用者訊息判斷路由意圖，並回傳信心分數。

    規則僅保留高確定性的訂單編號判斷，其餘交由 LLM 分類。
    """
    has_order_id = bool(ORDER_ID_REGEX.search(message))

    if has_order_id:
        return "order_status", 1.0

    label, confidence = classify_intent_with_ollama(message)
    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        return "clarify", confidence
    return label, confidence


def classify_intent_with_ollama(message: str) -> tuple[str, float]:
    """使用 Ollama 做意圖分類，回傳固定標籤與信心分數。"""
    prompt = (
        "你是意圖分類器。請回傳 JSON，格式為 "
        '{"label":"order_status|faq_query|clarify","confidence":0~1}。\n'
        "判斷規則：\n"
        "- 訂單狀態、物流進度、查貨 -> order_status\n"
        "- 產品政策、FAQ、規則說明 -> faq_query\n"
        "- 同時包含多種需求或語意不清 -> clarify\n\n"
        f"使用者訊息：{message}\n"
        "只輸出 JSON，不要其他文字。"
    )

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=8,
        )
        resp.raise_for_status()
        raw = str(resp.json().get("response", "")).strip()
    except Exception:
        return "clarify", 0.0

    label = "clarify"
    confidence = 0.0

    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        lowered = raw.lower()
        if "order_status" in lowered:
            label = "order_status"
        elif "faq_query" in lowered:
            label = "faq_query"
        elif "clarify" in lowered:
            label = "clarify"
        confidence = 0.0

    if label in {"order_status", "faq_query", "clarify"}:
        return label, max(0.0, min(1.0, confidence))
    return "clarify", 0.0


def extract_order_id(message: str) -> str | None:
    """從訊息中擷取訂單編號（格式：ORDER-數字）。"""
    match = ORDER_ID_REGEX.search(message)
    if not match:
        return None
    return match.group(0).upper()


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


def retrieve_context(question: str, top_k: int) -> list[dict[str, Any]]:
    """從 FAISS 取回 top-k 片段，分數使用內積（近似 cosine）。"""
    index, metadata, embedder = get_rag_resources()
    query_vec = embedder.encode([question], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, top_k)
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
    return results


def generate_answer_with_ollama(question: str, contexts: list[dict[str, Any]]) -> str:
    """使用本地 Ollama 根據檢索片段產生有依據的回答。"""
    context_text = "\n\n".join(
        [f"[{i + 1}] ({c['source']}::{c['id']})\n{c['text']}" for i, c in enumerate(contexts)]
    )
    prompt = (
        "你是客服助理。只能根據提供的內容回答，不可臆測。\n"
        f"若資訊不足，請只回覆：{FALLBACK_MESSAGE}\n\n"
        f"問題：{question}\n\n"
        f"內容：\n{context_text}\n\n"
        "請用繁體中文簡潔回答。"
    )

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response", "")).strip() or FALLBACK_MESSAGE


def answer_faq_query(message: str) -> ChatResponse:
    """處理 FAQ 路由：檢索、生成回答、附上引用來源。"""
    try:
        contexts = retrieve_context(message, RAG_TOP_K)
    except FileNotFoundError:
        return ChatResponse(answer=f"{FALLBACK_MESSAGE}（尚未建立索引）", route="faq_query", citations=[])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {exc}") from exc

    if not contexts or contexts[0]["score"] < RAG_SCORE_THRESHOLD:
        return ChatResponse(answer=FALLBACK_MESSAGE, route="faq_query", citations=[])

    citations = [f"{c['source']}::{c['id']} (score={c['score']:.3f})" for c in contexts]

    try:
        answer = generate_answer_with_ollama(message, contexts)
    except Exception:
        # 若 Ollama 不可用，退回擷取式回答避免整體失敗。
        answer = contexts[0]["text"][:220]

    return ChatResponse(answer=answer, route="faq_query", citations=citations)


@app.get("/health")
def health() -> dict[str, str]:
    """健康檢查端點，回傳服務可用狀態。"""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> Any:
    """聊天主端點：依意圖分流到訂單查詢、FAQ RAG 或澄清分支。"""
    route, confidence = route_intent(payload.message)

    if route == "clarify":
        return ChatResponse(
            answer=(
                "我可以幫你查訂單狀態或回答文件規則。"
                f"（目前判斷信心 {confidence:.2f}）請告訴我你要先處理哪一項。"
            ),
            route=route,
            citations=[],
        )

    if route == "order_status":
        order_id = extract_order_id(payload.message)
        if not order_id:
            return ChatResponse(
                answer="請提供訂單編號（例如 ORDER-001），我才能幫你查詢狀態。",
                route="clarify",
                citations=[],
            )

        try:
            resp = requests.get(f"{ORDER_API_BASE_URL}/orders/{order_id}", timeout=3)
            resp.raise_for_status()
            order = resp.json()
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return ChatResponse(
                    answer=f"查無訂單 {order_id}，請確認編號是否正確。",
                    route=route,
                    citations=[],
                )
            raise HTTPException(status_code=502, detail=f"order api unavailable: {exc}") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"order api unavailable: {exc}") from exc

        return ChatResponse(
            answer=f"訂單 {order['order_id']} 目前狀態為 {order['status']}，最後更新時間 {order['updated_at']}。",
            route=route,
            citations=[],
        )

    return answer_faq_query(payload.message)
