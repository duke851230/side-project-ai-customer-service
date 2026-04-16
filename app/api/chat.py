from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import requests

from app.core.config import FALLBACK_MESSAGE, RAG_SCORE_THRESHOLD, RAG_TOP_K
from app.core.timing import elapsed_ms
from app.services.intent import extract_order_id, route_intent
from app.services.llm import generate_answer_with_ollama, postprocess_faq_answer
from app.services.order import fetch_order
from app.services.rag import retrieve_context

logger = logging.getLogger("uvicorn.error")
router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    route: str
    citations: list[str] = Field(default_factory=list)


def answer_faq_query(message: str, request_id: str | None = None) -> ChatResponse:
    """處理 FAQ 路由：檢索、生成回答、附上引用來源。"""
    try:
        t_retrieve = time.perf_counter()
        contexts = retrieve_context(message, RAG_TOP_K, request_id=request_id)
        logger.info("[req=%s] rag_total_ms=%.1f", request_id, elapsed_ms(t_retrieve))
    except FileNotFoundError:
        return ChatResponse(answer=f"{FALLBACK_MESSAGE}（尚未建立索引）", route="faq_query", citations=[])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {exc}") from exc

    if not contexts or contexts[0]["score"] < RAG_SCORE_THRESHOLD:
        return ChatResponse(answer=FALLBACK_MESSAGE, route="faq_query", citations=[])

    citations = [f"{c['source']}::{c['id']} (score={c['score']:.3f})" for c in contexts]

    try:
        answer = generate_answer_with_ollama(message, contexts, request_id=request_id)
    except Exception:
        # 若 Ollama 不可用，退回擷取式回答避免整體失敗。
        answer = contexts[0]["text"][:220]

    answer = postprocess_faq_answer(answer)
    return ChatResponse(answer=answer, route="faq_query", citations=citations)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> Any:
    """聊天主端點：依意圖分流到訂單查詢、FAQ RAG 或澄清分支。"""
    request_id = uuid.uuid4().hex[:8]
    req_start = time.perf_counter()
    logger.info("[req=%s] /chat start message_len=%d", request_id, len(payload.message))

    t_route = time.perf_counter()
    route, confidence = route_intent(payload.message, request_id=request_id)
    logger.info(
        "[req=%s] route=%s confidence=%.2f route_ms=%.1f",
        request_id,
        route,
        confidence,
        elapsed_ms(t_route),
    )

    if route == "clarify":
        logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
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
            t_order = time.perf_counter()
            order = fetch_order(order_id)
            logger.info("[req=%s] order_api_ms=%.1f", request_id, elapsed_ms(t_order))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
                return ChatResponse(
                    answer=f"查無訂單 {order_id}，請確認編號是否正確。",
                    route=route,
                    citations=[],
                )
            raise

        logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
        return ChatResponse(
            answer=f"訂單 {order['order_id']} 目前狀態為 {order['status']}，最後更新時間 {order['updated_at']}。",
            route=route,
            citations=[],
        )

    result = answer_faq_query(payload.message, request_id=request_id)
    logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
    return result
