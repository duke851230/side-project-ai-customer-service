from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import requests

from app.core.config import (
    FALLBACK_MESSAGE,
    RAG_SCORE_THRESHOLD,
    RAG_TOP_K,
    SESSION_HISTORY_TURN_LIMIT,
    SESSION_TTL_SEC,
)
from app.core.timing import elapsed_ms
from app.services.intent import extract_order_id, route_intent
from app.services.llm import generate_answer_with_ollama, postprocess_faq_answer
from app.services.memory import append_session_turn, get_session_history
from app.services.order import fetch_order
from app.services.rag import retrieve_context, rewrite_query_with_history

logger = logging.getLogger("uvicorn.error")
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str
    session_id: str = ""
    citations: list[str] = Field(default_factory=list)


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{8,128}$")


def _resolve_session_id(client_session_id: str | None) -> str:
    if client_session_id:
        session_id = client_session_id.strip()
        if SESSION_ID_PATTERN.match(session_id):
            return session_id
    return uuid.uuid4().hex


def _find_recent_order_id(history: list[dict[str, str]]) -> str | None:
    for turn in reversed(history):
        if turn["role"] != "user":
            continue
        maybe_order_id = extract_order_id(turn["text"])
        if maybe_order_id:
            return maybe_order_id
    return None


def answer_faq_query(
    message: str,
    history: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> ChatResponse:
    """處理 FAQ 路由：檢索、生成回答、附上引用來源。"""
    try:
        retrieve_query = rewrite_query_with_history(message, history)
        if retrieve_query != message:
            logger.info(
                "[req=%s] rag_query_rewritten original=%r rewritten=%r",
                request_id,
                message[:120],
                retrieve_query[:220],
            )

        t_retrieve = time.perf_counter()
        contexts = retrieve_context(retrieve_query, RAG_TOP_K, request_id=request_id)
        logger.info("[req=%s] rag_total_ms=%.1f", request_id, elapsed_ms(t_retrieve))
    except FileNotFoundError:
        return ChatResponse(answer=f"{FALLBACK_MESSAGE}（尚未建立索引）", route="faq_query", citations=[])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RAG retrieval failed: {exc}") from exc

    if not contexts or contexts[0]["score"] < RAG_SCORE_THRESHOLD:
        return ChatResponse(answer=FALLBACK_MESSAGE, route="faq_query", citations=[])

    citations = [f"{c['source']}::{c['id']} (score={c['score']:.3f})" for c in contexts]

    try:
        answer = generate_answer_with_ollama(message, contexts, history=history, request_id=request_id)
    except Exception:
        # 若 Ollama 不可用，退回擷取式回答避免整體失敗。
        answer = contexts[0]["text"][:220]

    answer = postprocess_faq_answer(answer)
    return ChatResponse(answer=answer, route="faq_query", citations=citations)


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> Any:
    """聊天主端點：依意圖分流到訂單查詢、FAQ RAG 或澄清分支。"""
    request_id = uuid.uuid4().hex[:8]
    session_id = _resolve_session_id(payload.session_id)
    req_start = time.perf_counter()
    logger.info(
        "[req=%s] /chat start session=%s message_len=%d",
        request_id,
        session_id,
        len(payload.message),
    )

    try:
        history = get_session_history(session_id)
    except Exception:
        logger.exception("[req=%s] failed_to_load_history session=%s", request_id, session_id)
        history = []

    t_route = time.perf_counter()
    route, confidence = route_intent(payload.message, history=history, request_id=request_id)
    logger.info(
        "[req=%s] route=%s confidence=%.2f route_ms=%.1f",
        request_id,
        route,
        confidence,
        elapsed_ms(t_route),
    )

    result: ChatResponse
    if route == "clarify":
        result = ChatResponse(
            answer=(
                "我可以幫你查訂單狀態或回答文件規則。"
                f"（目前判斷信心 {confidence:.2f}）請告訴我你要先處理哪一項。"
            ),
            route=route,
            session_id=session_id,
            citations=[],
        )
        _save_turns(session_id, payload.message, result.answer, request_id=request_id)
        logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
        return result

    if route == "order_status":
        order_id = extract_order_id(payload.message)
        if not order_id:
            order_id = _find_recent_order_id(history)
        if not order_id:
            result = ChatResponse(
                answer="請提供訂單編號（例如 ORDER-001），我才能幫你查詢狀態。",
                route="clarify",
                session_id=session_id,
                citations=[],
            )
            _save_turns(session_id, payload.message, result.answer, request_id=request_id)
            logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
            return result

        try:
            t_order = time.perf_counter()
            order = fetch_order(order_id)
            logger.info("[req=%s] order_api_ms=%.1f", request_id, elapsed_ms(t_order))
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
                result = ChatResponse(
                    answer=f"查無訂單 {order_id}，請確認編號是否正確。",
                    route=route,
                    session_id=session_id,
                    citations=[],
                )
                _save_turns(session_id, payload.message, result.answer, request_id=request_id)
                return result
            raise

        result = ChatResponse(
            answer=f"訂單 {order['order_id']} 目前狀態為 {order['status']}，最後更新時間 {order['updated_at']}。",
            route=route,
            session_id=session_id,
            citations=[],
        )
        _save_turns(session_id, payload.message, result.answer, request_id=request_id)
        logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
        return result

    result = answer_faq_query(payload.message, history=history, request_id=request_id)
    result.session_id = session_id
    _save_turns(session_id, payload.message, result.answer, request_id=request_id)
    logger.info("[req=%s] /chat done total_ms=%.1f", request_id, elapsed_ms(req_start))
    return result


def _save_turns(session_id: str, user_text: str, assistant_text: str, request_id: str | None = None) -> None:
    try:
        append_session_turn(
            session_id,
            "user",
            user_text,
            history_turn_limit=SESSION_HISTORY_TURN_LIMIT,
            ttl_sec=SESSION_TTL_SEC,
        )
        append_session_turn(
            session_id,
            "assistant",
            assistant_text,
            history_turn_limit=SESSION_HISTORY_TURN_LIMIT,
            ttl_sec=SESSION_TTL_SEC,
        )
    except Exception:
        logger.exception("[req=%s] failed_to_save_history session=%s", request_id, session_id)
