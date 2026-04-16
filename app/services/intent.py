from __future__ import annotations

import json
import logging
import re
import time

import requests

from app.core.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_LLM_TIMEOUT_SEC,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from app.core.timing import elapsed_ms

logger = logging.getLogger("uvicorn.error")
ORDER_ID_REGEX = re.compile(r"\bORDER-\d+\b", re.IGNORECASE)


def classify_intent_with_ollama(message: str, request_id: str | None = None) -> tuple[str, float]:
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
        t0 = time.perf_counter()
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=INTENT_LLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        raw = str(resp.json().get("response", "")).strip()
        logger.info("[req=%s] intent_llm_ms=%.1f", request_id, elapsed_ms(t0))
    except Exception:
        logger.exception("[req=%s] intent_llm_failed", request_id)
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


def route_intent(message: str, request_id: str | None = None) -> tuple[str, float]:
    """根據使用者訊息判斷路由意圖，並回傳信心分數。"""
    has_order_id = bool(ORDER_ID_REGEX.search(message))
    if has_order_id:
        return "order_status", 1.0

    label, confidence = classify_intent_with_ollama(message, request_id=request_id)
    if confidence < INTENT_CONFIDENCE_THRESHOLD:
        return "clarify", confidence
    return label, confidence


def extract_order_id(message: str) -> str | None:
    """從訊息中擷取訂單編號（格式：ORDER-數字）。"""
    match = ORDER_ID_REGEX.search(message)
    if not match:
        return None
    return match.group(0).upper()
