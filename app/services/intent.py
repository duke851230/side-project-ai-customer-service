from __future__ import annotations

import json
import logging
import re
import time

import requests

from app.core.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    INTENT_LLM_NUM_PREDICT,
    INTENT_LLM_TIMEOUT_SEC,
    INTENT_OLLAMA_MODEL,
    OLLAMA_BASE_URL,
)
from app.core.timing import elapsed_ms

logger = logging.getLogger("uvicorn.error")
ORDER_ID_REGEX = re.compile(r"\bORDER-\d+\b", re.IGNORECASE)
LABEL_REGEX = re.compile(r"\b(order_status|faq_query|clarify)\b", re.IGNORECASE)
CONFIDENCE_REGEX = re.compile(r'confidence"?\s*:\s*([01](?:\.\d+)?)', re.IGNORECASE)


def _parse_intent_response(raw: str) -> tuple[str, float]:
    """解析模型輸出，盡量從非嚴格 JSON 中恢復 label/confidence。"""
    label = "clarify"
    confidence = 0.0

    # 先嘗試嚴格 JSON 解析。
    try:
        parsed = json.loads(raw)
        label = str(parsed.get("label", "")).strip().lower()
        confidence = float(parsed.get("confidence", 0.0))
    except Exception:
        # 再嘗試從文字中擷取 label 與 confidence，容忍部分格式錯誤。
        label_match = LABEL_REGEX.search(raw)
        if label_match:
            label = label_match.group(1).lower()

        confidence_match = CONFIDENCE_REGEX.search(raw)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
            except ValueError:
                confidence = 0.0

    if label not in {"order_status", "faq_query", "clarify"}:
        label = "clarify"
    confidence = max(0.0, min(1.0, confidence))
    return label, confidence


def _resolve_intent_with_llm(
    message: str,
    history: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> tuple[str, float]:
    """二次判斷：強制 LLM 在 order_status / faq_query 中擇一。"""
    prompt = (
        "你是意圖仲裁器。只能回傳 JSON，格式為 "
        '{"label":"order_status|faq_query","confidence":0~1}。\n'
        "判斷規則：\n"
        "- 訂單狀態、物流進度、查貨 -> order_status\n"
        "- 產品政策、FAQ、規則說明 -> faq_query\n"
        "- 即使語意不清，也要選最可能的一個，不可輸出 clarify。\n\n"
        f"最近對話：{_history_to_text(history)}\n"
        f"使用者訊息：{message}\n"
        "只輸出 JSON，不要其他文字。"
    )

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": INTENT_OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "num_predict": INTENT_LLM_NUM_PREDICT,
                    "temperature": 0,
                },
            },
            timeout=INTENT_LLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        raw = str(resp.json().get("response", "")).strip()
        logger.info("[req=%s] intent_resolve_llm_ms=%.1f", request_id, elapsed_ms(t0))
    except Exception:
        logger.exception("[req=%s] intent_resolve_llm_failed", request_id)
        return "clarify", 0.0

    label, confidence = _parse_intent_response(raw)
    if label in {"order_status", "faq_query"}:
        return label, confidence
    logger.warning("[req=%s] intent_resolve_unexpected raw=%r", request_id, raw[:200])
    return "clarify", confidence


def classify_intent_with_ollama(
    message: str,
    history: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> tuple[str, float]:
    """使用 Ollama 做意圖分類，回傳固定標籤與信心分數。"""
    prompt = (
        "你是意圖分類器。請回傳 JSON，格式為 "
        '{"label":"order_status|faq_query|clarify","confidence":0~1}。\n'
        "判斷規則：\n"
        "- 訂單狀態、物流進度、查貨 -> order_status\n"
        "- 產品政策、FAQ、規則說明 -> faq_query\n"
        "- 同時包含多種需求或語意不清 -> clarify\n\n"
        f"最近對話：{_history_to_text(history)}\n"
        f"使用者訊息：{message}\n"
        "只輸出 JSON，不要其他文字。"
    )

    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": INTENT_OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "num_predict": INTENT_LLM_NUM_PREDICT,
                    "temperature": 0,
                },
            },
            timeout=INTENT_LLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        raw = str(resp.json().get("response", "")).strip()
        logger.info("[req=%s] intent_llm_ms=%.1f", request_id, elapsed_ms(t0))
    except Exception:
        logger.exception("[req=%s] intent_llm_failed", request_id)
        return "clarify", 0.0

    label, confidence = _parse_intent_response(raw)
    if label == "clarify" and confidence == 0.0:
        logger.warning("[req=%s] intent_parse_low_conf raw=%r", request_id, raw[:200])

    if label in {"order_status", "faq_query", "clarify"}:
        return label, max(0.0, min(1.0, confidence))
    return "clarify", 0.0


def route_intent(
    message: str,
    history: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> tuple[str, float]:
    """根據使用者訊息判斷路由意圖，並回傳信心分數。"""
    has_order_id = bool(ORDER_ID_REGEX.search(message))
    if has_order_id:
        return "order_status", 1.0

    label, confidence = classify_intent_with_ollama(message, history=history, request_id=request_id)
    if label != "clarify" and confidence >= INTENT_CONFIDENCE_THRESHOLD:
        return label, confidence

    # 低信心或模型回 clarify 時，由 LLM 再做一次二選一仲裁。
    resolve_label, resolve_confidence = _resolve_intent_with_llm(
        message,
        history=history,
        request_id=request_id,
    )
    if resolve_label in {"order_status", "faq_query"}:
        return resolve_label, max(confidence, resolve_confidence)
    return "clarify", max(confidence, resolve_confidence)


def extract_order_id(message: str) -> str | None:
    """從訊息中擷取訂單編號（格式：ORDER-數字）。"""
    match = ORDER_ID_REGEX.search(message)
    if not match:
        return None
    return match.group(0).upper()


def _history_to_text(history: list[dict[str, str]] | None) -> str:
    if not history:
        return "（無）"
    lines: list[str] = []
    for turn in history[-12:]:
        role = "user" if turn["role"] == "user" else "assistant"
        lines.append(f"{role}: {turn['text']}")
    return " | ".join(lines)
