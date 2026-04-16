from __future__ import annotations

import logging
import re
import time
from typing import Any

import requests

from app.core.config import ANSWER_LLM_TIMEOUT_SEC, FALLBACK_MESSAGE, OLLAMA_BASE_URL, OLLAMA_MODEL
from app.core.timing import elapsed_ms

logger = logging.getLogger("uvicorn.error")


def generate_answer_with_ollama(
    question: str,
    contexts: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> str:
    """使用本地 Ollama 根據檢索片段產生有依據的回答。"""
    context_text = "\n\n".join(
        [f"[{i + 1}] ({c['source']}::{c['id']})\n{c['text']}" for i, c in enumerate(contexts)]
    )
    history_text = ""
    if history:
        lines: list[str] = []
        for turn in history[-12:]:
            role = "使用者" if turn["role"] == "user" else "助理"
            lines.append(f"{role}：{turn['text']}")
        history_text = "\n".join(lines)

    prompt = (
        "你是客服助理。只能根據提供的內容回答，不可臆測。\n"
        f"若資訊不足，請只回覆：{FALLBACK_MESSAGE}\n\n"
        f"最近對話（可用於理解代詞與追問語境）：\n{history_text or '（無）'}\n\n"
        f"問題：{question}\n\n"
        f"內容：\n{context_text}\n\n"
        "請用繁體中文客服口吻回答，語氣自然、簡潔、好懂。\n"
        "格式要求：\n"
        "1) 先用一句話直接回答問題（可明確回答是/否/條件）。\n"
        "2) 再補 1-2 句說明條件、限制或例外。\n"
        "3) 不要使用條列或 Markdown 標題，不要複製原文標題。\n"
        "4) 避免過度生硬或法條式措辭。"
    )

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=ANSWER_LLM_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        data = resp.json()
        logger.info("[req=%s] answer_llm_ms=%.1f", request_id, elapsed_ms(t0))
        return str(data.get("response", "")).strip() or FALLBACK_MESSAGE
    except Exception:
        logger.exception("[req=%s] answer_llm_failed_ms=%.1f", request_id, elapsed_ms(t0))
        raise


def postprocess_faq_answer(answer: str) -> str:
    """清理模型輸出，避免原文標題樣式直接外漏。"""
    cleaned = answer.strip()
    cleaned = re.sub(r"^\s*#+\s*", "", cleaned)
    return cleaned
