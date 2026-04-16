from __future__ import annotations

import json
import logging
from typing import Any

import redis

from app.core.config import REDIS_URL

logger = logging.getLogger("uvicorn.error")
_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _session_key(session_id: str) -> str:
    return f"chat:session:{session_id}:turns"


def get_session_history(session_id: str) -> list[dict[str, str]]:
    """讀取 session 內的對話歷史（依時間正序）。"""
    client = _get_redis()
    key = _session_key(session_id)
    rows = client.lrange(key, 0, -1)

    history: list[dict[str, str]] = []
    for row in rows:
        try:
            item = json.loads(row)
        except Exception:
            continue
        role = str(item.get("role", "")).strip()
        text = str(item.get("text", "")).strip()
        if role in {"user", "assistant"} and text:
            history.append({"role": role, "text": text})
    return history


def append_session_turn(
    session_id: str,
    role: str,
    text: str,
    *,
    history_turn_limit: int,
    ttl_sec: int,
) -> None:
    """寫入一筆對話，並限制最多保存最近 N 輪（user+assistant）。"""
    if role not in {"user", "assistant"} or not text.strip():
        return

    client = _get_redis()
    key = _session_key(session_id)
    payload = json.dumps({"role": role, "text": text}, ensure_ascii=False)

    max_items = max(2, history_turn_limit * 2)
    pipe = client.pipeline()
    pipe.rpush(key, payload)
    pipe.ltrim(key, -max_items, -1)
    pipe.expire(key, max(60, ttl_sec))
    pipe.execute()


def check_memory_backend() -> dict[str, Any]:
    """提供健康檢查資訊，便於 startup 日誌觀測。"""
    client = _get_redis()
    pong = client.ping()
    return {"redis": bool(pong), "url": REDIS_URL}
