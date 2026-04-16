from __future__ import annotations

import requests
from fastapi import HTTPException

from app.core.config import ORDER_API_BASE_URL


def fetch_order(order_id: str) -> dict[str, str]:
    """呼叫訂單 API 取得訂單資料。"""
    try:
        resp = requests.get(f"{ORDER_API_BASE_URL}/orders/{order_id}", timeout=3)
        resp.raise_for_status()
        return resp.json()
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise
        raise HTTPException(status_code=502, detail=f"order api unavailable: {exc}") from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"order api unavailable: {exc}") from exc
