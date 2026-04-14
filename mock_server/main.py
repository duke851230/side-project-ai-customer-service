from fastapi import FastAPI, HTTPException

app = FastAPI(title="Mock Order API")

MOCK_ORDERS = {
    "ORDER-001": {"order_id": "ORDER-001", "status": "shipped", "updated_at": "2026-04-13T09:00:00+08:00"},
    "ORDER-002": {"order_id": "ORDER-002", "status": "processing", "updated_at": "2026-04-13T10:30:00+08:00"},
}


@app.get("/health")
def health() -> dict[str, str]:
    """健康檢查端點，確認 mock API 正常運作。"""
    return {"status": "ok"}


@app.get("/orders/{order_id}")
def get_order(order_id: str) -> dict[str, str]:
    """依訂單編號回傳 mock 訂單狀態。"""
    if order_id not in MOCK_ORDERS:
        raise HTTPException(status_code=404, detail="order not found")
    return MOCK_ORDERS[order_id]
