from __future__ import annotations

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.services.rag import preload_rag_resources

app = FastAPI(title="Local AI Customer Support API")
app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    """健康檢查端點，回傳服務可用狀態。"""
    return {"status": "ok"}


@app.on_event("startup")
def startup() -> None:
    preload_rag_resources()
