from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.chat import router as chat_router
from app.core.config import CORS_ALLOW_ORIGINS
from app.services.rag import preload_rag_resources

app = FastAPI(title="Local AI Customer Support API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/health")
def health() -> dict[str, str]:
    """健康檢查端點，回傳服務可用狀態。"""
    return {"status": "ok"}


@app.on_event("startup")
def startup() -> None:
    preload_rag_resources()
