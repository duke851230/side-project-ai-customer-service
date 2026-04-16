from __future__ import annotations

import os

ORDER_API_BASE_URL = os.getenv("ORDER_API_BASE_URL", "http://localhost:9000")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
INTENT_OLLAMA_MODEL = os.getenv("INTENT_OLLAMA_MODEL", OLLAMA_MODEL)
INTENT_LLM_TIMEOUT_SEC = float(os.getenv("INTENT_LLM_TIMEOUT_SEC", "45"))
INTENT_LLM_NUM_PREDICT = int(os.getenv("INTENT_LLM_NUM_PREDICT", "48"))
ANSWER_LLM_TIMEOUT_SEC = float(os.getenv("ANSWER_LLM_TIMEOUT_SEC", "120"))

RAG_INDEX_DIR = os.getenv("RAG_INDEX_DIR", "./data/index")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.45"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.55"))
FALLBACK_MESSAGE = "目前文件中沒有相關資訊"

# Comma-separated list, e.g. "http://localhost:5173,http://127.0.0.1:5173"
CORS_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
