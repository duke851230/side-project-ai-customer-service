from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def read_text(file_path: Path) -> str:
    """讀取支援格式的檔案內容並回傳純文字。

    支援格式：
    - `.pdf`：逐頁擷取文字並串接
    - `.txt`、`.md`：以 UTF-8 讀取
    """
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return file_path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int, overlap: int) -> Iterable[str]:
    """將文字切成有重疊區間的字元片段。

    先做空白正規化，再依 `chunk_size` 切分，
    每段保留 `overlap` 以降低上下文斷裂。
    """
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def main() -> None:
    """從原始文件建立本地 FAISS 索引與 metadata。

    流程：
    1. 從 `--input` 載入支援的文件
    2. 切 chunk
    3. 產生 embedding
    4. 將向量索引與 chunk metadata 寫入 `--index-dir`
    """
    parser = argparse.ArgumentParser(description="Build local FAISS index from docs")
    parser.add_argument("--input", required=True, help="Directory containing txt/md/pdf files")
    parser.add_argument("--index-dir", default="./data/index", help="Output index directory")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    input_dir = Path(args.input)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    if not files:
        raise SystemExit(f"No supported files found under: {input_dir}")

    docs: list[dict[str, str]] = []
    for file_path in files:
        text = read_text(file_path)
        for idx, chunk in enumerate(chunk_text(text, args.chunk_size, args.overlap)):
            docs.append({"id": f"{file_path.name}::chunk-{idx}", "source": str(file_path), "text": chunk})

    if not docs:
        raise SystemExit("No chunks generated. Check file contents.")

    model = SentenceTransformer(args.embedding_model)
    vectors = model.encode([d["text"] for d in docs], normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, str(index_dir / "faiss.index"))
    (index_dir / "metadata.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Indexed {len(docs)} chunks from {len(files)} files into {index_dir}")


if __name__ == "__main__":
    main()
