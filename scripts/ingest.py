from __future__ import annotations

import argparse
import json
import re
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
    """將文字切成有重疊區間的片段，盡量保留 Markdown 結構。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    def split_long_block(block: str) -> list[str]:
        """將超長段落依句子邊界切分；若仍過長，再退化成字元切分。"""
        if len(block) <= chunk_size:
            return [block]

        sentence_like = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?；;])\s+", block)
            if part.strip()
        ]
        if len(sentence_like) <= 1:
            return [block[i : i + chunk_size] for i in range(0, len(block), chunk_size)]

        result: list[str] = []
        current = ""
        for unit in sentence_like:
            candidate = f"{current} {unit}".strip() if current else unit
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                result.append(current)
                current = unit
            else:
                result.extend([unit[i : i + chunk_size] for i in range(0, len(unit), chunk_size)])
                current = ""
        if current:
            result.append(current)
        return result

    def split_by_markdown_headings(value: str) -> tuple[list[str], bool]:
        """以 Markdown 標題切分；回傳 sections 與是否成功偵測標題。"""
        lines = value.split("\n")
        sections: list[str] = []
        current: list[str] = []

        for line in lines:
            if re.match(r"^\s{0,3}#{1,6}\s+\S", line):
                if current:
                    merged = "\n".join(current).strip()
                    if merged:
                        sections.append(merged)
                current = [line]
                continue
            current.append(line)

        if current:
            merged = "\n".join(current).strip()
            if merged:
                sections.append(merged)

        # 若沒有有效標題結構（例如純文字），交由一般段落流程處理。
        if len(sections) <= 1:
            return [], False
        return sections, True

    def build_chunks_from_blocks(blocks: list[str]) -> list[str]:
        chunks: list[str] = []
        current_blocks: list[str] = []
        current_len = 0

        def flush_current() -> None:
            nonlocal current_blocks, current_len
            if not current_blocks:
                return
            chunk = "\n\n".join(current_blocks).strip()
            if not chunk:
                return
            chunks.append(chunk)

            # 以段落為單位保留尾端 overlap，降低語意斷裂。
            carry: list[str] = []
            carry_len = 0
            for block in reversed(current_blocks):
                extra = len(block) + (2 if carry else 0)
                if carry_len + extra > overlap:
                    break
                carry.insert(0, block)
                carry_len += extra
            current_blocks = carry
            current_len = sum(len(b) for b in current_blocks) + max(0, len(current_blocks) - 1) * 2

        for block in blocks:
            sep_len = 2 if current_blocks else 0
            if current_blocks and current_len + sep_len + len(block) > chunk_size:
                flush_current()
                sep_len = 2 if current_blocks else 0

            if current_blocks and current_len + sep_len + len(block) > chunk_size:
                # overlap 導致仍裝不下時，先清空 overlap 再放入當前 block。
                current_blocks = []
                current_len = 0
                sep_len = 0

            current_blocks.append(block)
            current_len += sep_len + len(block)

        if current_blocks:
            flush_current()

        return chunks

    sections, has_headings = split_by_markdown_headings(normalized)
    if has_headings:
        # 每個標題章節獨立分塊，不跨標題合併。
        chunks: list[str] = []
        for section in sections:
            section_blocks = split_long_block(section)
            chunks.extend(build_chunks_from_blocks(section_blocks))
        return chunks

    # 無標題時，保留原本段落累積行為。
    raw_blocks = [block.strip() for block in re.split(r"\n\s*\n+", normalized) if block.strip()]
    blocks: list[str] = []
    for block in raw_blocks:
        blocks.extend(split_long_block(block))
    return build_chunks_from_blocks(blocks)


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
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    input_dir = Path(args.input)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}])
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
