from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import faiss
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard for local env
    raise SystemExit("faiss is not installed. Install dependencies from app/requirements.txt.") from exc


def main() -> None:
    """顯示 FAISS 索引摘要，並列出前幾筆向量與對應 metadata。"""
    parser = argparse.ArgumentParser(description="Inspect local FAISS index and mapped metadata")
    parser.add_argument("--index-dir", default="./data/index", help="Directory containing faiss.index and metadata.json")
    parser.add_argument("--show", type=int, default=3, help="How many rows to preview")
    parser.add_argument("--dims", type=int, default=8, help="How many vector dimensions to preview per row")
    parser.add_argument("--text-preview", type=int, default=90, help="How many chars to show for metadata text preview")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    index_path = index_dir / "faiss.index"
    metadata_path = index_dir / "metadata.json"

    if not index_path.exists() or not metadata_path.exists():
        raise SystemExit(f"Missing files under {index_dir}. Expect faiss.index and metadata.json")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise SystemExit("metadata.json must be a JSON array")

    index = faiss.read_index(str(index_path))

    print(f"index_path: {index_path}")
    print(f"metadata_path: {metadata_path}")
    print(f"index_type: {type(index).__name__}")
    print(f"vector_dim: {index.d}")
    print(f"index_ntotal: {index.ntotal}")
    print(f"metadata_count: {len(metadata)}")

    if index.ntotal != len(metadata):
        print("warning: index_ntotal != metadata_count; row mapping may be inconsistent.")

    show_count = max(0, min(args.show, index.ntotal, len(metadata)))
    dims = max(1, min(args.dims, index.d))

    for i in range(show_count):
        vec = np.asarray(index.reconstruct(i), dtype="float32")
        row = metadata[i]

        text_preview = str(row.get("text", "")).replace("\n", " ")[: args.text_preview]
        vec_preview = np.array2string(vec[:dims], precision=4, separator=", ")

        print(f"\n[{i}]")
        print(f"id: {row.get('id', '')}")
        print(f"source: {row.get('source', '')}")
        print(f"vector_first_{dims}: {vec_preview}")
        print(f"vector_l2_norm: {np.linalg.norm(vec):.6f}")
        print(f"text_preview: {text_preview}")


if __name__ == "__main__":
    # docker compose exec app python scripts/inspect_index.py --show 100
    main()
