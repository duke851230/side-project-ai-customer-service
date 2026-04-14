from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"


def load_eval_set(path: Path) -> list[dict[str, str]]:
    """從 JSON 載入評估資料。

    預期格式：
    - 最外層為陣列
    - 每筆可包含 `question`、`expected_source`、`is_unknown`
    """
    if not path.exists():
        raise SystemExit(f"Eval set not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Eval set must be a JSON array")
    return data


def main() -> None:
    """對既有 FAISS 索引做離線檢索評估。

    指標：
    - `retrieval_recall_at_k`：正確來源是否出現在 top-k
    - `unknown_query_refusal_rate`：對未知問題是否依 `score_threshold` 正確拒答
    """
    parser = argparse.ArgumentParser(description="Offline evaluation for retrieval")
    parser.add_argument("--index-dir", default="./data/index")
    parser.add_argument("--eval-set", default="./data/eval_set/eval.json")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--score-threshold", type=float, default=0.45)
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    metadata_path = index_dir / "metadata.json"
    index_path = index_dir / "faiss.index"

    if not metadata_path.exists() or not index_path.exists():
        raise SystemExit("Index files missing. Run scripts/ingest.py first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    index = faiss.read_index(str(index_path))
    eval_rows = load_eval_set(Path(args.eval_set))

    model = SentenceTransformer(args.embedding_model)

    total = len(eval_rows)
    if total == 0:
        raise SystemExit("Eval set is empty")

    retrieval_hit = 0
    should_refuse = 0
    refused = 0

    for row in eval_rows:
        question = row.get("question", "")
        expected_source = row.get("expected_source", "")
        is_unknown = bool(row.get("is_unknown", False))

        query_vec = model.encode([question], normalize_embeddings=True)
        query_vec = np.asarray(query_vec, dtype="float32")
        scores, indices = index.search(query_vec, args.top_k)

        top_score = float(scores[0][0]) if indices[0][0] != -1 else -1.0
        top_sources = [metadata[i]["source"] for i in indices[0] if i != -1]

        if expected_source and expected_source in top_sources:
            retrieval_hit += 1

        if is_unknown:
            should_refuse += 1
            if top_score < args.score_threshold:
                refused += 1

    recall_at_k = retrieval_hit / total
    refusal_rate = (refused / should_refuse) if should_refuse else 0.0

    print(json.dumps(
        {
            "total": total,
            "top_k": args.top_k,
            "retrieval_recall_at_k": round(recall_at_k, 4),
            "unknown_query_refusal_rate": round(refusal_rate, 4),
            "score_threshold": args.score_threshold,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
