"""③ 索引を作って data/index.json に保存する。

    python -m uv run python scripts/build_index.py [チャンク長] [重なり]

PDF の読み込み → 分割 → 埋め込み（Workers AI）→ 保存 までを通して実行する。
保存後に読み込み直し、件数とメタデータが一致するかを確かめる。
"""

import sys
import time

from ai_guideline_rag.config import INDEX_PATH
from ai_guideline_rag.embeddings import get_embeddings
from ai_guideline_rag.index import build_index, load_index, save_index
from ai_guideline_rag.loader import load_guidelines
from ai_guideline_rag.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_documents,
)

chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHUNK_SIZE
chunk_overlap = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHUNK_OVERLAP

chunks = split_documents(load_guidelines(), chunk_size, chunk_overlap)
total_chars = sum(len(c.page_content) for c in chunks)
print(f"チャンク {len(chunks)}件（{total_chars:,}文字）を埋め込みます")
print(f"（長さ {chunk_size} / 重なり {chunk_overlap}）")

embeddings = get_embeddings()
start = time.perf_counter()
store = build_index(chunks, embeddings)
print(f"埋め込み完了: {time.perf_counter() - start:.1f}秒")

save_index(store)
size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
print(f"保存: {INDEX_PATH}（{size_mb:.1f} MB）")

# 読み込み直して、件数・メタデータ・ベクトルの次元を確かめる
loaded = load_index(embeddings)
records = list(loaded.store.values())
print(f"読み込み直し: {len(records)}件（作成時 {len(chunks)}件）")
print(f"ベクトルの次元: {len(records[0]['vector'])}")
print(f"メタデータの例: {records[0]['metadata']}")
assert len(records) == len(chunks), "件数が一致しない"
