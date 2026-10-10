"""index の保存・読み込みのテスト。埋め込みAPIの代わりに偽の埋め込みモデルを使う。"""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ai_guideline_rag.index import build_index, chunk_id, load_index, save_index


def _chunks() -> list[Document]:
    return [
        Document(
            page_content="ソースコードの開示", metadata={"version": "1.2", "page": 20, "chunk": 2}
        ),
        Document(
            page_content="AIエージェントの定義", metadata={"version": "1.2", "page": 11, "chunk": 1}
        ),
    ]


def test_chunk_id_is_version_page_chunk():
    # 通る例   : "1.2-p20-c2"
    # はじく例 : "e9315ecb-…"（ランダムなID。作り直すたびに全件が変わる）
    assert chunk_id(_chunks()[0]) == "1.2-p20-c2"


def test_save_and_load_roundtrip(tmp_path):
    # 通る例   : 保存して読み込み直すと、ID・本文・メタデータが同じで、ベクトルの差は丸め誤差以内
    # はじく例 : 件数が減る／メタデータ（版・ページ）が消える／ベクトルが大きく変わる
    embeddings = DeterministicFakeEmbedding(size=8)
    store = build_index(_chunks(), embeddings)
    path = tmp_path / "index.json"
    save_index(store, path)
    loaded = load_index(embeddings, path)

    assert set(loaded.store) == {"1.2-p20-c2", "1.2-p11-c1"}
    original, restored = store.store["1.2-p20-c2"], loaded.store["1.2-p20-c2"]
    assert restored["text"] == original["text"]
    assert restored["metadata"] == original["metadata"]
    assert (
        max(abs(a - b) for a, b in zip(original["vector"], restored["vector"], strict=True)) < 1e-6
    )
