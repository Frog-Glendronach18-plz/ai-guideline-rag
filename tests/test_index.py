"""index の保存・読み込みのテスト。埋め込みAPIの代わりに偽の埋め込みモデルを使う。

--- 学習メモ：なぜ偽の埋め込みモデル（DeterministicFakeEmbedding）を使うのか ---------

■ 偽の埋め込みモデルとは
  LangChain が用意しているテスト専用の埋め込みモデル。文章をもとに乱数のベクトルを作るだけ。
  - 同じ文章からは毎回同じベクトルが出る（Deterministic＝決定的）
  - 意味は理解しない：「ソースコードの公開」と「ソースコードの開示」は全く違うベクトルになる
  - API を呼ばない：PC の中だけで一瞬で計算する（キー不要・費用0円）

■ テストで確かめたいのは「品質」ではなく「処理の正しさ」
  このファイルと test_retriever.py で見ているのは、
  保存して読み込み直しても中身が変わらないか、版で絞り込むと他の版が混ざらないか、
  出典の見出しが正しく付くか、といった自分たちが書いた処理の仕様。
  これらはベクトルに意味があるかどうかと関係なく確かめられる。

■ 本物の API をテストで使うと困ること
  1. GitHub Actions（CI）に API キーを渡す必要があり、秘密情報の管理が増える
  2. ネットの不調や Cloudflare の障害でテストが落ちる（コードは正しいのに失敗する不安定なテスト）
  3. テストを回すたびに無料枠を消費し、時間もかかる

■ 役割分担
  - 処理が仕様どおりか → tests/（pytest）＋ 偽の埋め込み
  - 意味で正しく探せるか（正解ページが上位に来るか）→ scripts/try_search.py ＋ 本物の埋め込み

■ 用語
  外部の API などに頼る部品を、テストのときだけ偽物に差し替える手法を「テストダブル」と呼ぶ
  （モック、スタブ、フェイクなど）。LangChain の埋め込みは共通の形（Embeddings）なので、
  本物と偽物をそのまま入れ替えられる。
"""

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
