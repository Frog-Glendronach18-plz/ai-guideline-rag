"""③ 索引（ベクトルストア）の作成・保存・読み込み。

チャンクごとに埋め込みベクトルを計算し、本文・メタデータと一緒に保存する。
アプリ起動時は保存済みのファイルを読み込むだけなので、チャンクの埋め込みを毎回作り直さない。
（検索時に質問文の埋め込みだけは毎回作る）

保存形式は InMemoryVectorStore.dump() と同じ JSON（{id: {id, vector, text, metadata}}）だが、
リポジトリに含めるためサイズを抑える：インデントなし、ベクトルは小数6桁に丸める（7.2MB → 2.6MB）。
ID は「版-ページ-チャンク番号」（例 1.2-p20-c2）に固定し、作り直したときの差分を読みやすくする。

build_index.py：索引を作って保存するスクリプト
python -m uv run python scripts/build_index.py

--- 学習メモ ------------------------------------------------------------------

■ RAG 全体の流れの中での索引の役割
  【事前に1回】文書 → チャンク →(埋め込み)→ ベクトル
               index.json に「ベクトル＋元の文章＋メタデータ」をセットで保存
  【質問のたび】質問文 →(同じ埋め込みモデル)→ 質問のベクトル
               → ベクトルの近さで近いチャンクを探す → 見つかったチャンクの「元の文章」を取り出す
               → 元の文章と出典を LLM に渡す
  ベクトルは「探すための目次」として使うだけ。LLM（Claude）はベクトルを見ず、普通の文章を読む。

■ ハッシュとの違い
  ハッシュ：似た入力でもまったく違う値になる。完全一致で引く（ある／ない）。
  埋め込み：意味が似ていれば近い値になる。近い順に並べて返す（必ず何かしら返る）。
  質問と本文の言い回しが違っても（「公開」と「開示」）見つけられるのは、この性質のおかげ。

■ 今の検索は「全件比較」
  InMemoryVectorStore には検索を速くするインデックス構造がない。質問のたびに全252件と
  類似度を計算して並べ替える。数万件までならこれで正確かつ十分速い。

■ 大規模になったら：近似最近傍探索（ANN）のインデックス
  FAISS、pgvector、Cloudflare Vectorize、Pinecone などが持つ。DB の B-tree にあたる。
  「おおよそ近いもの」を高速に返す代わりに、まれに本当の1位を取りこぼす（速さと正確さの交換）。
  - HNSW：近いベクトル同士をグラフでつないでたどる。学習不要で更新に強く、今の標準。
  - IVF ：k-means でクラスタに分け、近いクラスタの中だけ探す。索引作成前に「学習」が要る。
  - PQ  ：ベクトルを分割し、k-means で作った代表値の番号に置き換えて圧縮する。
  - LSH ：似たものが「同じ」ハッシュ値になりやすいよう設計した特殊なハッシュ。
  IVF・PQ はデータの分布が変わると精度が落ち、作り直し（再学習）が必要になる。
  使う側が調整するのは正確さと速さのバランス（HNSW の M・ef_search、IVF の nlist・nprobe）。
  正解を取りこぼさない割合（recall@k）と応答時間を測って決める。
  ここでの機械学習は k-means のような古典的な手法。文章の「意味」を扱うのは埋め込みモデルの側。

■ 従来の DB 技術の領域
  保存と復旧、更新と削除、メタデータでの絞り込み（「第1.2版だけ」。ANN と組み合わせると
  絞り込みを先にするか後にするかで速さと正確さが変わる）、分散、権限管理。

■ 社内文書で使うときの注意
  - 索引ファイルには元の文章がそのまま入る。元の文書と同じ機密レベルで管理する
    （今回は公開文書なので GitHub に置いてよい）。
  - ベクトルだけでも元の文章をある程度推測できるという研究がある。ベクトルも機密扱いが安全。
  - 文章は外部 API に送られる：索引作成時に全文を埋め込み API（Cloudflare）へ、
    質問のたびに検索結果を LLM の API（Anthropic）へ。学習に使われないか・保存期間を確認する。
"""

import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore

from ai_guideline_rag.config import INDEX_PATH

VECTOR_DIGITS = 6


def chunk_id(doc: Document) -> str:
    m = doc.metadata
    return f"{m['version']}-p{m['page']}-c{m['chunk']}"


def build_index(chunks: list[Document], embeddings: Embeddings) -> InMemoryVectorStore:
    """チャンクを埋め込んでベクトルストアを作る（埋め込みAPIを呼ぶ）。"""
    store = InMemoryVectorStore(embeddings)
    store.add_documents(chunks, ids=[chunk_id(c) for c in chunks])
    return store


def save_index(store: InMemoryVectorStore, path: Path = INDEX_PATH) -> None:
    records = {
        id_: {**rec, "vector": [round(x, VECTOR_DIGITS) for x in rec["vector"]]}
        for id_, rec in store.store.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, separators=(",", ":")), "utf-8")


def load_index(embeddings: Embeddings, path: Path = INDEX_PATH) -> InMemoryVectorStore:
    """保存済みの索引を読み込む。embeddings は検索時に質問文を埋め込むために使う。"""
    return InMemoryVectorStore.load(str(path), embeddings)
