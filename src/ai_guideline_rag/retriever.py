"""④ 検索 と ⑤ 検索結果のコンテキスト整形。

- search()：質問に近いチャンクを上位k件取る。version を指定すると、その版だけに絞り込む。
- search_each_version()：版ごとに上位k件ずつ取る（差分の質問用。片方の版に偏らないようにする）。
- table_of_contents()：各版の目次（p1）の文字列。章の番号を指す質問の書き換えに使う。
- format_context()：チャンクを「[第1.2版 p20]」の見出し付きの文字列にまとめ、
  LLM に渡す CONTEXT にする。LLM はこの見出しを見て出典の版・ページを答える。

try_search.py：確認用のスクリプト（評価用の質問で、正解のページが何位に来るか）
python -m uv run python scripts/try_search.py

--- 学習メモ ------------------------------------------------------------------

■ 版での絞り込み（メタデータフィルタ）
  第1.1版と第1.2版には同じ文が多く、絞り込まないと単一の版への質問に古い版が混ざる。
  InMemoryVectorStore は filter に「Document を受け取って True/False を返す関数」を渡せる。
  全件比較なので、絞り込んでから比べても正確さは落ちない（ANN のDBでは、絞り込みを
  先にするか後にするかで取りこぼしが出ることがある）。

■ スコア
  コサイン類似度（-1〜1、大きいほど近い）。値そのものより「順位」と「差」を見る。
  上位の差がわずかなことが多いので、k を小さくしすぎると正解を取りこぼす。

■ LangChain の Retriever
  store.as_retriever(search_kwargs={"k": 4}) で、`|` でつなげる Runnable 形式の検索部品にもできる。
  ここでは版の絞り込みとスコアを扱いやすくするため、関数として書いている。
"""

import re

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ai_guideline_rag.config import GUIDELINE_PDFS, LATEST_VERSION

DEFAULT_K = 5


def search(
    store: InMemoryVectorStore,
    question: str,
    version: str | None = LATEST_VERSION,
    k: int = DEFAULT_K,
) -> list[tuple[Document, float]]:
    """質問に近いチャンクを (Document, 類似度) の組で上位k件返す。version=None なら全版から探す。"""
    if version is None:
        return store.similarity_search_with_score(question, k=k)
    return store.similarity_search_with_score(
        question, k=k, filter=lambda doc: doc.metadata["version"] == version
    )


def search_each_version(
    store: InMemoryVectorStore,
    question: str,
    versions: list[str] | None = None,
    k: int = DEFAULT_K,
) -> dict[str, list[tuple[Document, float]]]:
    """版ごとに上位k件ずつ取る（差分の質問用）。"""
    return {v: search(store, question, v, k) for v in versions or list(GUIDELINE_PDFS)}


def source_label(doc: Document) -> str:
    """出典の見出し。例：「第1.2版 p20」。表紙（p0）は「第1.2版 表紙」。"""
    m = doc.metadata
    page = "表紙" if m["page"] == 0 else f"p{m['page']}"
    return f"第{m['version']}版 {page}"


def format_context(docs: list[Document]) -> str:
    """チャンクを見出し付きでつなげ、LLM に渡す参照テキストにする。"""
    return "\n\n".join(f"[{source_label(d)}]\n{d.page_content}" for d in docs)


def table_of_contents(store: InMemoryVectorStore) -> str:
    """索引から各版の目次（p1）を取り出し、版ごとの見出し付きの文字列にする。

    章の番号（「第2部 D.」など）を指す質問を、内容の言葉に書き換えるときに LLM に渡す。
    目次の点線（……）は省いて短くする。
    """
    parts = []
    for version in GUIDELINE_PDFS:
        chunks = sorted(
            (
                r
                for r in store.store.values()
                if r["metadata"]["version"] == version and r["metadata"]["page"] == 1
            ),
            key=lambda r: r["metadata"]["chunk"],
        )
        text = "\n".join(r["text"] for r in chunks)
        parts.append(f"[第{version}版 目次]\n{re.sub(r'\s*\.{3,}\s*', ' … p', text)}")
    return "\n\n".join(parts)
