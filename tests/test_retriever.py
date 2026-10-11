"""retriever の版の絞り込みとコンテキスト整形のテスト。偽の埋め込みモデルを使う。

偽の埋め込みモデルを使う理由は test_index.py の「学習メモ」を参照。
偽物は意味を理解しないが「同じ文章なら同じベクトル」にはなるので、
検索語と同じ文字列（"同じ文"）を本文に入れて、確実に一致させている。
"""

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from ai_guideline_rag.index import build_index
from ai_guideline_rag.retriever import (
    format_context,
    search,
    search_each_version,
    source_label,
    table_of_contents,
)


def _store():
    chunks = [
        Document(page_content="同じ文", metadata={"version": "1.1", "page": 19, "chunk": 1}),
        Document(page_content="同じ文", metadata={"version": "1.2", "page": 20, "chunk": 2}),
        Document(page_content="新設の文", metadata={"version": "1.2", "page": 11, "chunk": 1}),
    ]
    return build_index(chunks, DeterministicFakeEmbedding(size=8))


def test_search_filters_by_version():
    # 通る例   : version="1.2" なら第1.2版のチャンクだけが返る
    # はじく例 : 同じ文を持つ第1.1版 p19 が混ざる（単一の版への質問に古い版が入る）
    results = search(_store(), "同じ文", version="1.2", k=5)
    assert {doc.metadata["version"] for doc, _ in results} == {"1.2"}


def test_search_without_version_uses_all_versions():
    # 通る例   : version=None なら両方の版から返る
    results = search(_store(), "同じ文", version=None, k=5)
    assert {doc.metadata["version"] for doc, _ in results} == {"1.1", "1.2"}


def test_search_each_version_returns_both_versions():
    # 通る例   : {"1.1": [第1.1版のみ], "1.2": [第1.2版のみ]}
    # はじく例 : 片方の版が空、または版が混ざる
    results = search_each_version(_store(), "同じ文", versions=["1.1", "1.2"], k=5)
    assert {doc.metadata["version"] for doc, _ in results["1.1"]} == {"1.1"}
    assert {doc.metadata["version"] for doc, _ in results["1.2"]} == {"1.2"}


def test_format_context_has_source_headings():
    # 通る例   : "[第1.2版 p20]\n同じ文\n\n[第1.2版 表紙]\n表題"
    # はじく例 : 見出しがなく本文だけ（LLM が出典ページを答えられない）
    docs = [
        Document(page_content="同じ文", metadata={"version": "1.2", "page": 20}),
        Document(page_content="表題", metadata={"version": "1.2", "page": 0}),
    ]
    assert format_context(docs) == "[第1.2版 p20]\n同じ文\n\n[第1.2版 表紙]\n表題"
    assert source_label(docs[1]) == "第1.2版 表紙"


def test_table_of_contents_per_version():
    # 通る例   : "[第1.1版 目次]\nD. 高度な… … p25\n\n[第1.2版 目次]\nD. 広島… … p26"
    #            （各版の p1 だけを、版ごとの見出し付きで、点線を「 … p」に縮めて並べる）
    # はじく例 : p1 以外のページが混ざる／版の見出しがない／点線が残る
    chunks = [
        Document(
            page_content="D. 高度な指針 ........ 25",
            metadata={"version": "1.1", "page": 1, "chunk": 0},
        ),
        Document(
            page_content="D. 広島の指針 ........ 26",
            metadata={"version": "1.2", "page": 1, "chunk": 0},
        ),
        Document(page_content="本文", metadata={"version": "1.2", "page": 20, "chunk": 0}),
    ]
    store = build_index(chunks, DeterministicFakeEmbedding(size=8))
    assert table_of_contents(store) == (
        "[第1.1版 目次]\nD. 高度な指針 … p25\n\n[第1.2版 目次]\nD. 広島の指針 … p26"
    )
