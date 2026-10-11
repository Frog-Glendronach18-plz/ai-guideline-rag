"""answer の出典確認のテスト。LLM は呼ばない（LLM の回答の質は ask.py で確認する）。"""

from langchain_core.documents import Document

from ai_guideline_rag.answer import Answer, Source, find_unknown_sources

DOCS = [
    Document(page_content="…", metadata={"version": "1.2", "page": 36}),
    Document(page_content="…", metadata={"version": "1.2", "page": 7}),
]


def _answer(*sources: tuple[str, int]) -> Answer:
    return Answer(
        answer="…",
        sources=[Source(version=v, page=p) for v, p in sources],
        answerable=True,
    )


def test_sources_in_given_docs_are_not_flagged():
    # 通る例   : 渡したチャンク（第1.2版 p36・p7）だけを出典に挙げた → 指摘なし
    assert find_unknown_sources(_answer(("1.2", 36), ("1.2", 7)).sources, DOCS) == []


def test_source_not_in_given_docs_is_flagged():
    # はじく例 : 渡していない第1.2版 p99 を出典に挙げた → 「渡していない出典」として返す
    #            同じページでも版が違う（第1.1版 p36）なら渡していない扱い
    unknown = find_unknown_sources(_answer(("1.2", 36), ("1.2", 99), ("1.1", 36)).sources, DOCS)
    assert [(s.version, s.page) for s in unknown] == [("1.2", 99), ("1.1", 36)]
