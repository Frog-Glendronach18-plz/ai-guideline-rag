"""splitter が「指定どおりに分割できているか」のテスト。チャンクサイズの良し悪しは扱わない。"""

from langchain_core.documents import Document

from ai_guideline_rag.splitter import merge_short_texts, split_documents


def test_period_stays_at_end_of_sentence():
    # 通る例   : ["一文目です。一文目です。…", "一文目です。…"]   … どのチャンクも「。」で終わる
    # はじく例 : ["一文目です。一文目です", "。一文目です。…"]   … 「。」が次のチャンクの先頭に付く
    #            （keep_separator="end" を外すとこうなる）
    doc = Document(page_content="一文目です。" * 20, metadata={"version": "1.2", "page": 5})
    chunks = split_documents([doc], chunk_size=50, chunk_overlap=0)
    assert all(c.page_content.endswith("。") for c in chunks)
    assert not any(c.page_content.startswith("。") for c in chunks)


def test_metadata_is_inherited_with_chunk_number():
    # 通る例   : {"version": "1.1", "page": 3, "chunk": 0}, {… "chunk": 1}, …
    # はじく例 : {"chunk": 0}                          … 版・ページが消えて出典が出せない
    #            {"version": "1.1", "page": 3, "chunk": 0}, {… "chunk": 0} … 通し番号が振られない
    doc = Document(page_content="あ。" * 100, metadata={"version": "1.1", "page": 3})
    chunks = split_documents([doc], chunk_size=50, chunk_overlap=0)
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))
    assert all(c.metadata["version"] == "1.1" and c.metadata["page"] == 3 for c in chunks)


def test_short_heading_is_merged_into_next_text():
    # 通る例   : ["はじめに\n本文です。本文です。…"]       … 見出しが本文の先頭に付く
    # はじく例 : ["はじめに", "本文です。本文です。…"]     … 見出しだけのチャンクが残る
    body = "本文です。" * 20
    assert merge_short_texts(["はじめに", body]) == [f"はじめに\n{body}"]


def test_trailing_short_text_is_kept():
    # 通る例   : ["本文です。…", "末尾"]   … 結合先がない短い文もそのまま残る
    # はじく例 : ["本文です。…"]           … 末尾の短い文が黙って消える
    body = "本文です。" * 20
    assert merge_short_texts([body, "末尾"]) == [body, "末尾"]
