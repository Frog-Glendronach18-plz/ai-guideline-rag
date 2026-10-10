"""② ページ単位の Document → チャンク。

区切りの優先順：空行（段落）→ 改行（箇条書き・見出し）→ 句点 → 読点 → 空白 → 文字単位。
指定の長さに収まるまで、上位の区切りから順に試して分割する。
句点・読点は直前の文の末尾に残す（keep_separator="end"）。

見出しだけが取り残された短いチャンクは、同じページの次のチャンクの先頭にくっつける。

チャンクはページをまたがない（ページの境目で文が切れることがある）。
メタデータ（版・ページ）は元のページから引き継ぎ、同じページ内での通し番号 chunk を追加する。

try_split.py：確認用のスクリプト
python -m uv run python scripts/try_split.py
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 600
DEFAULT_CHUNK_OVERLAP = 100

JAPANESE_SEPARATORS = ["\n\n", "\n", "。", "、", " ", ""]

# これより短いチャンクは見出しとみなし、次のチャンクと結合する
MIN_CHUNK_LENGTH = 50


def merge_short_texts(texts: list[str], min_length: int = MIN_CHUNK_LENGTH) -> list[str]:
    """短いテキストを次のテキストの先頭に結合する。最後に残った短いものはそのまま。"""
    merged: list[str] = []
    carry = ""
    for text in texts:
        text = f"{carry}\n{text}" if carry else text
        if len(text) < min_length:
            carry = text
        else:
            merged.append(text)
            carry = ""
    if carry:
        merged.append(carry)
    return merged


def split_documents(
    docs: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """ページ単位の Document をチャンクに分割する。長さは文字数で数える。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=JAPANESE_SEPARATORS,
        keep_separator="end",
    )
    chunks = []
    for doc in docs:
        texts = merge_short_texts(splitter.split_text(doc.page_content))
        for i, text in enumerate(texts):
            chunks.append(Document(page_content=text, metadata={**doc.metadata, "chunk": i}))
    return chunks
