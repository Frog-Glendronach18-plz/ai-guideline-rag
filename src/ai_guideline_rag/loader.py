"""① PDF → ページ単位の Document。

- 文字を NFKC 正規化する（第1.1版に混ざる康熙部首「⽉」「⽤」などを通常の漢字にそろえる）
- メタデータに版（version）と印刷されたページ番号（page）を持たせる
- 各ページ先頭の印刷ページ番号の行を本文から取り除く

try_load.py：確認用のスクリプト
python -m uv run python scripts/try_load.py
"""

import re
import unicodedata
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from ai_guideline_rag.config import GUIDELINE_PDFS, PRINTED_PAGE_OFFSET

# ページ先頭の「10 」のような、印刷ページ番号だけの行
_LEADING_PAGE_NUMBER = re.compile(r"\A\s*\d+\s*\n")

# PDFのレイアウトによる折り返し：句点などで終わっていない行の後ろに、和文が続く改行
_WRAPPED_LINE = re.compile(r"(?<=[^\s。:])\n(?=[぀-ヿ一-鿿、。「」()ー])")

# 私用領域の文字（U+E000〜U+F8FF）。PDFのフォント固有の箇条書き記号がこの範囲で残る
# （U+F0B2・U+F0D8・U+F06C・U+F09F）。NFKC では変わらないので「・」に置き換える
_PRIVATE_USE = re.compile(r"[-]")


def normalize_text(text: str) -> str:
    """NFKC 正規化し、行末の空白・連続する空行・和文の途中の折り返し・箇条書き記号を整理する。"""
    text = unicodedata.normalize("NFKC", text)
    lines = [line.rstrip() for line in text.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    text = _WRAPPED_LINE.sub("", text)
    # 「・」はカタカナの範囲にあるため、折り返しの結合より後で置き換える（先にすると改行が消える）
    return _PRIVATE_USE.sub("・", text)


def printed_page(pdf_index: int) -> int:
    """PDF内の位置（0始まり）を、冊子に印刷されたページ番号に変換する。表紙は0。"""
    return pdf_index + 1 + PRINTED_PAGE_OFFSET


def load_pdf(path: Path, version: str) -> list[Document]:
    """1つのPDFを、ページごとの Document のリストにする。"""
    docs = []
    for i, pdf_page in enumerate(PdfReader(path).pages):
        raw = pdf_page.extract_text() or ""
        text = normalize_text(_LEADING_PAGE_NUMBER.sub("", raw, count=1))
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"version": version, "page": printed_page(i), "source": path.name},
            )
        )
    return docs


def load_guidelines(versions: list[str] | None = None) -> list[Document]:
    """config の GUIDELINE_PDFS から、指定した版（省略時は全版）を読み込む。"""
    targets = versions or list(GUIDELINE_PDFS)
    return [doc for v in targets for doc in load_pdf(GUIDELINE_PDFS[v], v)]
