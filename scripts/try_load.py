"""① loader の動作確認。

    uv run python scripts/try_load.py

確認すること
- 版ごとのページ数と文字数
- 印刷ページ番号の付き方（表紙=0、目次=1、本文…）
- NFKC 正規化で康熙部首が残っていないか
"""

import unicodedata
from collections import Counter

from ai_guideline_rag.loader import load_guidelines

docs = load_guidelines()

print("===== 版ごとの件数 =====")
for version, count in Counter(d.metadata["version"] for d in docs).items():
    chars = sum(len(d.page_content) for d in docs if d.metadata["version"] == version)
    pages = [d.metadata["page"] for d in docs if d.metadata["version"] == version]
    print(f"第{version}版: {count}ページ（p{min(pages)}〜p{max(pages)}）、{chars:,}文字")

print("\n===== サンプル（各版の p0・p11） =====")
for doc in docs:
    if doc.metadata["page"] in (0, 11):
        print(f"--- {doc.metadata} ---")
        print(doc.page_content[:150].replace("\n", " / "), "...")

print("\n===== NFKC 正規化の確認 =====")
leftover = {c for d in docs for c in d.page_content if unicodedata.normalize("NFKC", c) != c}
print("正規化で変わる文字の残り:", leftover or "なし")
print("第1.1版 p0 に「月」があるか:", "月" in docs[0].page_content)
