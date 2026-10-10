"""② splitter の動作確認。

    python -m uv run python scripts/try_split.py [チャンク長] [重なり]

確認すること
- チャンク数と長さの分布（短すぎる・長すぎるものがないか）
- 切れ目が文の途中になっていないか
- 評価用の質問の正解の文が、1つのチャンクに収まっているか
"""

import statistics
import sys

from ai_guideline_rag.loader import load_guidelines
from ai_guideline_rag.splitter import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    split_documents,
)

chunk_size = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CHUNK_SIZE
chunk_overlap = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_CHUNK_OVERLAP

pages = load_guidelines()
chunks = split_documents(pages, chunk_size, chunk_overlap)
lengths = [len(c.page_content) for c in chunks]

print(f"===== チャンク長 {chunk_size} / 重なり {chunk_overlap} =====")
print(f"ページ {len(pages)} → チャンク {len(chunks)}")
print(
    f"長さ: 最小 {min(lengths)} / 中央値 {statistics.median(lengths):.0f} / "
    f"平均 {statistics.mean(lengths):.0f} / 最大 {max(lengths)}"
)
short = [c for c in chunks if len(c.page_content) < 50]
print(f"50文字未満のチャンク: {len(short)}件")
for c in short[:5]:
    print(f"  {c.metadata} {c.page_content!r}")

print("\n===== 切れ目の確認（チャンク末尾の文字） =====")
endings = ["。", "\n", "、", ")", "」"]
tail_counts = {e: sum(c.page_content.endswith(e) for c in chunks) for e in endings}
others = len(chunks) - sum(tail_counts.values())
print({repr(k): v for k, v in tail_counts.items()}, f"その他: {others}")
print("「その他」で終わるチャンクの例:")
for c in [c for c in chunks if not c.page_content.endswith(tuple(endings))][:5]:
    print(f"  {c.metadata} …{c.page_content[-40:]!r}")

print("\n===== 評価用の質問の正解が1チャンクに収まっているか =====")
targets = {
    "S1": ("1.2", "日本企業 9 社を含む 25 組織"),
    "S2": ("1.2", "RAG(検索拡張生成)の活用が適切ではない場合もある"),
    "S3": ("1.2", "ソースコードの開示を必ずしも想定するものではなく"),
    "D1": ("1.2", "高度な自律状態だけを指しているのではなく"),
    "D2": ("1.2", "全ての AI システムに関係する事業者は"),
}
for qid, (version, phrase) in targets.items():
    hits = [c for c in chunks if c.metadata["version"] == version and phrase in c.page_content]
    where = ", ".join(f"p{c.metadata['page']}#{c.metadata['chunk']}" for c in hits)
    print(f"{qid}: {'OK ' + where if hits else '見つからない（切れ目にかかっている可能性）'}")

print("\n===== サンプル（第1.2版 p20 のチャンク） =====")
for c in chunks:
    if c.metadata["version"] == "1.2" and c.metadata["page"] == 20:
        print(f"--- {c.metadata} {len(c.page_content)}文字 ---")
        print(c.page_content)
