"""④⑤ retriever の動作確認。評価用の質問で、正解のページが検索結果の何位に来るかを見る。

    python -m uv run python scripts/try_search.py                 # 保存済みの索引で確認
    python -m uv run python scripts/try_search.py 300 600 1000    # チャンク長ごとに索引を作って比較

正解の判定：検索結果に、questions.json の sources にある（版, ページ）が含まれていれば当たり。
- 単一の版の質問（S1〜S3）：その版で絞り込んで検索する
- 差分の質問（D1・D2）：版ごとに検索し、sources にページがある版それぞれで判定する
LLM は呼ばない（費用は質問文の埋め込みだけ）。

引数の受け取り：sys.argv はコマンドラインの文字列のリスト。
  python scripts/try_search.py 300 600 1000
  → sys.argv = ["scripts/try_search.py", "300", "600", "1000"]（[0] はスクリプト自身、中身は文字列）
  → [int(a) for a in sys.argv[1:]] で [300, 600, 1000] に変換（リスト内包表記）
  引数が増えるなら標準ライブラリの argparse を使う（名前付き引数・既定値・--help）。

--- 結果と読み方のメモ（2026-10-10） --------------------------------------------

■ 実行結果（正解判定6件。D1 の第1.1版は「記載がない」ことの確認用なので判定から除外）
  ===== チャンク長ごとのヒット率 =====
  長さ  300 / 重なり  60（480件）: top1 3/6 / top3 4/6 / top5 4/6 / top10 4/6
  長さ  600 / 重なり 100（252件）: top1 3/6 / top3 3/6 / top5 4/6 / top10 4/6
  長さ 1000 / 重なり 100（156件）: top1 2/6 / top3 3/6 / top5 3/6 / top10 3/6
  600文字の内訳：S1・S2・D1 は1位、S3 は4位（上位とのスコア差 0.700〜0.667 とわずか）、
  D2 は両方の版で圏外。→ 600文字を維持し、検索件数 k は 4 → 5 に増やした。

■ 重なりの割合はほぼ効いていない
  300は20%、600は17%、1000は10%。300と600の割合はほぼ同じなので、この2つの差は重なりでは
  説明できない。重なりの役割は答えの文がチャンクの切れ目で分断されるのを防ぐこと。
  600文字でも評価用の正解の文はすべて1チャンクに収まっている（try_split.py で確認）ので、
  今回は重なりが効く場面がほとんどない。

■ 効いているのは「1つのベクトルに入る話題の量」
  埋め込みはチャンク全体の意味を1つのベクトルに要約する。
  短いチャンク：話題が絞られ、ベクトルがその話題をはっきり表す → 質問と一致しやすい
  長いチャンク：複数の話題が混ざり、ベクトルが平均のようにぼやける（希釈）→ 一致しにくい
  「短い方が上手くベクトル化できている」というより、同じモデルでも質問と一致しやすい単位に
  切れている、と捉えるのが正確。

■ 「件数が減れば当たりやすい」という直感について
  長いチャンクほど上位k件で読める本文の量は増える（上位10件が本文に占める割合：
  300文字で約2%、1000文字で約6%）。ただし検索はくじ引きではなく類似度の順に並べるので、
  正解のベクトルがぼやけて順位が下がれば、範囲が広くても拾えない。
  今回は「範囲が広がる効果」より「ぼやける悪影響」の方が大きかった。

■ ただし結論はまだ弱い
  判定は6件だけで、300と600の差も、1000が悪い差も、それぞれ1問分。
  言えるのは「1000文字は悪くなる傾向がありそう」「300と600は区別がつかない」まで。
  この程度の差で決めると評価用の質問に合わせ込みすぎる。質問を20問以上に増やして比べ直す。
  D2 はどの長さでも圏外：章の番号（第2部「D.」）を指す質問で本文の言葉を含まないため。
  チャンクの長さではなく質問の形の問題で、Phase 2 のクエリ書き換えで改善を確かめる。
"""

import json
import sys

from langchain_core.vectorstores import InMemoryVectorStore

from ai_guideline_rag.config import ROOT_DIR
from ai_guideline_rag.embeddings import get_embeddings
from ai_guideline_rag.index import build_index, load_index
from ai_guideline_rag.loader import load_guidelines
from ai_guideline_rag.retriever import format_context, search, source_label
from ai_guideline_rag.splitter import DEFAULT_CHUNK_OVERLAP, split_documents

MAX_K = 10
K_LEVELS = (1, 3, 5, 10)

questions = json.loads((ROOT_DIR / "data/eval/questions.json").read_text("utf-8"))["questions"]


def targets(q: dict) -> list[tuple[str, set[int]]]:
    """(検索する版, 正解ページの集合) のリスト。"""
    by_version: dict[str, set[int]] = {}
    for s in q["sources"]:
        by_version.setdefault(s["version"], set()).add(s["page"])
    return list(by_version.items())


def first_hit_rank(store: InMemoryVectorStore, question: str, version: str, pages: set[int]):
    """正解ページが最初に現れた順位（1始まり）。MAX_K 件以内になければ None。"""
    results = search(store, question, version, k=MAX_K)
    for rank, (doc, _) in enumerate(results, start=1):
        if doc.metadata["page"] in pages:
            return rank, results
    return None, results


def evaluate(store: InMemoryVectorStore, verbose: bool) -> list[int | None]:
    ranks = []
    for q in questions:
        for version, pages in targets(q):
            rank, results = first_hit_rank(store, q["question"], version, pages)
            ranks.append(rank)
            if not verbose:
                continue
            want = ",".join(f"p{p}" for p in sorted(pages))
            print(f"\n[{q['id']}] 第{version}版 正解 {want} → {f'{rank}位' if rank else '圏外'}")
            print(f"  {q['question']}")
            for i, (doc, score) in enumerate(results[:5], start=1):
                mark = "◯" if doc.metadata["page"] in pages else " "
                head = doc.page_content[:40].replace("\n", " ")
                print(
                    f"  {mark}{i}. {score:.3f} {source_label(doc)}#{doc.metadata['chunk']} {head}"
                )
    return ranks


def summary(ranks: list[int | None]) -> str:
    return " / ".join(
        f"top{k} {sum(1 for r in ranks if r and r <= k)}/{len(ranks)}" for k in K_LEVELS
    )


embeddings = get_embeddings()
chunk_sizes = [int(a) for a in sys.argv[1:]]

if not chunk_sizes:
    store = load_index(embeddings)
    ranks = evaluate(store, verbose=True)
    print(f"\n===== ヒット率（保存済みの索引） =====\n{summary(ranks)}")

    q = questions[2]  # S3
    print(f"\n===== CONTEXT の例（{q['id']}、上位2件） =====")
    print(format_context([doc for doc, _ in search(store, q["question"], k=2)]))
else:
    pages = load_guidelines()
    print("===== チャンク長ごとのヒット率 =====")
    for size in chunk_sizes:
        overlap = min(DEFAULT_CHUNK_OVERLAP, size // 5)
        chunks = split_documents(pages, size, overlap)
        store = build_index(chunks, embeddings)
        label = f"長さ {size:>4} / 重なり {overlap:>3}（{len(chunks)}件）"
        print(f"{label}: {summary(evaluate(store, False))}")
