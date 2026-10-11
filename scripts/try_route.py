"""判定（router）の動作確認。

    python -m uv run python scripts/try_route.py

評価用の質問（予備も含む）ごとに、
- 判定結果（single / diff、版）と、書き換えた検索用の文
- 正解ページの順位：元の質問で検索した場合 → 書き換えた文で検索した場合
を表示する。判定の LLM 呼び出しは質問1つにつき1回（Haiku 5.5）。
"""

from ai_guideline_rag.embeddings import get_embeddings
from ai_guideline_rag.evaluation import first_hit_rank, hit_summary, load_questions, targets
from ai_guideline_rag.index import load_index
from ai_guideline_rag.router import route_question

store = load_index(get_embeddings())
before_ranks, after_ranks = [], []


def fmt(rank: int | None) -> str:
    return f"{rank}位" if rank else "圏外"


for q in load_questions(include_spares=True):
    route = route_question(store, q["question"])
    expected_kind = "diff" if q["type"] == "diff" else "single"
    ok = "◯" if route.kind == expected_kind else "✕"
    print(f"\n[{q['id']}] {q['question']}")
    print(f"  判定: {ok} {route.kind}（版 {route.version}） 想定: {expected_kind}")
    print(f"  検索用の文: {route.search_query}")
    for version, pages in targets(q):
        before, _ = first_hit_rank(store, q["question"], version, pages)
        after, _ = first_hit_rank(store, route.search_query, version, pages)
        if q["id"].startswith(("S", "D")):  # 予備（X）は集計に含めない
            before_ranks.append(before)
            after_ranks.append(after)
        want = ",".join(f"p{p}" for p in sorted(pages))
        print(f"  第{version}版 正解 {want}: 元の質問 {fmt(before)} → 書き換え後 {fmt(after)}")

print("\n===== ヒット率（予備を除く） =====")
print(f"元の質問  : {hit_summary(before_ranks)}")
print(f"書き換え後: {hit_summary(after_ranks)}")
