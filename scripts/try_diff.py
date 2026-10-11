"""差分の回答（diff）の動作確認。

    python -m uv run python scripts/try_diff.py            # 評価用の差分の質問（D1・D2・X1）
    python -m uv run python scripts/try_diff.py D2         # 指定した質問だけ

router で検索用の文を作り、その文で版ごとに検索して、差分を答えさせる。
変更点と出典を、questions.json の想定回答・出典と見比べる。LLM 呼び出しは質問1つにつき2回。
"""

import sys

from ai_guideline_rag.diff import answer_diff
from ai_guideline_rag.embeddings import get_embeddings
from ai_guideline_rag.evaluation import load_questions
from ai_guideline_rag.index import load_index
from ai_guideline_rag.retriever import source_label
from ai_guideline_rag.router import route_question


def pages(sources) -> str:
    return "、".join(f"第{s.version}版 p{s.page}" for s in sources) or "なし"


store = load_index(get_embeddings())
ids = sys.argv[1:]
questions = [
    q
    for q in load_questions(include_spares=True)
    if q["type"] == "diff" and (not ids or q["id"] in ids)
]

for q in questions:
    route = route_question(store, q["question"])
    result = answer_diff(store, q["question"], search_query=route.search_query)
    a = result.answer

    print(f"\n{'=' * 70}\n[{q['id']}] {q['question']}")
    print(f"検索用の文: {route.search_query}")
    print(
        "検索結果: "
        + " / ".join(
            f"第{v}版 " + ",".join(f"p{d.metadata['page']}" for d, _ in r)
            for v, r in result.docs.items()
        )
    )
    print(f"\n■ 要約（answerable={a.answerable}）\n{a.summary}")
    for i, c in enumerate(a.changes, start=1):
        print(f"\n■ 変更点{i}：{c.topic}［{c.change_type}］")
        print(f"  旧（1.1）: {c.old}  ← {pages(c.old_sources)}")
        print(f"  新（1.2）: {c.new}  ← {pages(c.new_sources)}")
        print(f"  要点    : {c.point}")
    if result.unknown_sources:
        print(f"\n！渡していない出典: {pages(result.unknown_sources)}")
    print(f"\n■ 想定回答: {q['expected_answer']}")
    print(f"■ 想定の出典: {', '.join(f'第{s["version"]}版 p{s["page"]}' for s in q['sources'])}")
    print(
        "■ 渡したチャンク: "
        + ", ".join(
            f"{source_label(d)}#{d.metadata['chunk']}" for r in result.docs.values() for d, _ in r
        )
    )
