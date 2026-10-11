"""質問して、回答と出典を表示する（コマンドライン版のアプリ）。pipeline.ask() を呼ぶ。

    python -m uv run python scripts/ask.py "質問"
    python -m uv run python scripts/ask.py "質問" --mode 1.1 --show-context
    python -m uv run python scripts/ask.py --eval S1 D2     # 評価用の質問を流す（自動判定）

--mode：auto（自動判定、既定）／1.2／1.1／diff（版の差分）
引数は標準ライブラリの argparse で受け取る（python scripts/ask.py --help で一覧が出る）。
"""

import argparse

from ai_guideline_rag.answer import Source
from ai_guideline_rag.embeddings import get_embeddings
from ai_guideline_rag.evaluation import load_questions
from ai_guideline_rag.index import load_index
from ai_guideline_rag.pipeline import PipelineResult, ask
from ai_guideline_rag.retriever import format_context, source_label

parser = argparse.ArgumentParser(description="AI事業者ガイドラインに質問する")
parser.add_argument("question", nargs="?", help="質問文")
parser.add_argument("--mode", default="auto", choices=["auto", "1.2", "1.1", "diff"])
parser.add_argument("--show-context", action="store_true", help="LLM に渡した参照テキストを表示")
parser.add_argument(
    "--eval", nargs="+", metavar="ID", help="questions.json の質問を流す（例 S1 D2）"
)
args = parser.parse_args()


def pages(sources: list[Source]) -> str:
    return "、".join(f"第{s.version}版 p{s.page}" for s in sources) or "なし"


def show(question: str, result: PipelineResult, expected: str | None = None) -> None:
    r = result.route
    target = "版の差分" if result.kind == "diff" else f"第{result.version}版"
    print(f"\n■ 質問: {question}")
    print(f"■ 判定: {r.kind}（版 {r.version}）→ 使った処理: {target}")
    print(f"■ 検索用の文: {r.search_query}")

    if result.single:
        a = result.single.answer
        print(f"■ 回答（answerable={a.answerable}）:\n{a.answer}")
        print(f"■ 出典: {pages(a.sources)}")
        unknown, docs = result.single.unknown_sources, result.single.docs
    else:
        a = result.diff.answer
        print(f"■ 要約（answerable={a.answerable}）:\n{a.summary}")
        for i, c in enumerate(a.changes, start=1):
            print(f"■ 変更点{i}：{c.topic}［{c.change_type}］")
            print(f"  旧（1.1）: {c.old}  ← {pages(c.old_sources)}")
            print(f"  新（1.2）: {c.new}  ← {pages(c.new_sources)}")
            print(f"  要点    : {c.point}")
        unknown = result.diff.unknown_sources
        docs = [d for results in result.diff.docs.values() for d in results]

    if unknown:
        print(f"  ！渡していない出典: {pages(unknown)}")
    print("■ 検索結果（LLM に渡したチャンク）:")
    for doc, score in docs:
        print(f"  {score:.3f} {source_label(doc)}#{doc.metadata['chunk']}")
    if expected:
        print(f"■ 想定回答: {expected}")
    if args.show_context:
        print(f"■ 参照テキスト:\n{format_context([d for d, _ in docs])}")


store = load_index(get_embeddings())

if args.eval:
    by_id = {q["id"]: q for q in load_questions(include_spares=True)}
    for qid in args.eval:
        q = by_id[qid]
        show(f"[{qid}] {q['question']}", ask(store, q["question"]), q["expected_answer"])
elif args.question:
    show(args.question, ask(store, args.question, args.mode))
else:
    parser.print_help()
