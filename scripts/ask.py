"""⑥ 質問して、回答と出典を表示する（コマンドライン版のアプリ）。

    python -m uv run python scripts/ask.py "質問"
    python -m uv run python scripts/ask.py "質問" --version 1.1 --k 3 --show-context
    python -m uv run python scripts/ask.py --eval S1 S2 S3     # 評価用の質問を流す

引数は標準ライブラリの argparse で受け取る（python scripts/ask.py --help で一覧が出る）。
"""

import argparse
import json

from ai_guideline_rag.answer import AnswerResult, answer_question
from ai_guideline_rag.config import GUIDELINE_PDFS, LATEST_VERSION, ROOT_DIR
from ai_guideline_rag.embeddings import get_embeddings
from ai_guideline_rag.index import load_index
from ai_guideline_rag.retriever import DEFAULT_K, format_context, source_label

parser = argparse.ArgumentParser(description="AI事業者ガイドラインに質問する")
parser.add_argument("question", nargs="?", help="質問文")
parser.add_argument("--version", default=LATEST_VERSION, choices=list(GUIDELINE_PDFS))
parser.add_argument("--k", type=int, default=DEFAULT_K, help="検索する件数")
parser.add_argument("--show-context", action="store_true", help="LLM に渡した参照テキストを表示")
parser.add_argument(
    "--eval", nargs="+", metavar="ID", help="questions.json の質問を流す（例 S1 S3）"
)
args = parser.parse_args()


def show(question: str, result: AnswerResult, expected: str | None = None) -> None:
    a = result.answer
    print(f"\n■ 質問: {question}")
    print(f"■ 回答（answerable={a.answerable}）:\n{a.answer}")
    cited = ", ".join(f"第{s.version}版 p{s.page}" for s in a.sources) or "なし"
    print(f"■ 出典: {cited}")
    if result.unknown_sources:
        print(f"  ！渡していない出典: {[(s.version, s.page) for s in result.unknown_sources]}")
    print("■ 検索結果（LLM に渡したチャンク）:")
    for doc, score in result.docs:
        print(f"  {score:.3f} {source_label(doc)}#{doc.metadata['chunk']}")
    if expected:
        print(f"■ 想定回答: {expected}")
    if args.show_context:
        print(f"■ 参照テキスト:\n{format_context([d for d, _ in result.docs])}")


store = load_index(get_embeddings())

if args.eval:
    questions = json.loads((ROOT_DIR / "data/eval/questions.json").read_text("utf-8"))
    by_id = {q["id"]: q for q in questions["questions"] + questions["spares"]}
    for qid in args.eval:
        q = by_id[qid]
        version = q.get("version", LATEST_VERSION)
        result = answer_question(store, q["question"], version, args.k)
        show(f"[{qid}] {q['question']}", result, q["expected_answer"])
elif args.question:
    show(args.question, answer_question(store, args.question, args.version, args.k))
else:
    parser.print_help()
