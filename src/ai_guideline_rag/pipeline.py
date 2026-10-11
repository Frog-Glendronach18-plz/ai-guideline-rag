"""質問から回答までの流れをまとめる（Phase 2 ①）。画面（app.py）と ask.py はここを呼ぶ。

    質問 → router（判定＋検索用の文の書き換え。LLM 1回目）
         ├ single → answer.py：指定の版だけを検索して回答（LLM 2回目）
         └ diff   → diff.py  ：版ごとに検索して差分を回答（LLM 2回目）

方針：
- 検索用の文の書き換えは、差分の質問に限らず全ての質問に共通の工程にする。
  LLM 自体が答えを知っているのではなく、一度検索を挟むため、検索しやすい文に直す効果がある
  （評価用の質問で top3 のヒット率 3/6 → 6/6。D2 は圏外 → 1位、S3 は4位 → 3位）。
- 検索には書き換えた文を、回答の LLM には利用者の元の質問を渡す。書き換えで質問の条件が
  落ちても、回答は元の質問に照らして作られる。
- 画面で版や差分を明示的に選んだ場合（mode が auto 以外）は、router の判定（kind・version）
  だけを上書きし、書き換えた検索用の文はそのまま使う。
- 書き換えで悪化していないかは、評価の質問を増やしたときに書き換え前後のヒット率で見張る
  （try_route.py）。必要なら元の質問と書き換えた文の両方で検索して結果を合わせる。

router を answer.py の中で呼ばない理由：answer.py は「指定の版で答える」、
diff.py は「差分を答える」、router.py は「どちらに回すか決める」と役割を分けるため。
また router.py は answer.py の get_llm() を使うので、answer.py から router.py を呼ぶと
import が循環する。
"""

from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import InMemoryVectorStore

from ai_guideline_rag.answer import AnswerResult, answer_question
from ai_guideline_rag.diff import DiffResult, answer_diff
from ai_guideline_rag.router import Route, route_question

Mode = Literal["auto", "1.2", "1.1", "diff"]
Kind = Literal["single", "diff"]


@dataclass
class PipelineResult:
    route: Route  # router の判定（上書き前）と検索用の文
    kind: Kind  # 実際に使った処理
    version: str | None  # single のときの版（diff のときは None）
    single: AnswerResult | None = None
    diff: DiffResult | None = None


def resolve(route: Route, mode: Mode) -> tuple[Kind, str | None]:
    """router の判定と画面のモードから、使う処理（single / diff）と版を決める。"""
    if mode == "diff":
        return "diff", None
    if mode in ("1.1", "1.2"):
        return "single", mode
    if route.kind == "diff":
        return "diff", None
    return "single", route.version


def ask(
    store: InMemoryVectorStore,
    question: str,
    mode: Mode = "auto",
    llm: BaseChatModel | None = None,
) -> PipelineResult:
    route = route_question(store, question, llm)
    kind, version = resolve(route, mode)
    if kind == "diff":
        diff = answer_diff(store, question, llm=llm, search_query=route.search_query)
        return PipelineResult(route=route, kind=kind, version=None, diff=diff)
    single = answer_question(store, question, version, llm=llm, search_query=route.search_query)
    return PipelineResult(route=route, kind=kind, version=version, single=single)
