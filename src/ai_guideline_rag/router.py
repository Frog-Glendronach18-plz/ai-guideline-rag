"""質問の種類の判定と、検索用のクエリの書き換え（Phase 2 ①）。

1回の LLM 呼び出しで、構造化出力として次の3つを返させる。
- kind：単一の版への質問（single）か、版の差分を聞く質問（diff）か
- version：single のときの対象の版（指定がなければ最新版）
- search_query：本文の検索に使う文

クエリの書き換え：「第2部 D. の章はどう変わった？」のように章の番号を指す質問は、
本文の言葉を含まないため、意味の近さで探す検索では見つからない（評価用の質問 D2）。
判定のときに各版の目次を渡し、章の番号をその章の見出しの言葉に置き換えさせる。
人が目次を見てから本文を探すのと同じ手順。

版の名前を検索用の文に入れない：「第1.2版では AIエージェントは…」のような質問をそのまま検索すると、
「AI事業者ガイドライン(第1.2版)」としか書かれていない表紙（p0）が上位に来て、本文が圏外になった。
プロンプトで禁止したうえで、clean_search_query() でも機械的に取り除く。

try_route.py：確認用のスクリプト（判定結果と、書き換え前後で正解ページの順位がどう変わるか）
python -m uv run python scripts/try_route.py
"""

import re
from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from pydantic import BaseModel, Field

from ai_guideline_rag.answer import get_llm
from ai_guideline_rag.retriever import table_of_contents


class Route(BaseModel):
    kind: Literal["single", "diff"] = Field(
        description="single：1つの版の内容を聞く質問。diff：版の間で何が変わったかを聞く質問"
    )
    version: Literal["1.1", "1.2"] = Field(
        description="single のときの対象の版。質問で版の指定がなければ最新の 1.2。diff のときは 1.2"
    )
    search_query: str = Field(
        description="ガイドライン本文の検索に使う文。本文に出てきそうな言葉で書く"
    )


SYSTEM_PROMPT = """\
あなたは総務省・経済産業省「AI事業者ガイドライン」（第1.1版・第1.2版）の質問を振り分ける係です。
質問を読み、次の3つを決めてください。

1. kind：版の間の違い・変更・追加・削除を聞いていれば diff、それ以外は single。
2. version：single のとき、質問が「第1.1版」「旧版」などを指定していれば 1.1、それ以外は 1.2。
3. search_query：ガイドライン本文を意味の近さで検索するための文。調べたい対象の言葉だけで書く。
   - 版の名前（第1.1版・第1.2版・旧版・新版など）は入れない。版の絞り込みは別に行うため。
     入れると、版の名前しか書かれていない表紙が検索結果の上位に来てしまう。
   - 変化を表す言葉（変更・追加・削除・違い・扱い・どう変わったか など）は入れない。
   - 用語について聞かれていれば「定義」を加える（例：AIエージェント 定義）。
   - 質問が章・節の番号（例：「第2部 D.」「第3部」）を指している場合は、下の目次を見て、
     その章の見出しの言葉に置き換える。版によって見出しが違えば、両方の見出しの言葉を含める。

目次：
{toc}"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "質問: {question}")]
)

# 版の名前。LLM の出力は揺れるので、プロンプトの指示に加えてコードでも確実に取り除く
_VERSION_LABEL = re.compile(r"第?\s*1\.[0-9]+\s*版|旧版|新版|最新版")


def clean_search_query(query: str) -> str:
    """検索用の文から版の名前を取り除き、空白を詰める。何も残らなければ元の文を返す。"""
    cleaned = " ".join(_VERSION_LABEL.sub(" ", query).split())
    return cleaned or query


def route_question(
    store: InMemoryVectorStore, question: str, llm: BaseChatModel | None = None
) -> Route:
    chain = PROMPT | (llm or get_llm()).with_structured_output(Route)
    route = chain.invoke({"toc": table_of_contents(store), "question": question})
    return route.model_copy(update={"search_query": clean_search_query(route.search_query)})
