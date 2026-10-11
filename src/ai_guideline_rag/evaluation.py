"""評価用の共通処理。評価用の質問の読み込みと、検索で正解のページが何位に来るかの判定。

try_search.py・try_route.py・（今後の）eval.py から使う。
"""

import json

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from ai_guideline_rag.config import ROOT_DIR
from ai_guideline_rag.retriever import search

QUESTIONS_PATH = ROOT_DIR / "data" / "eval" / "questions.json"
MAX_K = 10
K_LEVELS = (1, 3, 5, 10)


def load_questions(include_spares: bool = False) -> list[dict]:
    data = json.loads(QUESTIONS_PATH.read_text("utf-8"))
    return data["questions"] + (data["spares"] if include_spares else [])


def targets(question: dict) -> list[tuple[str, set[int]]]:
    """(検索する版, 正解ページの集合) のリスト。sources にある版ごとにまとめる。"""
    by_version: dict[str, set[int]] = {}
    for s in question["sources"]:
        by_version.setdefault(s["version"], set()).add(s["page"])
    return list(by_version.items())


def first_hit_rank(
    store: InMemoryVectorStore, query: str, version: str, pages: set[int], k: int = MAX_K
) -> tuple[int | None, list[tuple[Document, float]]]:
    """正解ページが最初に現れた順位（1始まり）と検索結果。k 件以内になければ順位は None。"""
    results = search(store, query, version, k=k)
    for rank, (doc, _) in enumerate(results, start=1):
        if doc.metadata["page"] in pages:
            return rank, results
    return None, results


def hit_summary(ranks: list[int | None]) -> str:
    """「top1 3/6 / top3 4/6 …」の形の要約。"""
    return " / ".join(
        f"top{k} {sum(1 for r in ranks if r and r <= k)}/{len(ranks)}" for k in K_LEVELS
    )
