"""pipeline のモードの扱いのテスト。LLM は呼ばない（判定結果 Route は手で作る）。"""

import pytest

from ai_guideline_rag.pipeline import resolve
from ai_guideline_rag.router import Route, clean_search_query

SINGLE_11 = Route(kind="single", version="1.1", search_query="…")
DIFF = Route(kind="diff", version="1.2", search_query="…")


@pytest.mark.parametrize(
    ("route", "mode", "expected"),
    [
        # 自動判定：router の判定どおり
        (SINGLE_11, "auto", ("single", "1.1")),
        (DIFF, "auto", ("diff", None)),
        # 版を明示：router が差分と判定しても、指定の版の single にする
        (DIFF, "1.2", ("single", "1.2")),
        (SINGLE_11, "1.2", ("single", "1.2")),
        # 差分を明示：router が single と判定しても diff にする
        (SINGLE_11, "diff", ("diff", None)),
    ],
)
def test_resolve_mode_overrides_route(route, mode, expected):
    # 通る例   : 画面で選んだモードが優先され、自動判定のときだけ router の判定に従う
    # はじく例 : 「第1.2版」を選んだのに router の判定で差分の処理に回る
    assert resolve(route, mode) == expected


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # 通る例   : 版の名前が消え、内容の言葉だけが残る
        ("AIエージェント 追加 第1.1版 第1.2版 扱い", "AIエージェント 追加 扱い"),
        ("第 1.2 版の透明性", "の透明性"),
        ("旧版と新版の報告枠組み", "と の報告枠組み"),
        # はじく例 : 版の名前が残る（表紙が検索の上位に来る）
        # 何も残らない場合は元の文を返す（空の文で検索しない）
        ("第1.2版", "第1.2版"),
    ],
)
def test_clean_search_query_removes_version_labels(query, expected):
    assert clean_search_query(query) == expected
