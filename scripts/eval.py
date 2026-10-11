"""評価：評価用の質問を pipeline（自動判定）に通しで流し、結果を表にして保存する。

    python -m uv run python scripts/eval.py                     # 全問
    python -m uv run python scripts/eval.py --label baseline    # 結果ファイル名に付ける名前
    python -m uv run python scripts/eval.py --ids S1 D2         # 指定した質問だけ

1問ごとに記録すること
- 判定：種類（single / diff）と、single なら対象の版が想定どおりか
- 検索：LLM に渡した上位k件に正解のページが入っているか（差分は版ごと）。入っていれば何位か
- キーワード：回答に must_include のキーワードが全部入っているか（NFKC 正規化・空白を無視）
- 出典：回答が挙げた出典に、正解のページが1つ以上あるか
- 回答可否：answerable が想定（expected_answerable、既定 true）どおりか
- 正答：答えられる質問は「answerable=true かつキーワードが全部入っている」、
        答えられないはずの質問は「answerable=false」
- 渡していない出典の数、トークン数と料金、応答時間

結果：data/eval/results/<日時>_<label>.json（全記録）と .md（表）に保存する。

注意：キーワードでの正答判定は目安。言い換えや部分的な誤りは拾えないので、表の回答を目でも確認する
（拡張：LLM に採点させる LLM-as-a-judge）。
LLM の出力は実行ごとに揺れるので、比べるときは複数回流す。
"""

import argparse
import json
import time
import unicodedata
from datetime import datetime, timedelta, timezone

from langchain_core.callbacks import get_usage_metadata_callback

from ai_guideline_rag.answer import Source
from ai_guideline_rag.config import EMBEDDING_MODEL, LLM_MODEL, ROOT_DIR
from ai_guideline_rag.embeddings import get_embeddings
from ai_guideline_rag.evaluation import load_questions, targets
from ai_guideline_rag.index import load_index
from ai_guideline_rag.pipeline import PipelineResult, ask
from ai_guideline_rag.retriever import DEFAULT_K
from ai_guideline_rag.splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE

# Claude Haiku 5.5 の料金（100万トークンあたりのドル。2026年10月時点、入力10万トークン以下）
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}
USD_JPY = 150  # 円換算の目安
RESULTS_DIR = ROOT_DIR / "data" / "eval" / "results"
JST = timezone(timedelta(hours=9))


def norm(text: str) -> str:
    return "".join(unicodedata.normalize("NFKC", text).split())


def answer_text(result: PipelineResult) -> tuple[str, list[Source], bool]:
    """（回答の全文, 挙げた出典, answerable）"""
    if result.single:
        a = result.single.answer
        return a.answer, a.sources, a.answerable
    a = result.diff.answer
    # 変更点の種類（新設・削除・変更）も採点の対象に含める
    parts = [a.summary] + [
        f"{c.topic}（{c.change_type}） {c.old} {c.new} {c.point}" for c in a.changes
    ]
    sources = [s for c in a.changes for s in c.old_sources + c.new_sources]
    return "\n".join(parts), sources, a.answerable


def passed_docs(result: PipelineResult, version: str):
    """LLM に渡したチャンクのうち、指定の版のもの（順位順）。"""
    docs = result.single.docs if result.single else result.diff.docs.get(version, [])
    return [d for d, _ in docs if d.metadata["version"] == version]


def evaluate(store, q: dict) -> dict:
    expected_kind = "diff" if q["type"] == "diff" else "single"
    expected_answerable = q.get("expected_answerable", True)

    start = time.perf_counter()
    with get_usage_metadata_callback() as usage_cb:
        result = ask(store, q["question"])
    seconds = time.perf_counter() - start
    usage = {"input": 0, "output": 0}
    for u in usage_cb.usage_metadata.values():
        usage["input"] += u.get("input_tokens", 0)
        usage["output"] += u.get("output_tokens", 0)
    cost_usd = sum(usage[t] * PRICE_PER_MTOK[t] / 1_000_000 for t in usage)

    route_ok = result.kind == expected_kind and (
        expected_kind == "diff" or result.version == q.get("version", "1.2")
    )

    retrieval = []
    for version, pages in targets(q):
        ranks = [
            i for i, d in enumerate(passed_docs(result, version), 1) if d.metadata["page"] in pages
        ]
        retrieval.append(
            {"version": version, "pages": sorted(pages), "rank": ranks[0] if ranks else None}
        )

    text, sources, answerable = answer_text(result)
    keywords = q.get("must_include", [])
    missing = [k for k in keywords if norm(k) not in norm(text)]
    expected_sources = {(s["version"], s["page"]) for s in q["sources"]}
    cited = {(s.version, s.page) for s in sources}
    unknown = (result.single or result.diff).unknown_sources

    if expected_answerable:
        correct = answerable and not missing
    else:
        correct = not answerable

    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "expected_answerable": expected_answerable,
        "route": {
            "kind": result.kind,
            "version": result.version,
            "search_query": result.route.search_query,
            "ok": route_ok,
        },
        "route_ok": route_ok,
        "retrieval": retrieval,
        "retrieval_ok": all(r["rank"] for r in retrieval) if retrieval else None,
        "answerable": answerable,
        "answerable_ok": answerable == expected_answerable,
        "missing_keywords": missing,
        "source_ok": bool(cited & expected_sources) if expected_sources else None,
        "cited_sources": sorted(cited),
        "unknown_sources": len(unknown),
        "correct": correct,
        "answer": text,
        "expected_answer": q["expected_answer"],
        "tokens": usage,
        "cost_usd": cost_usd,
        "seconds": round(seconds, 2),
    }


def rate(records: list[dict], key: str) -> str:
    values = [r[key] for r in records if r[key] is not None]
    return f"{sum(values)}/{len(values)}" if values else "-"


def summarize(records: list[dict]) -> list[str]:
    groups = {
        "全体": records,
        "単一の版": [r for r in records if r["type"] == "single" and r["expected_answerable"]],
        "版の差分": [r for r in records if r["type"] == "diff"],
        "答えられないはず": [r for r in records if not r["expected_answerable"]],
    }
    lines = [
        "| 区分 | 問数 | 正答 | 判定 | 検索（上位k件） | 出典 | 回答可否 | 平均秒 | 1問の料金 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, rs in groups.items():
        if not rs:
            continue
        sec = sum(r["seconds"] for r in rs) / len(rs)
        yen = sum(r["cost_usd"] for r in rs) / len(rs) * USD_JPY
        lines.append(
            f"| {name} | {len(rs)} | {rate(rs, 'correct')} | {rate(rs, 'route_ok')} | "
            f"{rate(rs, 'retrieval_ok')} | {rate(rs, 'source_ok')} | {rate(rs, 'answerable_ok')} | "
            f"{sec:.1f} | {yen:.2f}円 |"
        )
    return lines


def detail(records: list[dict]) -> list[str]:
    def mark(v):
        return "-" if v is None else ("◯" if v else "✕")

    lines = [
        "| ID | 正答 | 判定 | 検索の順位 | 出典 | 不足キーワード | 秒 | 検索に使った文 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        ranks = " ".join(f"{x['version']}:{x['rank'] or '圏外'}" for x in r["retrieval"]) or "-"
        route = f"{mark(r['route']['ok'])} {r['route']['kind']}"
        if r["route"]["version"]:
            route += f" {r['route']['version']}"
        missing = r.get("error") or "、".join(r["missing_keywords"]) or "-"
        lines.append(
            f"| {r['id']} | {mark(r['correct'])} | {route} | {ranks} | {mark(r['source_ok'])} | "
            f"{missing} | {r['seconds']} | {r['route']['search_query']} |"
        )
    return lines


parser = argparse.ArgumentParser(description="評価用の質問を通しで流して結果を記録する")
parser.add_argument("--ids", nargs="+", help="流す質問の ID（省略時は全問）")
parser.add_argument("--label", default="run", help="結果ファイル名に付ける名前")
args = parser.parse_args()

store = load_index(get_embeddings())
questions = [q for q in load_questions() if not args.ids or q["id"] in args.ids]


def error_record(q: dict, error: Exception, seconds: float) -> dict:
    """エラーになった質問の記録。誤答として数える（集計から外すと正答率が実際より良く見えるため）。
    料金は途中まで消費した分を取れないので 0 として扱う。"""
    has_targets = bool(q["sources"])
    return {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "expected_answerable": q.get("expected_answerable", True),
        "error": f"エラー：{type(error).__name__}",
        "error_detail": str(error)[:500],
        "route": {"kind": "-", "version": None, "search_query": "-", "ok": False},
        "route_ok": False,
        "retrieval": [],
        "retrieval_ok": False if has_targets else None,
        "answerable": None,
        "answerable_ok": False,
        "missing_keywords": [],
        "source_ok": False if has_targets else None,
        "cited_sources": [],
        "unknown_sources": 0,
        "correct": False,
        "answer": "",
        "expected_answer": q["expected_answer"],
        "tokens": {"input": 0, "output": 0},
        "cost_usd": 0.0,
        "seconds": round(seconds, 2),
    }


records = []
for q in questions:
    start = time.perf_counter()
    try:
        r = evaluate(store, q)
    except Exception as e:  # 1問の失敗で全体を止めず、誤答として記録する
        r = error_record(q, e, time.perf_counter() - start)
        print(f"[{q['id']}] {r['error']}: {r['error_detail'][:200]}")
    records.append(r)
    print(f"[{r['id']}] 正答={'◯' if r['correct'] else '✕'} {r['seconds']}秒")

total_usd = sum(r["cost_usd"] for r in records)
config = {
    "llm": LLM_MODEL,
    "embedding": EMBEDDING_MODEL,
    "chunk_size": DEFAULT_CHUNK_SIZE,
    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
    "k": DEFAULT_K,
    "questions": len(records),
}
now = datetime.now(JST)
report = [
    f"# 評価結果 {now:%Y-%m-%d %H:%M}（{args.label}）",
    "",
    "設定：" + "、".join(f"{k}={v}" for k, v in config.items()),
    "",
    "## 区分ごとの集計",
    "",
    *summarize(records),
    "",
    f"合計料金：${total_usd:.4f}（約{total_usd * USD_JPY:.1f}円）",
    "",
    "## 質問ごとの結果",
    "",
    *detail(records),
]
print("\n" + "\n".join(report))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
stem = RESULTS_DIR / f"{now:%Y%m%d-%H%M}_{args.label}"
stem.with_suffix(".json").write_text(
    json.dumps({"config": config, "records": records}, ensure_ascii=False, indent=2), "utf-8"
)
stem.with_suffix(".md").write_text("\n".join(report) + "\n", "utf-8")
print(f"\n保存：{stem}.json / .md")
