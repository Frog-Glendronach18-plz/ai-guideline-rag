"""Gradio の画面。質問すると、AI事業者ガイドラインの版・ページの出典付きで回答する。
処理は pipeline.ask()（判定 → 単一の版の回答／版の差分の回答）を呼ぶだけ。

    python -m uv run python app.py      # http://localhost:7860

Render では環境変数 PORT で指定されたポートで待ち受ける。

悪用対策（公開URLから自分の API キーで課金されるため）：
- 同時に処理する質問は2件まで、待ち行列は10件まで
- 1日あたりの質問回数の上限（DAILY_LIMIT、既定100回。日付は日本時間）。
  サーバーの再起動でリセットされる
- 質問の長さの上限（MAX_QUESTION_LENGTH 文字）
- Anthropic Console 側でも月の利用上限額を設定済み
"""

import logging
import os
import threading
from datetime import datetime, timedelta, timezone

# Gradio の利用統計（gradio.app・huggingface.co への送信）を止める。import より前に設定する
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from ai_guideline_rag.answer import Source  # noqa: E402
from ai_guideline_rag.embeddings import get_embeddings  # noqa: E402
from ai_guideline_rag.index import load_index  # noqa: E402
from ai_guideline_rag.pipeline import PipelineResult, ask  # noqa: E402
from ai_guideline_rag.retriever import source_label  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "100"))
MAX_QUESTION_LENGTH = 300
JST = timezone(timedelta(hours=9))
SOURCE_URL = (
    "https://www.meti.go.jp/shingikai/mono_info_service/ai_shakai_jisso/20260331_report.html"
)
GITHUB_URL = "https://github.com/Frog-Glendronach18-plz/ai-guideline-rag"

# 画面の選択肢 → pipeline のモード。自動判定では、質問の内容から版・差分を判定する
MODE_CHOICES = {
    "自動判定": "auto",
    "第1.2版（最新）": "1.2",
    "第1.1版": "1.1",
    "版の差分": "diff",
}

EXAMPLES = [
    "広島AIプロセスの国際行動規範の「報告枠組み」に回答を提出・公表した組織はいくつありますか。"
    "そのうち日本企業は何社ですか。",
    "RAG（検索拡張生成）を使えばハルシネーションを抑えられるので、どんな業務でもRAGを使うのが望ましいですか。",
    "ガイドラインの「透明性」に従うと、AIシステムのアルゴリズムやソースコードを公開しなければなりませんか。",
    "第2部「D.」の章は、第1.1版から第1.2版でどう変わりましたか。"
    "対象となる事業者の範囲に注目して説明してください。",
    "第1.2版では「AIエージェント」はどのように扱われるようになりましたか。",
]

ANSWER_PLACEHOLDER = (
    "質問を入力して「質問する」を押すと、ここに回答と出典（版・ページ）が表示されます。"
)

# 起動時に1回だけ索引を読み込む（チャンクの埋め込みは作り直さない）
store = load_index(get_embeddings())


class DailyCounter:
    """日本時間の日付ごとに回数を数える。複数の質問が同時に来ても数え間違えないようロックする。"""

    def __init__(self, limit: int):
        self.limit = limit
        self.date = ""
        self.count = 0
        self.lock = threading.Lock()

    def try_acquire(self) -> bool:
        today = datetime.now(JST).strftime("%Y-%m-%d")
        with self.lock:
            if today != self.date:
                self.date, self.count = today, 0
            if self.count >= self.limit:
                return False
            self.count += 1
            return True


counter = DailyCounter(DAILY_LIMIT)


def _pages(sources: list[Source]) -> str:
    return "、".join(f"第{s.version}版 p{s.page}" for s in sources) or "なし"


def _render(result: PipelineResult) -> tuple[str, str]:
    """pipeline の結果を（回答の Markdown, 参照テキストの Markdown）にする。"""
    if result.kind == "diff":
        a = result.diff.answer
        parts = [f"**対象：版の差分（第1.1版 → 第1.2版）**\n\n{a.summary}"]
        for i, c in enumerate(a.changes, start=1):
            parts.append(
                f"#### 変更点{i}：{c.topic}（{c.change_type}）\n"
                f"- **第1.1版：** {c.old}（{_pages(c.old_sources)}）\n"
                f"- **第1.2版：** {c.new}（{_pages(c.new_sources)}）\n"
                f"- **要点：** {c.point}"
            )
        answerable, unknown = a.answerable, result.diff.unknown_sources
        docs = [d for results in result.diff.docs.values() for d in results]
    else:
        a = result.single.answer
        parts = [f"**対象：第{result.version}版**\n\n{a.answer}\n\n**出典：** {_pages(a.sources)}"]
        answerable, unknown = a.answerable, result.single.unknown_sources
        docs = result.single.docs

    if not answerable:
        parts.append("※ 参照したガイドラインの範囲では答えられませんでした。")
    if unknown:
        parts.append("※ 検索結果にないページが出典に含まれています。原文で確認してください。")
    parts.append(f"<sub>検索に使った文：{result.route.search_query}</sub>")

    docs_md = "\n\n".join(
        f"**{i}. {source_label(doc)}**（類似度 {score:.3f}）\n\n> "
        + doc.page_content.replace("\n", "\n> ")
        for i, (doc, score) in enumerate(docs, start=1)
    )
    return "\n\n".join(parts), docs_md


def respond(question: str, mode_label: str) -> tuple[str, str]:
    """画面の「質問する」ボタンの処理。（回答の Markdown, 参照テキストの Markdown）を返す。"""
    question = (question or "").strip()
    if not question:
        return "質問を入力してください。", ""
    if len(question) > MAX_QUESTION_LENGTH:
        return f"質問は{MAX_QUESTION_LENGTH}文字以内で入力してください。", ""
    if not counter.try_acquire():
        return "本日の質問回数の上限に達しました。明日（日本時間）以降にお試しください。", ""

    try:
        result = ask(store, question, MODE_CHOICES.get(mode_label, "auto"))
    except Exception:
        logger.exception("回答の生成に失敗しました")
        return "回答の生成に失敗しました。時間をおいて再度お試しください。", ""
    return _render(result)


with gr.Blocks(title="AI事業者ガイドライン Q&A", analytics_enabled=False) as demo:
    gr.Markdown(
        "# AI事業者ガイドライン Q&A（RAGデモ）\n"
        "総務省・経済産業省「AI事業者ガイドライン」（第1.1版・第1.2版 本編）に、"
        "版とページの出典付きで答えます。「版の差分」では、2つの版で何が変わったかを答えます。\n\n"
        f"※ 個人が学習目的で作成した**非公式**のデモです。回答は誤りを含む可能性があるため、"
        f"必ず[公式の原文]({SOURCE_URL})で確認してください。"
        f"　ソースコード・設計・評価：[GitHub]({GITHUB_URL})"
    )
    # 入力：質問欄 → 質問の例（折りたたみ）→ 対象の選択と「質問する」ボタン。
    # 画面の幅が狭くても「質問 → ボタン → 回答」が縦に続くよう、列に分けず上から順に並べる
    question = gr.Textbox(
        label="質問",
        placeholder="例：透明性のためにソースコードを公開する必要はありますか？",
        lines=3,
        max_length=MAX_QUESTION_LENGTH,
    )
    with gr.Accordion("質問の例を見る（クリックで質問欄に入力）", open=False):
        gr.Examples(examples=EXAMPLES, inputs=question, label="質問の例")
    with gr.Row(equal_height=True):
        mode = gr.Radio(list(MODE_CHOICES), value="自動判定", label="対象", scale=3)
        ask_button = gr.Button("質問する", variant="primary", size="lg", scale=1)

    # 出力：見出しの下の枠に回答を表示する。処理中は枠の上に進行状況が出る
    # （Markdown 部品は label を指定しても見出しが表示されないため、見出しは別に置く）
    gr.Markdown("### 回答")
    answer_out = gr.Markdown(ANSWER_PLACEHOLDER, container=True, padding=True, min_height=160)
    with gr.Accordion("検索で見つかった参照テキスト（LLM に渡した内容）", open=False):
        docs_out = gr.Markdown()

    ask_button.click(respond, inputs=[question, mode], outputs=[answer_out, docs_out])
    question.submit(respond, inputs=[question, mode], outputs=[answer_out, docs_out])


if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=2, api_open=False)
    # footer_links から "api" を外し、画面を通さず API で直接呼ぶ方法の案内を表示しない
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        footer_links=["gradio"],
    )
