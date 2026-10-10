"""Gradio の画面。質問すると、AI事業者ガイドラインの版・ページの出典付きで回答する。

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

from ai_guideline_rag.answer import answer_question  # noqa: E402
from ai_guideline_rag.config import LATEST_VERSION  # noqa: E402
from ai_guideline_rag.embeddings import get_embeddings  # noqa: E402
from ai_guideline_rag.index import load_index  # noqa: E402
from ai_guideline_rag.retriever import source_label  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "100"))
MAX_QUESTION_LENGTH = 300
JST = timezone(timedelta(hours=9))
SOURCE_URL = (
    "https://www.meti.go.jp/shingikai/mono_info_service/ai_shakai_jisso/20260331_report.html"
)

VERSION_CHOICES = {"第1.2版（最新）": "1.2", "第1.1版": "1.1"}

EXAMPLES = [
    "広島AIプロセスの国際行動規範の「報告枠組み」に回答を提出・公表した組織はいくつありますか。"
    "そのうち日本企業は何社ですか。",
    "RAG（検索拡張生成）を使えばハルシネーションを抑えられるので、どんな業務でもRAGを使うのが望ましいですか。",
    "ガイドラインの「透明性」に従うと、AIシステムのアルゴリズムやソースコードを公開しなければなりませんか。",
]

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


def respond(question: str, version_label: str) -> tuple[str, str]:
    """画面の「質問する」ボタンの処理。（回答の Markdown, 検索結果の Markdown）を返す。"""
    question = (question or "").strip()
    if not question:
        return "質問を入力してください。", ""
    if len(question) > MAX_QUESTION_LENGTH:
        return f"質問は{MAX_QUESTION_LENGTH}文字以内で入力してください。", ""
    if not counter.try_acquire():
        return "本日の質問回数の上限に達しました。明日（日本時間）以降にお試しください。", ""

    version = VERSION_CHOICES.get(version_label, LATEST_VERSION)
    try:
        result = answer_question(store, question, version)
    except Exception:
        logger.exception("回答の生成に失敗しました")
        return "回答の生成に失敗しました。時間をおいて再度お試しください。", ""

    a = result.answer
    cited = "、".join(f"第{s.version}版 p{s.page}" for s in a.sources) or "なし"
    answer_md = f"{a.answer}\n\n**出典：** {cited}"
    if not a.answerable:
        answer_md += "\n\n※ 参照したガイドラインの範囲では答えられませんでした。"
    if result.unknown_sources:
        answer_md += "\n\n※ 検索結果にないページが出典に含まれています。原文で確認してください。"

    docs_md = "\n\n".join(
        f"**{i}. {source_label(doc)}**（類似度 {score:.3f}）\n\n> "
        + doc.page_content.replace("\n", "\n> ")
        for i, (doc, score) in enumerate(result.docs, start=1)
    )
    return answer_md, docs_md


with gr.Blocks(title="AI事業者ガイドライン Q&A", analytics_enabled=False) as demo:
    gr.Markdown(
        "# AI事業者ガイドライン Q&A（RAGデモ）\n"
        "総務省・経済産業省「AI事業者ガイドライン」（第1.1版・第1.2版 本編）に、"
        "版とページの出典付きで答えます。\n\n"
        f"※ 個人が学習目的で作成した**非公式**のデモです。回答は誤りを含む可能性があるため、"
        f"必ず[公式の原文]({SOURCE_URL})で確認してください。"
    )
    with gr.Row():
        question = gr.Textbox(
            label="質問",
            placeholder="例：透明性のためにソースコードを公開する必要はありますか？",
            lines=3,
            max_length=MAX_QUESTION_LENGTH,
            scale=4,
        )
        version = gr.Radio(
            list(VERSION_CHOICES), value="第1.2版（最新）", label="対象の版", scale=1
        )
    ask = gr.Button("質問する", variant="primary")
    answer_out = gr.Markdown(label="回答")
    with gr.Accordion("検索で見つかった参照テキスト（LLM に渡した内容）", open=False):
        docs_out = gr.Markdown()
    gr.Examples(examples=EXAMPLES, inputs=question, label="質問の例")

    ask.click(respond, inputs=[question, version], outputs=[answer_out, docs_out])
    question.submit(respond, inputs=[question, version], outputs=[answer_out, docs_out])


if __name__ == "__main__":
    demo.queue(max_size=10, default_concurrency_limit=2, api_open=False)
    # footer_links から "api" を外し、画面を通さず API で直接呼ぶ方法の案内を表示しない
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        footer_links=["gradio"],
    )
