"""LangChain 練習スクリプト（Phase 1 午前）。

    python -m uv run python scripts/hello_lang.py

RAG の回答生成で使う3つの部品を、順に試す。
  1. チャットモデルを直接呼ぶ
  2. プロンプトテンプレートで、質問と参照テキストを差し込む
  3. 構造化出力で、「回答」と「出典ページ」を決まった形で受け取る
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

# 文字列でモデルを指定すると、"claude-" から Anthropic 用のクラスが選ばれる。
# Haiku 5.5 は思考（thinking）が既定でオンなので、練習では effort を low にして速く・安くする。
model = init_chat_model(
    "claude-haiku-5-5",
    max_tokens=2048,
    output_config={"effort": "low"},
)

# RAG で検索結果として渡される想定の参照テキスト（第1.2版 p20 の抜粋）
# 今回の２つの版で約10万字、7万トークン前後、
# PDF全体をそのままCONTEXTに入れてもClaudeの入力上限に収まる。
# ただし、RAGを使えば、文章量の拡張性、トークン・コスト、出典の追跡しやすさで優位
CONTEXT = """[第1.2版 p20]
上記の「2関連するステークホルダーへの情報提供」は、アルゴリズム又はソースコードの開示を
必ずしも想定するものではなく、プライバシー及び営業秘密を尊重して、採用する技術の特性
及び用途に照らし、社会的合理性が認められる範囲で実施する"""

QUESTION = "透明性を確保するには、ソースコードを公開しなければなりませんか？"


def section(title: str) -> None:
    print(f"\n===== {title} =====")


# --- 1. チャットモデルを直接呼ぶ -------------------------------------------
section("1. チャットモデルを直接呼ぶ")

reply = model.invoke("RAG（検索拡張生成）を一文で説明してください。")
print(reply.text)  # 本文。思考ブロックなどを除いたテキストだけが取れる
print("トークン数:", reply.usage_metadata)
# 比較用
reply_only_question = model.invoke(QUESTION)
print("コンテキストなし回答:", reply_only_question.text)


# --- 2. プロンプトテンプレート -----------------------------------------------
section("2. プロンプトテンプレート")

# {context} と {question} が後から差し込まれる穴になる
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "あなたはAI事業者ガイドラインに詳しいアシスタントです。"
            "参照テキストに書かれている内容だけを根拠に、日本語で簡潔に答えてください。"
            "参照テキストで答えられない場合は「参照テキストからは分かりません」と答えてください。",
        ),
        ("human", "参照テキスト:\n{context}\n\n質問: {question}"),
    ]
)

# 差し込み後のメッセージを確認（LLMに実際に渡る内容。ここではまだLLMを呼ばない）
print("--- LLMへの入力（先頭60文字） ---")
for m in prompt.invoke({"context": CONTEXT, "question": QUESTION}).to_messages():
    preview = m.content[:60].replace("\n", " / ")
    print(f"[{m.type}] {preview}...")
print("--- LLMの出力 ---")

# `|` でつなぐと「テンプレート → モデル」の処理の流れ（チェーン）になる
chain = prompt | model
print("\n回答:", chain.invoke({"context": CONTEXT, "question": QUESTION}).text)

# 参照テキストにないことを聞くと、どう答えるか
print(
    "回答（範囲外の質問）:",
    chain.invoke({"context": CONTEXT, "question": "AI推進法の法律番号は？"}).text,
)


# --- 3. 構造化出力 ------------------------------------------------------------
section("3. 構造化出力")


# 受け取りたい形を Pydantic のモデルで定義する。
# Field の description は LLM への説明として渡されるので、具体的に書く。
class Answer(BaseModel):
    answer: str = Field(description="質問への回答（日本語、3文以内）")
    pages: list[int] = Field(description="根拠にした参照テキストのページ番号。根拠がなければ空")
    answerable: bool = Field(description="参照テキストだけで答えられたか")


structured_chain = prompt | model.with_structured_output(Answer)

result = structured_chain.invoke({"context": CONTEXT, "question": QUESTION})
print(type(result).__name__, "型で返ってくる:")
print("  answer    :", result.answer)
print("  pages     :", result.pages)
print("  answerable:", result.answerable)

result = structured_chain.invoke({"context": CONTEXT, "question": "AI推進法の法律番号は？"})
print("範囲外の質問:", result.model_dump())
