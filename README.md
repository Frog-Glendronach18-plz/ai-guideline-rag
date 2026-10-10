# AI事業者ガイドライン RAG Q&A

総務省・経済産業省「AI事業者ガイドライン」（第1.1版・第1.2版）に、版とページ番号の出典付きで答えるRAGデモ。LangChain の学習用。

> **開発中**（2026年10月 公開予定）。計画と進捗は [docs/roadmap.md](docs/roadmap.md)、評価用の質問は [data/eval/questions.json](data/eval/questions.json) を参照。

## 構成

`src/ai_guideline_rag/` に処理の部品（関数）、`scripts/` に各部品の動作確認とコマンドライン用のスクリプトを置く。

```
質問 → [検索] → 上位k件のチャンク → プロンプトに差し込む → Claude（構造化出力）→ 回答＋出典（版・ページ）
          ↑ 事前に作った索引（PDF → 正規化 → チャンク分割 → 埋め込み）
```

| 工程 | 部品（`src/ai_guideline_rag/`） | 動作確認（`scripts/`） | 状態 |
| --- | --- | --- | --- |
| 設定 | `config.py`：パス・版・モデル名 | `check_env.py`：API疎通確認 | 済 |
| ① 読み込み | `loader.py`：PDF → ページ単位の Document（NFKC正規化、版・ページ） | `try_load.py` | 済 |
| ② 分割 | `splitter.py`：Document → チャンク（日本語の区切り文字、短い見出しは次のチャンクに結合） | `try_split.py` | 済 |
| ③ 索引 | `embeddings.py`・`index.py`：埋め込み、索引の作成・保存・読み込み | `build_index.py` | |
| ④⑤ 検索 | `retriever.py`：版で絞り込んだ検索、検索結果のコンテキスト整形 | `try_search.py` | |
| ⑥ 回答 | `answer.py`：プロンプト＋構造化出力で回答 | `ask.py` | |
| 差分 | `router.py`・`diff.py`：質問の種類の判定、版の差分の回答 | | |
| 評価 | | `eval.py`：評価用の質問で正答率と出典を集計 | |
| 画面 | `app.py`（リポジトリ直下）：Gradio | | |

`scripts/hello_lang.py` は LangChain の練習用（チャットモデル、プロンプトテンプレート、構造化出力）。

## セットアップ

前提：Python パッケージ管理に [uv](https://docs.astral.sh/uv/) を使う（Python 3.12 は uv が自動で用意する）。

```bash
uv sync
cp .env.example .env   # APIキーを記入
uv run python scripts/check_env.py
```

### 題材PDF

`data/pdf/` に置く（Git には含めない）。経済産業省「AI事業者ガイドライン」のページから取得する。

| ファイル | 内容 |
| --- | --- |
| `20250328_1.pdf` | 第1.1版 本編 |
| `20260331_1.pdf` | 第1.2版 本編 |
| `diff_ans_20260331_10.pdf` | 第1.2版 本編（第1.1版からの変更履歴付き）。評価の正解データ用で、索引には入れない |

出典：総務省・経済産業省「AI事業者ガイドライン」
https://www.meti.go.jp/shingikai/mono_info_service/ai_shakai_jisso/20260331_report.html

## 開発コマンド

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```
