# AI事業者ガイドライン RAG Q&A

総務省・経済産業省「AI事業者ガイドライン」（第1.1版・第1.2版）に、版とページ番号の出典付きで答えるRAGデモ。LangChain の学習用。

## デモ

**[https://ai-guideline-rag.onrender.com/](https://ai-guideline-rag.onrender.com/)**

- 質問すると、回答と出典（版・ページ）を返します。質問の内容から、単一の版への質問か版の差分を聞く質問かを自動で判定します（第1.2版／第1.1版／版の差分を指定することもできます）
- 無料プランのため、しばらくアクセスがないと最初の表示に1分ほどかかります
- 個人が学習目的で作成した非公式のデモです。回答は必ず公式の原文で確認してください

> **開発中**（評価のベースラインを記録済み。改善とREADMEの仕上げを予定）。計画と進捗は [docs/roadmap.md](docs/roadmap.md)、評価用の質問は [data/eval/questions.json](data/eval/questions.json) を参照。

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
| ③ 索引 | `embeddings.py`・`index.py`：埋め込み（Workers AI bge-m3）、索引の作成・保存・読み込み（`data/index.json`） | `build_index.py` | 済 |
| ④⑤ 検索 | `retriever.py`：版で絞り込んだ検索（上位5件）、検索結果のコンテキスト整形 | `try_search.py` | 済 |
| ⑥ 回答 | `answer.py`：プロンプト＋構造化出力で回答（回答・出典・回答可否）、渡していない出典の検出 | `ask.py` | 済 |
| 判定 | `router.py`：質問の種類（単一の版／差分）の判定と、検索用の文の書き換え（目次を渡して章番号を見出しの言葉に。版の名前は除く） | `try_route.py` | 済 |
| 差分 | `diff.py`：版ごとに検索し、変更点ごとに旧／新／要点と出典を回答 | `try_diff.py` | 済 |
| 流れ | `pipeline.py`：判定 → 単一の版の回答／版の差分の回答。画面と `ask.py` はここを呼ぶ | `ask.py` | 済 |
| 評価 | `evaluation.py`：評価用の質問の読み込み、正解ページの順位 | `eval.py`：評価用の質問（25問）を通しで流し、正答・判定・検索・出典・料金・応答時間を記録 | 済 |
| 画面 | `app.py`（リポジトリ直下）：Gradio。同時処理2件・1日100回・質問300文字までの制限、利用統計の送信なし | `python -m uv run python app.py` | 済 |

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

## 評価

評価用の質問25問（`data/eval/questions.json`）を `scripts/eval.py` で通しで流した結果（2026-10-11、改善前のベースライン）：

| 区分 | 問数 | 正答 | 検索（上位5件に正解のページ） | 1問の料金 | 平均秒 |
| --- | --- | --- | --- | --- | --- |
| 全体 | 25 | 20/25 | 22/23 | 0.12円 | 4.1 |
| 単一の版 | 12 | 8/12 | 12/12 | 0.09円 | 3.1 |
| 版の差分 | 11 | 10/11 | 10/11 | 0.15円 | 5.4 |
| 答えられないはずの質問 | 2 | 2/2 | - | 0.08円 | 2.9 |

誤答5問はいずれも、誤った内容を言い切ったものではなく「分からない」と答えたか、エラーだった。原因と改善策は [docs/roadmap.md](docs/roadmap.md) の「② 評価」、全記録は `data/eval/results/` を参照。

## デプロイ（Render）

GitHub の `main` へ push すると自動でデプロイされる。`uv.lock` があるため Render が uv を用意し、Python は `.python-version`（3.12）を使う。

| 項目 | 設定値 |
| --- | --- |
| Build Command | `uv sync --frozen --no-dev` |
| Start Command | `uv run --no-sync python app.py` |
| Environment Variables | `ANTHROPIC_API_KEY`、`CF_ACCOUNT_ID`、`CF_AI_API_TOKEN`（任意：`DAILY_LIMIT`） |

索引（`data/index.json`）はリポジトリに含めているので、デプロイ時に埋め込みを作り直さない。PDF は不要。

## 開発コマンド

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```
