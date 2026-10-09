# AI事業者ガイドライン RAG Q&A

総務省・経済産業省「AI事業者ガイドライン」（第1.1版・第1.2版）に、版とページ番号の出典付きで答えるRAGデモ。LangChain の学習用。

> **開発中**（2026年10月 公開予定）。計画と進捗は [docs/roadmap.md](docs/roadmap.md)、評価用の質問は [data/eval/questions.json](data/eval/questions.json) を参照。

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

## 開発コマンド

```bash
uv run ruff check .
uv run ruff format .
uv run pytest
```
