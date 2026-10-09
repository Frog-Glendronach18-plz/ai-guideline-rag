"""プロジェクト共通の設定（パス・題材・モデル名）。"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT_DIR / "data" / "pdf"
INDEX_PATH = ROOT_DIR / "data" / "index.json"

# 検索対象にする版。キーはメタデータ `version` に入れる値。
GUIDELINE_PDFS: dict[str, Path] = {
    "1.1": PDF_DIR / "20250328_1.pdf",
    "1.2": PDF_DIR / "20260331_1.pdf",
}
LATEST_VERSION = "1.2"

# 差分の正解データ（第1.1版からの変更履歴付き）。索引には入れない。
# 削除・追加された文字がテキスト抽出で混ざるため、評価用の質問は目で読んで作る。
DIFF_ANSWER_PDF = PDF_DIR / "diff_ans_20260331_10.pdf"

# 表紙にページ番号がないため、印刷されたページ番号 = PDFのページ位置(1始まり) - 1
PRINTED_PAGE_OFFSET = -1

LLM_MODEL = "claude-haiku-5-5"
EMBEDDING_MODEL = "@cf/baai/bge-m3"
