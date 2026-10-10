"""loader の文字整形とページ番号のテスト。PDFを使わずに確認できる部分だけを扱う。"""

from ai_guideline_rag.loader import normalize_text, printed_page


def test_printed_page_cover_is_zero():
    # 通る例   : PDFの1枚目（位置0）→ 0（表紙）、位置10 → 10
    # はじく例 : 位置10 → 11  … PDFのページラベルをそのまま使うと1ずれる
    assert printed_page(0) == 0
    assert printed_page(10) == 10


def test_normalize_kangxi_radicals():
    # 通る例   : "令和7年3月28日"   … 通常の漢字「月」(U+6708)「日」(U+65E5)
    # はじく例 : "令和7年3⽉28⽇"   … 見た目は同じ康熙部首(U+2F49, U+2F47)のまま。検索で一致しない
    assert normalize_text("令和7年3⽉28⽇") == "令和7年3月28日"


def test_join_wrapped_japanese_lines():
    # 通る例   : "各主体が取り組む"     … PDFの折り返し位置の改行が消える
    # はじく例 : "各主\n体が取り組む"   … 単語の途中に改行が残り、分割の切れ目になりうる
    assert normalize_text("各主\n体が取り組む") == "各主体が取り組む"


def test_keep_line_break_after_sentence_and_before_bullet():
    # 通る例   : "記載する。\n次の文"、"全般\n➢ 学習及び評価の手法"   … 文末・箇条書きの改行は残る
    # はじく例 : "記載する。次の文"、"全般➢ 学習及び評価の手法"     … つなげすぎて構造が消える
    assert normalize_text("記載する。\n次の文") == "記載する。\n次の文"
    assert normalize_text("全般\n➢ 学習及び評価の手法") == "全般\n➢ 学習及び評価の手法"
