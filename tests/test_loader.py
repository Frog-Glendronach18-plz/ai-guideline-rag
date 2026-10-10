from ai_guideline_rag.loader import normalize_text, printed_page


def test_printed_page_cover_is_zero():
    assert printed_page(0) == 0
    assert printed_page(10) == 10


def test_normalize_kangxi_radicals():
    # 康熙部首の「⽉」「⽇」(U+2F49, U+2F47) が通常の漢字になる
    assert normalize_text("令和7年3⽉28⽇") == "令和7年3月28日"


def test_join_wrapped_japanese_lines():
    assert normalize_text("各主\n体が取り組む") == "各主体が取り組む"


def test_keep_line_break_after_sentence_and_before_bullet():
    assert normalize_text("記載する。\n次の文") == "記載する。\n次の文"
    assert normalize_text("全般\n➢ 学習及び評価の手法") == "全般\n➢ 学習及び評価の手法"
