from ai_guideline_rag.config import GUIDELINE_PDFS, LATEST_VERSION


def test_latest_version_is_indexed():
    assert LATEST_VERSION in GUIDELINE_PDFS
