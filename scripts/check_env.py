"""開発環境の疎通確認。

    uv run python scripts/check_env.py

PDF・Claude・Workers AI・Tavily を順に確認する。キーが未設定の項目はスキップする。
API呼び出しは各1回だけで、費用はほぼかからない。
"""

import os
import sys
import unicodedata

from pypdf import PdfReader

from ai_guideline_rag.config import (
    EMBEDDING_MODEL,
    GUIDELINE_PDFS,
    LLM_MODEL,
    PRINTED_PAGE_OFFSET,
)

results: list[tuple[str, str]] = []


def check_pdfs() -> None:
    for version, path in GUIDELINE_PDFS.items():
        reader = PdfReader(path)
        text = reader.pages[10].extract_text() or ""
        printed = 11 + PRINTED_PAGE_OFFSET
        assert text.strip().startswith(str(printed)), "ページ番号のずれが想定と違う"
        changed = sum(1 for c in text if unicodedata.normalize("NFKC", c) != c)
        msg = f"OK {len(reader.pages)}ページ / NFKCで変わる文字 {changed}個（p11）"
        results.append((f"PDF 第{version}版", msg))


def check_claude() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        results.append(("Claude", "SKIP ANTHROPIC_API_KEY 未設定"))
        return
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=LLM_MODEL, max_tokens=1024, output_config={"effort": "low"})
    reply = llm.invoke("「疎通確認OK」とだけ返してください。")
    results.append(("Claude", f"OK {LLM_MODEL}: {reply.text.strip()}"))


def check_embeddings() -> None:
    if not (os.getenv("CF_ACCOUNT_ID") and os.getenv("CF_AI_API_TOKEN")):
        results.append(("Workers AI", "SKIP CF_ACCOUNT_ID / CF_AI_API_TOKEN 未設定"))
        return
    from langchain_cloudflare import CloudflareWorkersAIEmbeddings

    emb = CloudflareWorkersAIEmbeddings(model_name=EMBEDDING_MODEL)
    vec = emb.embed_query("AI事業者ガイドライン")
    results.append(("Workers AI", f"OK {EMBEDDING_MODEL}: {len(vec)}次元"))


def check_tavily() -> None:
    if not os.getenv("TAVILY_API_KEY"):
        results.append(("Tavily", "SKIP TAVILY_API_KEY 未設定（Phase 2 まで不要）"))
        return
    from langchain_tavily import TavilySearch

    res = TavilySearch(max_results=1).invoke("AI事業者ガイドライン 第1.2版")
    results.append(("Tavily", f"OK {len(res.get('results', []))}件"))


def main() -> int:
    failed = False
    for check in (check_pdfs, check_claude, check_embeddings, check_tavily):
        try:
            check()
        except Exception as e:  # 疎通確認なので種類を問わず表示して続行する
            failed = True
            results.append((check.__name__, f"NG {type(e).__name__}: {e}"))
    for name, msg in results:
        print(f"[{name}] {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
