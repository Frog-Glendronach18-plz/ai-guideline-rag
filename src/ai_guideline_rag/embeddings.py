"""③ 埋め込みモデル（Cloudflare Workers AI の bge-m3）。

文章を 1024 次元のベクトルに変換する。意味が近い文章ほど、ベクトルの向きが近くなる。
アカウントIDとトークンは環境変数 CF_ACCOUNT_ID / CF_AI_API_TOKEN から読む
（config で .env を読み込む）。

索引を作るとき（チャンク）と検索するとき（質問）で、必ず同じモデルを使う。

--- 選定理由 ------------------------------------------------------------------

Cloudflare Workers AI の bge-m3 を採用した。
  - 費用：無料枠に収まり、API の有料契約を LLM（Claude）の1つに抑えられる
  - 日本語：多言語対応のオープンモデルで評価が高く、日本語の検索でもよく使われる
  - 既存の Cloudflare アカウントで使える
主な理由はコストで、PoC のデファクトというわけではない。
  - 一般的な PoC・チュートリアルの定番：OpenAI text-embedding-3-small
  - 日本の企業：契約中のクラウドで完結するもの（Azure OpenAI、AWS Bedrock の Cohere・Titan など）
  - データを外に出せない場合：bge-m3、multilingual-e5、Ruri などを自前で動かす
本番で使う場合は、利用中のクラウドとデータの扱いの要件に合わせて選び直す前提。
LangChain の共通インターフェースにより、get_embeddings() の差し替えだけで切り替えられる。
注意点：langchain-cloudflare は利用者が少なく情報が見つけにくい。Workers AI の回数制限・
仕様変更の影響を受ける。
拡張候補：OpenAI text-embedding-3-small と検索のヒット率を比べ、「比較して選んだ」状態にする。

--- 学習メモ ------------------------------------------------------------------

■ このクラスがやっていること
  CloudflareWorkersAIEmbeddings 自体は計算しない。文章を Cloudflare のサーバーに送り、
  そこで動く bge-m3 モデルが返したベクトルを受け取るだけの「窓口」。
  LangChain の埋め込みクラスはどれも embed_documents()（チャンク用）と embed_query()（質問用）
  を持つので、get_embeddings() の1行を差し替えれば他のサービス・モデルに切り替えられる。

■ 費用（Workers AI 無料プラン）
  1日 10,000 Neurons まで無料（UTC 0時リセット）。bge-m3 は 100万トークンで 1,075 Neurons
  → 1日約900万トークンまで無料。索引作成（約10万文字）は無料枠の1〜2%、質問1回はほぼゼロ。
  無料プランで枠を超えるとその日はエラーになるだけで、課金はされない。

■ 埋め込みの手法
  - 密ベクトル（Dense）：ニューラルネットが文章の意味を数百〜数千次元の数値にする。
    言い換えに強い（「公開」と「開示」が近くなる）。今の主流で、今回もこれ。
  - 疎ベクトル（Sparse）／キーワード：BM25、TF-IDF など単語の出現に基づく。
    数値・固有名詞の完全一致に強い（「25組織」「第53号」）。
  - 実務では両者を組み合わせる「ハイブリッド検索」が定番（ロードマップの拡張候補）。

■ モデルの使い方は2通り
  - APIサービス：OpenAI、Cohere、Voyage AI、Google、Cloudflare Workers AI など。
    手軽だが、文章が外部に送られ、従量課金。
  - 自前で動かす：Hugging Face のモデル（bge-m3、multilingual-e5、日本語特化の Ruri など）を
    sentence-transformers で動かす。データを外に出さないが、PyTorch が必要で重い
    （Render 無料プランのメモリ 512MB では厳しい）。

■ モデルの選び方
  言語（日本語なら多言語対応か日本語特化。JMTEB などの評価が参考）、データの扱い（外部APIに
  送ってよいか）、費用と速さ、次元数（保存容量）と最大入力長（チャンクより短いと切り捨て）。

■ モデルを替えたら索引も作り直す
  違うモデルのベクトル同士は比べられない（数値の意味がモデルごとに違う）。
  モデルを替えたら build_index.py を実行し直す。
"""

from langchain_cloudflare import CloudflareWorkersAIEmbeddings
from langchain_core.embeddings import Embeddings

from ai_guideline_rag.config import EMBEDDING_MODEL


def get_embeddings() -> Embeddings:
    return CloudflareWorkersAIEmbeddings(model_name=EMBEDDING_MODEL)
