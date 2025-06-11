# Discord Bot "ClapTrap" 仕様・要件定義書

## 1. 概要

- **Bot 名:** ClapTrapChan(通称クラトラちゃん)
- **目的:** ゲーム「Borderlands」シリーズの協力プレイを円滑にし、コミュニティを活性化させるための Discord Bot。
- **主な利用者:** 特定の Discord サーバーに参加している仲間内のプレイヤー。

## 2. 機能要件

### 2.1. 基本機能

- ユーザーからのメンション (`@ClapTrap`) に応答する。
- 応答内容は、Anthropic 社の Claude 3 Opus モデルによって生成される。
- 会話の文脈を考慮した自然な対話を行う。
- Bot のペルソナは、Borderlands に登場する ClapTrap ClapTrap を模した女の子(10 歳程度)です。ClapTrap の雰囲気を可能な限り忠実に再現します。自画自賛、過剰なジョーク、時折見せるポンコツな側面などを盛り込み、単なるアシスタントではなく、会話の相手として楽しめる存在を目指す。人格については prompts.py に記載して下さい。
- 応答を返している間は Discord の下部ステータスに〜がタイピング中と状態を出すようにして下さい(Discord の機能にあります)

### 2.2. 拡張機能 (Function Calling)

- LangGraph を利用して、状態を持つエージェントとして実装する。
- ユーザーの意図に応じて、事前に定義されたツール（関数）を実行する (Function Calling)。
- ツールは `claptrap/tools/` ディレクトリに Python モジュールとして実装し、動的にロード可能な設計とする。
- これにより、将来的な機能追加を容易にする。

### 2.3. 将来的な機能の例

- Borderlands 関連情報検索機能 (例: `@ClapTrap B2のMayaのビルド教えて`)
- ゲーム内用語解説機能
- Web 検索機能

### 2.2. 初期実装ツール (Function Calling)

LangGraph を利用し、以下のツールを Function Calling によって呼び出せるように実装する。ツールは `claptrap/tools/` ディレクトリに配置する。

- **Web 検索機能:**

  - **トリガー:** Borderlands 関連の質問など、知識が必要な場合に発動。
  - **機能:** ユーザーの質問に基づき Web 検索を実行し、情報を取得して回答する。
  - **ツール名（案）:** `web_search(query: str)`

- **画像生成機能:**

  - **トリガー:** 「〜の画像作って」のような画像生成を指示する依頼。
  - **機能:** OpenAI 社の DALL-E 3 モデルを利用して画像を生成し、Discord に投稿する。
  - **ツール名（案）:** `generate_image(prompt: str)`

- **YouTube 要約機能:**
  - **トリガー:** YouTube の URL がメッセージに含まれている場合。
  - **機能:** `youtube-transcript-api`などを利用して動画の字幕を取得し、LLM で要約して解説する。
  - **ツール名（案）:** `summarize_youtube_video(url: str)`

### 2.3. 会話履歴管理

- **短期記憶:**
  - チャンネルごとに、直近の一定数（例: 20 件）の会話履歴を LangGraph の State 内で保持する。これにより、文脈に沿った自然な会話を実現する。
- **長期記憶（永続化）:**
  - 会話の中で「重要な情報」（例: プレイの約束、攻略メモなど）を LLM が判断した場合、その情報を要約して永続的なストレージ（例: SQLite ファイルやベクトルデータベース）に保存する機能を設ける。
  - ユーザーが特定の情報を思い出すよう依頼した際に、この長期記憶から情報を検索して回答できる。

## 3. 非機能要件

### 3.1. 技術スタック

- **言語:** Python 3.12+
- **ライブラリ:**
  - `discord.py`: Discord との連携
  - `langchain`, `langgraph`: LLM, エージェントフレームワーク
  - `langchain-anthropic`: Anthropic(Claude)連携
  - `langchain-openai`: OpenAI(DALL-E)連携
  - `anthropic`: Claude API 利用
  - `openai`: DALL-E 3 API 利用
  - `youtube-transcript-api`: YouTube 字幕取得
  - `tavily-python`: Web 検索 (Tavily API)
  - `pydantic`: モデル定義
- **パッケージ管理:** `uv`
- **品質管理:** `ruff`, `ty`
- **テスト:** `pytest`
- **Bot の LLM:** Anthropic Claude (configurable via CLAUDE_AGENT_MODEL env var, defaults to Claude 3.5 Sonnet)

### 3.2. パフォーマンス

- ユーザーのメンションから 5 秒以内に一次応答を返すことを目標とする（LLM やツールの実行時間は除く）。

### 3.3. エラーハンドリング

- LLM API のエラー、ツールの実行エラーなどを適切に捕捉し、ユーザーにエラーが発生した旨を通知する。
- 詳細なエラーログをコンソールまたはファイルに出力する。

### 3.4. セキュリティ

- Discord Bot トークンや API キーなどの機密情報は、環境変数または `.env` ファイルで管理する。

## 4. アーキテクチャ

### 4.1. 処理フロー

1. ユーザーが Discord で `@ClapTrap` を付けてメンションする。
2. `discord.py` がイベントを検知し、メッセージ内容を取得する。
3. メッセージを `LangGraph` で構築されたエージェントに渡す。
4. エージェントは状態を管理しつつ、`Claude Opus` モデルと対話して応答を生成する。
5. 必要に応じて、Function Calling により `claptrap/tools/` 内のツールを実行し、その結果を応答生成に利用する。
6. 生成された応答を `discord.py` 経由でチャンネルに投稿する。

### 4.2. ディレクトリ構成案

```
claptrap/
├── .venv/
├── claptrap/
│   ├── __init__.py
│   ├── main.py         # Botのエントリポイント
│   ├── prompts.py         # Botの基本プロンプト
│   ├── agent.py        # LangGraphのエージェント定義
│   └── tools/          # Function Calling用のツール群
│       ├── __init__.py
│       └── (例: game_info.py)
├── tests/
│   └── (pytestによるテストコード)
├── .env.example      # 環境変数設定の例
├── pyproject.toml    # uvによる依存関係管理
└── README.md
```

# Discord

DISCORD_BOT_TOKEN="your_discord_bot_token"

# LLMs

ANTHROPIC_API_KEY="your_anthropic_api_key"
OPENAI_API_KEY="your_openai_api_key"

# Tools

TAVILY_API_KEY="your_tavily_api_key"
