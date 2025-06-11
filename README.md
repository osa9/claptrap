# ClapTrap Discord Bot

ClapTrap ちゃん（クラトラちゃん）は、Borderlands ファンのための Discord Bot です。
Claude 3.5 Sonnet と LangGraph を使用して構築されており、ClapTrap のキャラクターを忠実に再現します。

## 特徴

- **ClapTrap のペルソナ**: Borderlands シリーズの ClapTrap を模した 10 歳程度の女の子のキャラクター
- **会話記憶**: チャンネルごとの短期・長期記憶管理機能
- **Function Calling**: 以下のツールを自動的に使い分けます
  - **Web 検索**: Borderlands 関連情報や一般的な質問への回答
  - **画像生成**: DALL-E 3 を使用した画像生成
  - **YouTube 要約**: 動画の字幕から内容を要約

## セットアップ

### 1. リポジトリのクローン

```bash
git clone <repository_url>
cd claptrap
```

### 2. 依存関係のインストール

このプロジェクトは `uv` を使用します：

```bash
# uvのインストール (未インストールの場合)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 依存関係のインストール
uv sync
```

### 3. 環境変数の設定

`.env.example` を `.env` にコピーして、必要な API キーを設定してください：

```bash
cp .env.example .env
```

以下の値を `.env` ファイルに設定する必要があります：

```env
# Discord Bot Token (Discord Developer Portal から取得)
DISCORD_BOT_TOKEN=your_discord_bot_token

# Anthropic API Key (Claude 3.5 Sonnet 使用)
ANTHROPIC_API_KEY=your_anthropic_api_key

# OpenAI API Key (DALL-E 3 画像生成用)
OPENAI_API_KEY=your_openai_api_key

# Tavily API Key (Web検索用)
TAVILY_API_KEY=your_tavily_api_key
```

### 4. Discord Bot の作成

1. [Discord Developer Portal](https://discord.com/developers/applications) にアクセス
2. 新しいアプリケーションを作成
3. Bot セクションでボットを作成し、トークンを取得
4. OAuth2 > URL Generator で以下の権限を設定：
   - Bot Permissions: `Send Messages`, `Read Message History`, `Use Slash Commands`
   - Bot Scopes: `bot`
5. 生成された URL からボットをサーバーに招待

### 5. ボットの実行

```bash
uv run python -m claptrap.main
```

## 使用方法

Discord でボットをメンションすると、ClapTrap ちゃんが応答します：

```
@ClapTrap こんにちは！
```

### 自動機能

- **YouTube URL**: メッセージに YouTube URL が含まれていると自動で要約します
- **画像生成**: 「〜の画像を作って」などの依頼で画像を生成します
- **Web 検索**: Borderlands 関連の質問や一般的な情報を検索します

## 開発

### コード品質

```bash
# フォーマット
uv run ruff format .

# リント
uv run ruff check .

# 型チェック
uv run mypy .
```

### テスト実行

```bash
uv run pytest
```

### プロジェクト構造

```
claptrap/
├── claptrap/
│   ├── __init__.py
│   ├── main.py           # Discord ボットのエントリーポイント
│   ├── agent.py          # LangGraph エージェント
│   ├── memory.py         # 会話記憶管理
│   ├── prompts.py        # ClapTrap のペルソナ定義
│   └── tools/           # Function Calling ツール群
│       ├── __init__.py
│       ├── web_search.py
│       ├── image_generation.py
│       └── youtube_summary.py
├── tests/               # テストコード
├── docs/               # ドキュメント
├── pyproject.toml      # プロジェクト設定
└── .env.example        # 環境変数テンプレート
```

## ライセンス

MIT License

## Dicord への Bot インストール

https://discord.com/oauth2/authorize?client_id=1382038844085633144&permissions=1755906353659456&integration_type=0&scope=bot
