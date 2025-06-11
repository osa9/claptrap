"""Discord bot main entry point for ClapTrap."""

import io
import logging
import os

import discord
from dotenv import load_dotenv

from .agent import AgentResponse, get_agent, process_user_message

# ログ設定
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 環境変数をロード
load_dotenv()


class ClapTrapBot(discord.Client):
    """ClapTrap Discord Bot クライアント"""

    def __init__(self, **kwargs):
        # Discordの必要なIntentsを有効化
        intents = discord.Intents.default()
        intents.message_content = True  # メッセージ内容を読み取るために必要
        intents.guilds = True
        intents.guild_messages = True

        super().__init__(intents=intents, **kwargs)

        # エージェントの初期化（遅延ロード）
        self._agent_initialized = False

    def _ensure_agent_initialized(self) -> None:
        """エージェントの遅延初期化"""
        if not self._agent_initialized:
            try:
                get_agent()  # エージェントを初期化
                self._agent_initialized = True
                logger.info("ClapTrapエージェントが初期化されました")
            except Exception as e:
                logger.error(f"エージェント初期化エラー: {e}")
                raise

    async def on_ready(self) -> None:
        """ボット起動時のイベント"""
        logger.info(f"ClapTrapちゃんがログインしました！ {self.user}")
        logger.info(f"ボットID: {self.user.id}")
        logger.info(f"参加中のサーバー数: {len(self.guilds)}")

        # エージェントを初期化
        try:
            self._ensure_agent_initialized()
        except Exception as e:
            logger.error(f"起動時エージェント初期化失敗: {e}")

    async def on_message(self, message: discord.Message) -> None:
        """メッセージ受信時のイベント"""
        # 自分のメッセージは無視
        if message.author == self.user:
            return

        # Botからのメッセージは無視
        if message.author.bot:
            return

        # メンション確認
        if self.user not in message.mentions:
            return

        try:
            # エージェントの初期化確認
            self._ensure_agent_initialized()

            # メンション部分を除去
            content = message.content
            for mention in message.mentions:
                content = content.replace(f"<@{mention.id}>", "").strip()
                content = content.replace(f"<@!{mention.id}>", "").strip()

            if not content:
                content = "こんにちは！"

            logger.info(
                f"メッセージ受信 - ユーザー: {message.author} "
                f"チャンネル: {message.channel} 内容: {content[:100]}..."
            )

            # タイピング表示開始
            async with message.channel.typing():
                # エージェントでメッセージ処理
                agent_response = await process_user_message(
                    content, str(message.channel.id), str(message.author.id)
                )

                # 応答データを処理
                response_text = agent_response.text
                print(f"応答チェック: {response_text[:200]}...")  # デバッグ用

                # 画像がある場合の処理
                if agent_response.has_image and agent_response.image_info:
                    image_result = await self._handle_image_from_agent_response(
                        agent_response
                    )
                    print(f"画像結果: {image_result}")  # デバッグ用

                    if image_result:
                        await message.reply(
                            response_text,
                            file=image_result["file"],
                            embed=image_result["embed"],
                        )
                        print("画像とembedを送信しました")
                    else:
                        await message.reply(response_text)
                        print("画像処理に失敗、テキストのみ送信")
                else:
                    await message.reply(response_text)
                    print("テキストのみの応答を送信しました")

                logger.info(f"応答送信完了 - 長さ: {len(response_text)}文字")

        except Exception as e:
            logger.error(f"メッセージ処理エラー: {e}", exc_info=True)

            # エラー応答を送信
            try:
                error_msg = (
                    "ごめんなさい！何だかエラーが起きちゃったっぺ〜。"
                    "もう一度試してみてほしいのだー！"
                )
                await message.reply(error_msg)
            except Exception as reply_error:
                logger.error(f"エラー応答送信失敗: {reply_error}")

    async def _handle_image_from_agent_response(
        self, agent_response: AgentResponse
    ) -> dict[str, discord.File | discord.Embed | None] | None:
        """
        AgentResponseから画像ファイルと埋め込みを作成

        Args:
            agent_response: エージェントからの構造化応答

        Returns:
            画像ファイルと埋め込み情報の辞書（画像がない場合はNone）
        """
        try:
            if not agent_response.has_image or not agent_response.image_info:
                return None

            image_info = agent_response.image_info
            logger.info(f"画像情報検出: プロンプト={image_info.prompt}")

            # 埋め込まれた画像データを取得
            b64_image = image_info.b64_data
            if b64_image:
                import base64

                logger.info("埋め込み画像データを取得")

                # base64データをバイトに変換
                image_bytes = base64.b64decode(b64_image)

                # Discordファイルを作成
                image_file = discord.File(
                    io.BytesIO(image_bytes),
                    filename="claptrap_generated.png",
                )

                # リッチ埋め込みを作成
                embed = discord.Embed(
                    title="🎨 ClapTrapちゃんが画像を生成したのだー！",
                    description=f"**プロンプト**: {image_info.prompt}",
                    color=0xFF6B35,  # オレンジっぽい色
                    timestamp=discord.utils.utcnow(),
                )
                embed.add_field(
                    name="画像設定",
                    value=f"サイズ: {image_info.size} | 品質: {image_info.quality}",
                    inline=False,
                )
                embed.set_image(url="attachment://claptrap_generated.png")
                embed.set_footer(
                    text="Powered by GPT-4 Vision ✨",
                )

                return {"file": image_file, "embed": embed}
            else:
                logger.warning("画像データが空です")

        except Exception as e:
            logger.warning(f"画像処理エラー: {e}")

        return None

    async def on_error(self, event: str, *args, **kwargs) -> None:
        """エラーハンドリング"""
        logger.error(f"Discord イベントエラー: {event}", exc_info=True)


def main() -> None:
    """メイン関数"""
    # 環境変数チェック
    bot_token = os.getenv("DISCORD_BOT_TOKEN")
    if not bot_token:
        raise ValueError(
            "DISCORD_BOT_TOKEN環境変数が設定されていません。"
            ".envファイルを確認してください。"
        )

    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        logger.warning("ANTHROPIC_API_KEY が設定されていません")

    # ログレベル設定
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))

    # ボット実行
    bot = ClapTrapBot()

    try:
        logger.info("ClapTrapボットを起動中...")
        bot.run(bot_token)
    except KeyboardInterrupt:
        logger.info("ボットを終了します...")
    except Exception as e:
        logger.error(f"ボット実行エラー: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
