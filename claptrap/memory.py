"""Memory management system for ClapTrap bot."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from .prompts import MEMORY_SUMMARY_PROMPT


class ConversationMemory:
    """チャンネル毎の短期・長期記憶を管理するクラス"""

    def __init__(self, db_path: str = "./data/claptrap.db", max_short_term: int = 20):
        """
        初期化メソッド

        Args:
            db_path: SQLiteデータベースのパス
            max_short_term: 短期記憶に保存する最大メッセージ数
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_short_term = max_short_term
        self.short_term_memory: dict[str, list[BaseMessage]] = {}
        self._init_database()

    def _init_database(self) -> None:
        """データベースの初期化"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    channel_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    raw_conversation TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_channel_id 
                ON long_term_memory(channel_id)
            """)
            conn.commit()

    def add_message(self, channel_id: str, message: BaseMessage) -> None:
        """
        メッセージを短期記憶に追加

        Args:
            channel_id: Discordチャンネル ID
            message: 追加するメッセージ
        """
        if channel_id not in self.short_term_memory:
            self.short_term_memory[channel_id] = []

        self.short_term_memory[channel_id].append(message)

        # 最大数を超えた場合、古いものを削除
        if len(self.short_term_memory[channel_id]) > self.max_short_term:
            # 古いメッセージを長期記憶に移動を検討
            old_messages = self.short_term_memory[channel_id][: -self.max_short_term]
            self._consider_long_term_storage(channel_id, old_messages)

            # 短期記憶から削除
            self.short_term_memory[channel_id] = self.short_term_memory[channel_id][
                -self.max_short_term :
            ]

    def get_short_term_memory(self, channel_id: str) -> list[BaseMessage]:
        """
        指定チャンネルの短期記憶を取得

        Args:
            channel_id: Discordチャンネル ID

        Returns:
            短期記憶のメッセージリスト
        """
        return self.short_term_memory.get(channel_id, [])

    def _consider_long_term_storage(
        self, channel_id: str, messages: list[BaseMessage]
    ) -> None:
        """
        メッセージが長期記憶に値するかを判断し、適切に保存する

        Args:
            channel_id: Discordチャンネル ID
            messages: 判断対象のメッセージリスト
        """
        if not messages or len(messages) < 3:  # 短すぎる会話は保存しない
            return

        # メッセージを文字列形式に変換
        conversation_text = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                conversation_text += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                conversation_text += f"ClapTrap: {msg.content}\n"

        # 重要度を判断（簡単なヒューリスティック）
        if self._is_important_conversation(conversation_text):
            self._save_to_long_term_memory(channel_id, conversation_text, messages)

    def _is_important_conversation(self, conversation: str) -> bool:
        """
        会話が重要かどうかを判断する

        Args:
            conversation: 会話内容

        Returns:
            重要な会話の場合True
        """
        # 重要なキーワードをチェック
        important_keywords = [
            "約束",
            "予定",
            "一緒に",
            "プレイ",
            "攻略",
            "質問",
            "borderlands",
            "maya",
            "salvador",
            "axton",
            "zer0",
            "覚えて",
            "忘れないで",
            "重要",
            "メモ",
        ]

        conversation_lower = conversation.lower()
        for keyword in important_keywords:
            if keyword in conversation_lower:
                return True

        # 一定以上の長さがある会話も保存対象
        return len(conversation) > 500

    def _save_to_long_term_memory(
        self, channel_id: str, conversation: str, messages: list[BaseMessage]
    ) -> None:
        """
        長期記憶に保存する

        Args:
            channel_id: Discordチャンネル ID
            conversation: 会話内容
            messages: メッセージリスト
        """
        try:
            # LLMで要約を生成
            summary = self._generate_summary(conversation)

            # データベースに保存
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO long_term_memory 
                    (channel_id, summary, raw_conversation)
                    VALUES (?, ?, ?)
                """,
                    (
                        channel_id,
                        summary,
                        json.dumps(
                            [
                                {"type": type(msg).__name__, "content": msg.content}
                                for msg in messages
                            ]
                        ),
                    ),
                )
                conn.commit()
        except Exception as e:
            print(f"長期記憶保存エラー: {e}")

    def _generate_summary(self, conversation: str) -> str:
        """
        LLMを使って会話の要約を生成

        Args:
            conversation: 会話内容

        Returns:
            要約テキスト
        """
        try:
            # 環境変数からAPIキーを取得してLLMを初期化
            load_dotenv()
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return "要約生成失敗: APIキーが設定されていません"

            # Get model from environment variable, fallback to default
            model = os.getenv("CLAUDE_AGENT_MODEL", "claude-3-5-sonnet-20241022")

            llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=0.1,
            )

            prompt = MEMORY_SUMMARY_PROMPT.format(conversation=conversation)
            response = llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            return f"要約生成エラー: {str(e)}"

    def search_long_term_memory(
        self, channel_id: str, query: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """
        長期記憶から関連する情報を検索

        Args:
            channel_id: Discordチャンネル ID
            query: 検索クエリ
            limit: 取得する最大件数

        Returns:
            検索結果のリスト
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT summary, timestamp, raw_conversation
                    FROM long_term_memory
                    WHERE channel_id = ? 
                    AND (summary LIKE ? OR raw_conversation LIKE ?)
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (channel_id, f"%{query}%", f"%{query}%", limit),
                )

                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "summary": row[0],
                            "timestamp": row[1],
                            "raw_conversation": json.loads(row[2]) if row[2] else None,
                        }
                    )

                return results
        except Exception as e:
            print(f"長期記憶検索エラー: {e}")
            return []

    def get_context_for_channel(self, channel_id: str) -> str:
        """
        指定チャンネルの会話コンテキストを文字列で取得

        Args:
            channel_id: Discordチャンネル ID

        Returns:
            フォーマットされたコンテキスト文字列
        """
        context_parts = []

        # 短期記憶を追加
        short_term = self.get_short_term_memory(channel_id)
        if short_term:
            context_parts.append("=== 最近の会話 ===")
            for msg in short_term[-10:]:  # 最新10件
                if isinstance(msg, HumanMessage):
                    context_parts.append(f"User: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_parts.append(f"ClapTrap: {msg.content}")

        # 長期記憶から関連情報を検索（キーワードベース）
        recent_memories = self.search_long_term_memory(channel_id, "", 3)
        if recent_memories:
            context_parts.append("\n=== 以前の重要な会話 ===")
            for memory in recent_memories:
                context_parts.append(f"- {memory['summary']}")

        return "\n".join(context_parts)

    def clear_channel_memory(self, channel_id: str) -> None:
        """
        指定チャンネルのメモリをクリア（デバッグ用）

        Args:
            channel_id: Discordチャンネル ID
        """
        if channel_id in self.short_term_memory:
            del self.short_term_memory[channel_id]

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "DELETE FROM long_term_memory WHERE channel_id = ?", (channel_id,)
                )
                conn.commit()
        except sqlite3.OperationalError:
            # テーブルが存在しない場合は無視（テスト時の :memory: ファイル等）
            pass
