"""Tests for ClapTrap agent functionality."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from claptrap.agent import ClapTrapAgent, get_agent, process_user_message


class TestClapTrapAgent:
    """ClapTrapエージェントのテストクラス"""

    @pytest.fixture
    def mock_env_vars(self):
        """環境変数のモック"""
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_openai_key",
                "GROK_API_KEY": "test_grok_key",
                "VENICE_API_KEY": "test_venice_key",
                "GEMINI_API_KEY": "test_gemini_key",
            },
        ):
            yield

    @pytest.fixture
    def agent(self, mock_env_vars):
        """テスト用のエージェントインスタンス"""
        with patch("claptrap.agent.ChatAnthropic"), patch("claptrap.agent.ChatOpenAI"):
            return ClapTrapAgent(memory_db_path=":memory:")

    def test_agent_initialization(self, agent):
        """エージェントの初期化テスト"""
        assert agent is not None
        assert agent.memory is not None
        assert agent.llm is not None
        assert agent.tools is not None
        assert len(agent.tools) == 3  # web_search, image_generation, youtube_summary

    def test_get_agent_singleton(self, mock_env_vars):
        """グローバルエージェントのシングルトンテスト"""
        with patch("claptrap.agent.ChatAnthropic"), patch("claptrap.agent.ChatOpenAI"):
            agent1 = get_agent()
            agent2 = get_agent()
            assert agent1 is agent2

    @pytest.mark.anyio
    async def test_process_message_basic(self, agent):
        """基本的なメッセージ処理テスト"""
        from langchain_core.messages import AIMessage

        # LLMのレスポンスをモック
        mock_response = AIMessage(content="こんにちはだっぺ〜！")

        with patch.object(
            agent.graph, "ainvoke", return_value={"messages": [mock_response]}
        ):
            response = await agent.process_message(
                "こんにちは", "channel_123", "user_456"
            )

            assert "こんにちはだっぺ〜！" in response.text

    @pytest.mark.anyio
    async def test_process_message_error_handling(self, agent):
        """エラーハンドリングのテスト"""
        with patch.object(agent.graph, "ainvoke", side_effect=Exception("Test error")):
            response = await agent.process_message(
                "テストメッセージ", "channel_123", "user_456"
            )

            assert "バグったのだ" in response.text

    @pytest.mark.anyio
    async def test_process_user_message_function(self, mock_env_vars):
        """便利関数のテスト"""
        from claptrap.agent import AgentResponse

        mock_agent = Mock()
        mock_agent.process_message = AsyncMock(
            return_value=AgentResponse(
                text="テスト応答", channel_id="channel_123", user_id="user_456"
            )
        )

        with patch("claptrap.agent.get_agent", return_value=mock_agent):
            response = await process_user_message(
                "テストメッセージ", "channel_123", "user_456"
            )

            assert response.text == "テスト応答"
            mock_agent.process_message.assert_called_once_with(
                "テストメッセージ", "channel_123", "user_456"
            )


class TestMemoryIntegration:
    """メモリ統合テスト"""

    @pytest.fixture
    def mock_env_vars(self):
        """環境変数のモック"""
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "test_key",
                "OPENAI_API_KEY": "test_openai_key",
                "GROK_API_KEY": "test_grok_key",
            },
        ):
            yield

    def test_memory_clear(self, mock_env_vars):
        """メモリクリア機能のテスト"""
        with patch("claptrap.agent.ChatAnthropic"):
            agent = ClapTrapAgent(memory_db_path=":memory:")
            agent.clear_memory("test_channel")
            # エラーが発生しないことを確認


@pytest.mark.anyio
async def test_youtube_url_detection():
    """YouTube URL検出のテスト"""
    from claptrap.tools.youtube_summary import extract_youtube_urls

    test_text = "このYouTube動画見て！ https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    urls = extract_youtube_urls(test_text)

    assert len(urls) == 1
    assert "dQw4w9WgXcQ" in urls[0]
