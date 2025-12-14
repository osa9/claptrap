"""Tests for YouTube summary tool."""

from unittest.mock import Mock, patch

import pytest

from claptrap.tools.youtube_summary import (
    YouTubeSummaryTool,
    create_youtube_summary_tool,
    extract_youtube_urls,
    summarize_youtube_video,
)


class TestYouTubeSummaryTool:
    """YouTube要約ツールのテストクラス"""

    @pytest.fixture
    def mock_env_vars(self):
        """環境変数のモック"""
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "test_anthropic_key"},
        ):
            yield

    @pytest.fixture
    def youtube_tool(self, mock_env_vars):
        """テスト用のYouTube要約ツールインスタンス"""
        with patch("claptrap.tools.youtube_summary.ChatAnthropic"):
            return YouTubeSummaryTool()

    def test_tool_initialization(self, youtube_tool: YouTubeSummaryTool) -> None:
        """ツールの初期化テスト"""
        assert youtube_tool.name == "summarize_youtube_video"
        assert "YouTube動画のURL" in youtube_tool.description
        assert youtube_tool.args_schema is not None

    def test_extract_video_id_standard_url(
        self, youtube_tool: YouTubeSummaryTool
    ) -> None:
        """標準的なYouTube URLからの動画ID抽出テスト"""
        test_cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&feature=youtu.be",
                "dQw4w9WgXcQ",
            ),
        ]

        for url, expected_id in test_cases:
            result = youtube_tool._extract_video_id(url)
            assert result == expected_id, f"Failed for URL: {url}"

    def test_extract_video_id_invalid_url(
        self, youtube_tool: YouTubeSummaryTool
    ) -> None:
        """無効なURLからの動画ID抽出テスト"""
        invalid_urls = [
            "https://www.google.com",
            "not_a_url",
            "https://youtube.com/invalid",
            "",
        ]

        for url in invalid_urls:
            result = youtube_tool._extract_video_id(url)
            assert result is None, f"Should return None for invalid URL: {url}"

    @patch("claptrap.tools.youtube_summary.YouTubeTranscriptApi")
    def test_get_transcript_success(
        self, mock_api: Mock, youtube_tool: YouTubeSummaryTool
    ) -> None:
        """字幕取得成功のテスト"""
        # モックの設定
        mock_transcript_list = Mock()
        mock_transcript = Mock()
        mock_transcript.language_code = "ja"
        mock_transcript.fetch.return_value = [
            {"text": "Hello", "start": 0.0, "duration": 2.0},
            {"text": "World", "start": 2.0, "duration": 2.0},
        ]
        mock_transcript_list.find_transcript.return_value = mock_transcript

        # イテレータのモックを設定
        mock_transcript_list.__iter__ = Mock(return_value=iter([mock_transcript]))

        mock_api.list_transcripts.return_value = mock_transcript_list

        # TextFormatterのモック
        with patch(
            "claptrap.tools.youtube_summary.TextFormatter"
        ) as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter.format_transcript.return_value = "Hello World"
            mock_formatter_class.return_value = mock_formatter

            result = youtube_tool._get_transcript("test_video_id", "ja")

            assert result == ("Hello World", None)
            mock_api.list_transcripts.assert_called_once_with("test_video_id")
            mock_transcript_list.find_transcript.assert_called_once_with(["ja"])

    @patch("claptrap.tools.youtube_summary.YouTubeTranscriptApi")
    def test_get_transcript_fallback_to_english(
        self, mock_api: Mock, youtube_tool: YouTubeSummaryTool
    ) -> None:
        """日本語字幕がない場合の英語へのフォールバックテスト"""
        mock_transcript_list = Mock()
        mock_transcript = Mock()
        mock_transcript.language_code = "en"
        mock_transcript.fetch.return_value = [
            {"text": "Hello", "start": 0.0, "duration": 2.0}
        ]

        # イテレータのモック
        mock_transcript_list.__iter__ = Mock(return_value=iter([mock_transcript]))

        # 日本語字幕は見つからず、英語字幕が見つかる
        mock_transcript_list.find_transcript.side_effect = [
            Exception(),
            mock_transcript,
        ]

        mock_api.list_transcripts.return_value = mock_transcript_list

        with patch(
            "claptrap.tools.youtube_summary.TextFormatter"
        ) as mock_formatter_class:
            mock_formatter = Mock()
            mock_formatter.format_transcript.return_value = "Hello"
            mock_formatter_class.return_value = mock_formatter

            result = youtube_tool._get_transcript("test_video_id", "ja")

            assert result == ("Hello", None)
            assert mock_transcript_list.find_transcript.call_count == 2

    @patch("claptrap.tools.youtube_summary.YouTubeTranscriptApi")
    def test_get_transcript_no_subtitles(
        self, mock_api: Mock, youtube_tool: YouTubeSummaryTool
    ) -> None:
        """字幕が利用できない場合のテスト"""
        mock_api.list_transcripts.side_effect = Exception("No transcripts available")

        result, error = youtube_tool._get_transcript("test_video_id", "ja")

        assert result is None
        assert error is not None
        assert "No transcripts available" in error

    def test_summarize_transcript(self, youtube_tool: YouTubeSummaryTool) -> None:
        """字幕要約のテスト"""
        test_transcript = "これはテスト用の字幕です。" * 10
        test_url = "https://www.youtube.com/watch?v=test123"

        # LLMのモック
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = "これはテスト動画の要約です。"
        mock_llm.invoke.return_value = mock_response

        with patch.object(type(youtube_tool), "llm", new_callable=lambda: mock_llm):
            result = youtube_tool._summarize_transcript(test_transcript, test_url)

            assert result == "これはテスト動画の要約です。"
            mock_llm.invoke.assert_called_once()

    def test_summarize_transcript_error(self, youtube_tool: YouTubeSummaryTool) -> None:
        """字幕要約エラーのテスト"""
        test_transcript = "テスト字幕"
        test_url = "https://www.youtube.com/watch?v=test123"

        # LLMのモック（エラーを発生させる）
        mock_llm = Mock()
        mock_llm.invoke.side_effect = Exception("API Error")

        with patch.object(type(youtube_tool), "llm", new_callable=lambda: mock_llm):
            result = youtube_tool._summarize_transcript(test_transcript, test_url)

            assert "エラーが発生しました" in result

    def test_run_success(self, youtube_tool: YouTubeSummaryTool) -> None:
        """正常なツール実行のテスト"""
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with (
            patch.object(youtube_tool, "_extract_video_id", return_value="dQw4w9WgXcQ"),
            patch.object(
                youtube_tool, "_get_transcript", return_value=("Test transcript", None)
            ),
            patch.object(
                youtube_tool,
                "_summarize_transcript",
                return_value="Test summary",
            ),
        ):
            result = youtube_tool._run(test_url)

            assert "YouTube動画の要約が完了しました" in result
            assert "Test summary" in result
            assert test_url in result

    def test_run_invalid_url(self, youtube_tool: YouTubeSummaryTool) -> None:
        """無効なURL実行のテスト"""
        test_url = "https://www.google.com"

        with patch.object(youtube_tool, "_extract_video_id", return_value=None):
            result = youtube_tool._run(test_url)

            assert "無効なYouTube URL" in result

    def test_run_no_transcript(self, youtube_tool: YouTubeSummaryTool) -> None:
        """字幕なし動画のテスト"""
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with (
            patch.object(youtube_tool, "_extract_video_id", return_value="dQw4w9WgXcQ"),
            patch.object(
                youtube_tool,
                "_get_transcript",
                return_value=(None, "文字起こしを取得できませんでした"),
            ),
        ):
            result = youtube_tool._run(test_url)

            assert "文字起こしを取得できませんでした" in result

    def test_run_unexpected_error(self, youtube_tool: YouTubeSummaryTool) -> None:
        """予期せぬエラーのテスト"""
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        with patch.object(
            youtube_tool, "_extract_video_id", side_effect=Exception("Unexpected error")
        ):
            result = youtube_tool._run(test_url)

            assert "予期せぬエラーが発生しました" in result

    @pytest.mark.asyncio
    async def test_arun(self, youtube_tool: YouTubeSummaryTool) -> None:
        """非同期実行のテスト"""
        test_url = "https://www.youtube.com/watch?v=test123"

        with patch.object(youtube_tool, "_run", return_value="Test result"):
            result = await youtube_tool._arun(test_url)

            assert result == "Test result"


class TestExtractYouTubeUrls:
    """YouTube URL抽出機能のテストクラス"""

    def test_extract_single_url(self) -> None:
        """単一URL抽出のテスト"""
        text = "このYouTube動画見て！ https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        urls = extract_youtube_urls(text)

        assert len(urls) == 1
        assert "dQw4w9WgXcQ" in urls[0]

    def test_extract_multiple_urls(self) -> None:
        """複数URL抽出のテスト"""
        text = """
        チェックしてみて：
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        https://youtu.be/oHg5SJYRHA0
        """
        urls = extract_youtube_urls(text)

        assert len(urls) == 2
        assert any("dQw4w9WgXcQ" in url for url in urls)
        assert any("oHg5SJYRHA0" in url for url in urls)

    def test_extract_no_urls(self) -> None:
        """URLなしのテスト"""
        text = (
            "これは普通のテキストです。YouTubeという単語はありますが、URLはありません。"
        )
        urls = extract_youtube_urls(text)

        assert len(urls) == 0

    def test_extract_duplicate_urls(self) -> None:
        """重複URL除去のテスト"""
        text = """
        同じ動画を2回貼ります：
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        """
        urls = extract_youtube_urls(text)

        assert len(urls) == 1  # 重複は除去される

    def test_extract_different_formats(self) -> None:
        """異なるURL形式のテスト"""
        text = """
        様々な形式：
        https://www.youtube.com/watch?v=dQw4w9WgXcQ
        https://youtu.be/oHg5SJYRHA0
        https://youtube.com/embed/jNQXAC9IVRw
        """
        urls = extract_youtube_urls(text)

        assert len(urls) == 3


class TestFactoryFunctions:
    """ファクトリー関数のテストクラス"""

    @pytest.fixture
    def mock_env_vars(self):
        """環境変数のモック"""
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "test_anthropic_key"},
        ):
            yield

    def test_create_youtube_summary_tool(self, mock_env_vars) -> None:
        """create_youtube_summary_tool関数のテスト"""
        with patch("claptrap.tools.youtube_summary.ChatAnthropic"):
            tool = create_youtube_summary_tool()

            assert isinstance(tool, YouTubeSummaryTool)
            assert tool.name == "summarize_youtube_video"

    def test_summarize_youtube_video_function(self, mock_env_vars) -> None:
        """summarize_youtube_video関数のテスト"""
        test_url = "https://www.youtube.com/watch?v=test123"

        with (
            patch("claptrap.tools.youtube_summary.ChatAnthropic"),
            patch.object(
                YouTubeSummaryTool, "_run", return_value="Test summary"
            ) as mock_run,
        ):
            result = summarize_youtube_video(test_url)

            assert result == "Test summary"
            mock_run.assert_called_once_with(url=test_url, language="ja")


class TestIntegration:
    """統合テストクラス"""

    @pytest.fixture
    def mock_env_vars(self):
        """環境変数のモック"""
        with patch.dict(
            "os.environ",
            {"ANTHROPIC_API_KEY": "test_anthropic_key"},
        ):
            yield

    def test_full_workflow_mock(self, mock_env_vars) -> None:
        """完全なワークフローのモックテスト"""
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

        # 全ての依存関係をモック
        with (
            patch("claptrap.tools.youtube_summary.ChatAnthropic"),
            patch("claptrap.tools.youtube_summary.YouTubeTranscriptApi") as mock_api,
            patch(
                "claptrap.tools.youtube_summary.TextFormatter"
            ) as mock_formatter_class,
        ):
            # YouTubeTranscriptApiのモック設定
            mock_transcript_list = Mock()
            mock_transcript_list.__iter__ = Mock(
                return_value=iter([Mock(language_code="ja")])
            )

            mock_transcript = Mock()
            mock_transcript.fetch.return_value = [
                {"text": "Hello", "start": 0.0, "duration": 2.0},
                {"text": "World", "start": 2.0, "duration": 2.0},
            ]
            mock_transcript_list.find_transcript.return_value = mock_transcript
            mock_api.list_transcripts.return_value = mock_transcript_list

            # TextFormatterのモック設定
            mock_formatter = Mock()
            mock_formatter.format_transcript.return_value = "Hello World"
            mock_formatter_class.return_value = mock_formatter

            # ツールを作成して実行
            tool = create_youtube_summary_tool()

            # LLMのモック設定
            mock_llm = Mock()
            mock_response = Mock()
            mock_response.content = "これは英語の挨拶動画です。"
            mock_llm.invoke.return_value = mock_response

            with patch.object(type(tool), "llm", new_callable=lambda: mock_llm):
                result = tool._run(test_url)

                # 結果の検証
                assert "YouTube動画の要約が完了しました" in result
                assert "これは英語の挨拶動画です。" in result
                assert test_url in result

                # API呼び出しの検証
                mock_api.list_transcripts.assert_called_once_with("dQw4w9WgXcQ")
                mock_llm.invoke.assert_called_once()

    def test_error_handling_chain(self, mock_env_vars) -> None:
        """エラーハンドリングチェーンのテスト"""

        with patch("claptrap.tools.youtube_summary.ChatAnthropic"):
            tool = create_youtube_summary_tool()

            # 各段階でのエラーをテスト

            # 無効なURLのテスト
            invalid_url = "https://google.com/watch?v=test"
            result = tool._run(invalid_url)
            assert "無効なYouTube URL" in result

            # 字幕なしのテスト
            no_transcript_url = "https://www.youtube.com/watch?v=no_transcript"
            with patch.object(
                tool,
                "_get_transcript",
                return_value=(None, "文字起こしを取得できませんでした"),
            ):
                result = tool._run(no_transcript_url)
                assert "文字起こしを取得できませんでした" in result
