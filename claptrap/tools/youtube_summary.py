"""YouTube動画の要約ツール。文字起こしAPIとLLMを利用します。"""

import os
import re
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi


class YouTubeSummaryInput(BaseModel):
    """YouTube動画要約ツールの入力スキーマ"""

    url: str = Field(description="対象のYouTube動画のURL")
    language: str = Field(default="ja", description="文字起こしの言語コード")


class YouTubeSummaryTool(BaseTool):
    """YouTube動画を要約するツール"""

    name: str = "summarize_youtube_video"
    description: str = (
        "YouTube動画のURLを受け取り、その内容を要約します。"
        "動画の概要を素早く把握したい場合に使用してください。"
    )
    args_schema: type[BaseModel] = YouTubeSummaryInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._llm: ChatAnthropic | None = None

    @property
    def llm(self) -> ChatAnthropic:
        """LLMインスタンスを初期化して返します。"""
        if self._llm is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEYが設定されていません。"
                    ".envファイルに記述するか、環境変数を設定してください。"
                )
            # Get model from environment variable, fallback to default
            model = os.getenv("CLAUDE_AGENT_MODEL", "claude-sonnet-4-20250514")

            self._llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=0.1,
            )
        return self._llm

    def _extract_video_id(self, url: str) -> str | None:
        """
        YouTubeのURLから動画IDを抽出します。

        Args:
            url: YouTubeのURL

        Returns:
            動画ID。見つからない場合はNoneを返します。
        """
        # 一般的なYouTube URLのパターン
        patterns = [
            r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)",
            r"youtube\.com/watch\?.*v=([^&\n?#]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        # URLが標準的でない場合のフォールバック
        try:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url)
            if "youtube.com" in parsed.netloc:
                query_params = parse_qs(parsed.query)
                if "v" in query_params:
                    return query_params["v"][0]
            elif "youtu.be" in parsed.netloc:
                return parsed.path.lstrip("/")
        except Exception:
            # URLパースエラーは無視（無効なURL）
            pass

        return None

    def _get_transcript(
        self, video_id: str, language: str = "ja"
    ) -> tuple[str | None, str | None]:
        """
        YouTube動画の文字起こしを取得します。

        Args:
            video_id: YouTubeの動画ID
            language: 文字起こしの言語

        Returns:
            文字起こしのテキストとエラーメッセージのタプル。
            成功時: (transcript_text, None)
            失敗時: (None, error_message)
        """
        try:
            print(f"動画 {video_id} の文字起こしを取得中...")

            # 新しいAPI (youtube-transcript-api 1.0+) を使用
            ytt_api = YouTubeTranscriptApi()

            # 利用可能な文字起こしリストを取得
            transcript_list = ytt_api.list(video_id)

            # 指定された言語の文字起こしを検索
            transcript = None
            available_languages = []

            try:
                available_transcripts = list(transcript_list)
                available_languages = [t.language_code for t in available_transcripts]
                print(f"利用可能な言語: {available_languages}")
            except Exception as list_error:
                error_msg = f"文字起こしリストの取得に失敗: {str(list_error)}"
                print(error_msg)
                return None, error_msg

            try:
                transcript = transcript_list.find_transcript([language])
                print(f"指定言語 '{language}' の文字起こしを取得")
            except Exception:
                print(f"指定言語 '{language}' が見つからない")
                # 指定言語が見つからない場合、英語を試す
                try:
                    transcript = transcript_list.find_transcript(["en"])
                    print("英語の文字起こしを使用")
                except Exception:
                    print("英語も見つからない")
                    # 英語も見つからない場合、利用可能な最初の文字起こしを使用
                    if available_transcripts:
                        transcript = available_transcripts[0]
                        lang_code = transcript.language_code
                        print(f"利用可能な最初の文字起こし ({lang_code}) を使用")

            if not transcript:
                error_msg = (
                    f"動画 {video_id} に利用可能な文字起こしが見つかりませんでした。"
                    f"利用可能な言語: {available_languages}"
                )
                print(error_msg)
                return None, error_msg

            # 文字起こしデータを取得
            try:
                print("文字起こしデータを取得中...")
                # 新しいAPIではfetch()がFetchedTranscriptを返す
                fetched_transcript = transcript.fetch()
                print(f"文字起こしデータを取得完了: {len(fetched_transcript)}エントリ")
            except Exception as fetch_error:
                fetch_error_msg = str(fetch_error)
                if "no element found" in fetch_error_msg.lower():
                    error_msg = (
                        f"動画 {video_id} の文字起こしデータが空または破損しています。"
                        f"詳細エラー: {fetch_error_msg}"
                    )
                else:
                    error_msg = f"文字起こしデータの取得エラー: {fetch_error_msg}"
                print(error_msg)
                return None, error_msg

            if not fetched_transcript:
                error_msg = f"動画 {video_id} の文字起こしデータが空でした"
                print(error_msg)
                return None, error_msg

            # テキスト形式にフォーマット
            # 新しいAPIでは FetchedTranscript オブジェクトからテキストを抽出
            try:
                print("文字起こしをテキスト形式にフォーマット中...")
                # FetchedTranscript はイテレーション可能
                formatted_transcript = " ".join(
                    snippet.text for snippet in fetched_transcript
                )
                print(f"フォーマット完了: {len(formatted_transcript)}文字")
                return formatted_transcript, None
            except Exception as format_error:
                print(f"文字起こしフォーマットエラー: {format_error}")
                # フォールバック: to_raw_data() を使用
                try:
                    print("to_raw_data() でフォールバック中...")
                    raw_data = fetched_transcript.to_raw_data()
                    manual_transcript = " ".join(
                        item.get("text", "") for item in raw_data
                    )
                    print(f"フォールバック完了: {len(manual_transcript)}文字")
                    return manual_transcript, None
                except Exception as manual_error:
                    error_msg = f"手動フォーマットも失敗: {str(manual_error)}"
                    print(error_msg)
                    return None, error_msg

        except Exception as e:
            error_msg = str(e)
            detailed_error = None

            if "no element found" in error_msg.lower():
                detailed_error = (
                    f"動画 {video_id} の文字起こしが利用できません（XMLパースエラー）。"
                    f"YouTubeのAPI仕様変更の可能性があります。詳細: {error_msg}"
                )
                print(detailed_error)
            elif "transcript" in error_msg.lower() and "disabled" in error_msg.lower():
                detailed_error = (
                    f"動画 {video_id} の文字起こしが無効になっています。"
                    f"投稿者が文字起こし機能を無効化している可能性があります。詳細: {error_msg}"
                )
                print(detailed_error)
            elif "unavailable" in error_msg.lower():
                detailed_error = (
                    f"動画 {video_id} は利用できません（非公開または削除済み）。"
                    f"詳細: {error_msg}"
                )
                print(detailed_error)
            elif "not found" in error_msg.lower():
                detailed_error = (
                    f"動画 {video_id} が見つかりません。"
                    f"URLが正しいか確認してください。詳細: {error_msg}"
                )
                print(detailed_error)
            else:
                detailed_error = f"文字起こしの取得に失敗しました: {error_msg}"
                print(detailed_error)

            return None, detailed_error

    def _summarize_transcript(self, transcript: str, video_url: str) -> str:
        """
        文字起こしをLLMで要約します。

        Args:
            transcript: 文字起こしのテキスト
            video_url: 動画のURL

        Returns:
            要約されたテキスト
        """
        try:
            # 長すぎる文字起こしを切り詰める
            max_transcript_length = 8000
            if len(transcript) > max_transcript_length:
                transcript = transcript[:max_transcript_length] + "..."

            prompt = f"""
あなたは、YouTubeの動画を要約するClapTrapというAIアシスタントです。
以下の文字起こしを要約してください。

動画URL: {video_url}

要約のルール:
- 動画の主題と最も重要な結論を最初に述べてください。
- 箇条書きを使って、3〜5個の主要なポイントをリストアップしてください。
- あなた、ClapTrapとして、一人称視点で、少しユーモラスに書いてください。
- Borderlands関連の話題があれば、特に注目してください。
- 全体で300字程度にまとめてください。

文字起こし:
{transcript}

要約:
"""

            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, "content") else str(response)

        except Exception as e:
            return f"要約の生成中にエラーが発生しました: {str(e)}"

    def _run(self, url: str, language: str = "ja", **kwargs: Any) -> str:
        """
        ツールのメイン処理。YouTube動画を要約します。

        Args:
            url: YouTube動画のURL
            language: 文字起こしの言語

        Returns:
            要約結果またはエラーメッセージ
        """
        try:
            # 動画IDを抽出
            video_id = self._extract_video_id(url)
            if not video_id:
                return (
                    "無効なYouTube URLです。動画IDが抽出できませんでした。"
                    "正しいYouTube URLを指定してください。"
                )

            # 文字起こしを取得
            transcript, error_msg = self._get_transcript(video_id, language)
            if not transcript:
                return (
                    "この動画の文字起こしを取得できませんでした。\n\n"
                    f"**詳細エラー情報:**\n{error_msg}\n\n"
                    "😭 このエラーは最近よく起こってるのだぁ〜\n"
                    "YouTubeがAPIを変更したせいで、ツールがうまく動かないのだぃ！\n\n"
                    "**考えられる原因：**\n"
                    "・ YouTubeのAPI仕様変更（最近多い）\n"
                    "・ 動画に文字起こしが存在しない\n"
                    "・ 非公開動画または制限された動画\n"
                    "・ 投稿者が文字起こしを無効にしている\n"
                    "・ ショート動画やライブ配信では利用できない\n"
                    "・ XMLパースエラー（YouTube側の問題）\n\n"
                    "**💡 回避策：**\n"
                    "・ 動画の内容を手動で要約してください\n"
                    "・ 他の動画で試してみてください\n"
                    "・ テキストベースのコンテンツを使ってください\n"
                    "・ しばらく時間をおいてから再試行してください"
                )

            # 要約を生成
            summary = self._summarize_transcript(transcript, url)

            return (
                f"**YouTube動画の要約が完了しました**\n\n{summary}\n\n参照元動画: {url}"
            )

        except Exception as e:
            import traceback

            error_details = traceback.format_exc()
            error_msg = (
                f"動画の要約中に予期せぬエラーが発生しました。\n\n"
                f"**エラーの詳細:**\n{str(e)}\n\n"
                f"**技術情報:**\n```\n{error_details}\n```\n\n"
                "😵 あれれ？何だか予想外のバグが起きちゃったのだ〜\n"
                "開発者さんに報告してもらえると助かるのだー！"
            )
            print(f"YouTube要約ツールでエラー: {e}")  # ログ出力
            print(f"スタックトレース: {error_details}")  # デバッグ用の詳細ログ
            return error_msg

    async def _arun(self, url: str, language: str = "ja", **kwargs: Any) -> str:
        """非同期でツールを実行します（現在は同期的実装のラッパーです）。"""
        return self._run(url, language, **kwargs)


def extract_youtube_urls(text: str) -> list[str]:
    """
    テキストからYouTubeのURLを抽出します。

    Args:
        text: 検索対象のテキスト

    Returns:
        見つかったYouTube URLのリスト
    """
    youtube_patterns = [
        r"https?://(?:www\.)?youtube\.com/watch\?[^\s]+",
        r"https?://youtu\.be/[^\s]+",
        r"https?://(?:www\.)?youtube\.com/embed/[^\s]+",
    ]

    urls = []
    for pattern in youtube_patterns:
        matches = re.findall(pattern, text)
        urls.extend(matches)

    return list(set(urls))  # 重複を削除


def create_youtube_summary_tool() -> YouTubeSummaryTool:
    """YouTube要約ツールのインスタンスを作成します。"""
    return YouTubeSummaryTool()


# LangGraph等で直接呼び出すための関数ラッパー
def summarize_youtube_video(url: str, language: str = "ja") -> str:
    """
    YouTube動画を要約します。

    Args:
        url: YouTube動画のURL
        language: 文字起こしの言語

    Returns:
        要約結果またはエラーメッセージ
    """
    tool = create_youtube_summary_tool()
    return tool._run(url=url, language=language)


def test_youtube_transcript(video_id: str) -> dict[str, Any]:
    """
    YouTube動画の文字起こしをテストするデバッグ関数

    Args:
        video_id: YouTube動画ID

    Returns:
        テスト結果の辞書
    """
    import traceback

    result: dict[str, Any] = {
        "video_id": video_id,
        "has_transcripts": False,
        "available_languages": [],
        "transcript_types": [],
        "error": None,
        "error_details": None,
        "transcript_sample": None,
        "total_entries": 0,
    }

    try:
        print(f"動画 {video_id} の文字起こし情報を調査中...")

        # 新しいAPI (youtube-transcript-api 1.0+) を使用
        ytt_api = YouTubeTranscriptApi()

        # 利用可能な文字起こしをリスト
        transcript_list = ytt_api.list(video_id)
        available_transcripts = list(transcript_list)

        if available_transcripts:
            result["has_transcripts"] = True
            result["available_languages"] = [
                t.language_code for t in available_transcripts
            ]
            result["transcript_types"] = [
                f"{t.language_code} ({'自動生成' if t.is_generated else '手動'})"
                for t in available_transcripts
            ]

            print(f"利用可能な文字起こし: {result['transcript_types']}")

            # 最初の文字起こしのサンプルを取得
            first_transcript = available_transcripts[0]
            try:
                print(f"サンプルデータを取得中: {first_transcript.language_code}")
                # 新しいAPIではfetch()がFetchedTranscriptを返す
                fetched_transcript = first_transcript.fetch()
                if fetched_transcript:
                    result["total_entries"] = len(fetched_transcript)
                    # FetchedTranscript からサンプルを取得
                    result["transcript_sample"] = [
                        {"text": s.text, "start": s.start, "duration": s.duration}
                        for s in list(fetched_transcript)[:3]
                    ]
                    print(f"取得成功: {len(fetched_transcript)}エントリ")
            except Exception as fetch_error:
                error_detail = traceback.format_exc()
                result["error"] = f"fetch_error: {str(fetch_error)}"
                result["error_details"] = error_detail
                print(f"フェッチエラー: {fetch_error}")
                print(f"詳細: {error_detail}")
        else:
            result["error"] = "利用可能な文字起こしが見つかりませんでした"
            print("文字起こしなし")

    except Exception as e:
        error_detail = traceback.format_exc()
        result["error"] = str(e)
        result["error_details"] = error_detail
        print(f"一般エラー: {e}")
        print(f"詳細: {error_detail}")

    return result
