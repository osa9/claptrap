"""Web search tool using xAI SDK."""

import os
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from xai_sdk import Client
from xai_sdk.chat import user
from xai_sdk.tools import x_search


class WebSearchInput(BaseModel):
    """Web検索の入力スキーマ"""

    query: str = Field(description="検索クエリ")


class WebSearchTool(BaseTool):
    """Web検索ツール (Grok with x_search)"""

    name: str = "web_search"
    description: str = (
        "インターネット検索を行い、最新の情報やTwitter上の話題を取得します。"
        "Grokを使用してTwitter(X)の検索結果を要約して返します。"
    )
    args_schema: type[BaseModel] = WebSearchInput

    def _run(self, query: str, **kwargs: Any) -> str:
        """
        Webを検索して結果を返します。

        Args:
            query: 検索クエリ

        Returns:
             検索結果の要約
        """
        api_key = os.getenv("GROK_API_KEY")
        if not api_key:
            return "エラー: GROK_API_KEYが設定されていません。.envを確認してください。"

        model_name = os.getenv("GROK_MODEL", "grok-4-1-fast-non-reasoning")
        print(f"Grok検索実行: {query} (Model: {model_name})")

        try:
            client = Client(api_key=api_key)

            prompt = (
                f"以下のクエリについてTwitter(X)とWebで検索し、"
                f"最新の情報を日本語で詳しく要約してください:\n\n"
                f"クエリ: {query}"
            )

            messages = [user(prompt)]

            chat = client.chat.create(
                model=model_name,
                messages=messages,
                tools=[x_search()],
            )

            response = chat.sample()
            print(response)

            # 応答からコンテンツを取得
            if response.content:
                return response.content
            return "検索結果が空でした。"

        except Exception as e:
            error_msg = f"検索中にエラーが発生しました: {str(e)}"
            print(error_msg)
            return error_msg

    async def _arun(self, query: str, **kwargs: Any) -> str:
        """非同期でWebを検索します。"""
        return self._run(query, **kwargs)


def create_web_search_tool() -> WebSearchTool:
    """Web検索ツールを作成するヘルパー関数"""
    return WebSearchTool()


# LangGraph等で直接呼び出すための関数
def web_search(query: str) -> str:
    """
    Web検索を実行する関数

    Args:
        query: 検索クエリ

    Returns:
         検索結果
    """
    tool = create_web_search_tool()
    return tool._run(query=query)
