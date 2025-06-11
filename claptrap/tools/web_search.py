"""Web search tool using Tavily API."""

import os
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient


class WebSearchInput(BaseModel):
    """Web検索の入力スキーマ"""

    query: str = Field(description="検索クエリ")
    max_results: int = Field(default=5, description="最大取得件数")


class WebSearchTool(BaseTool):
    """Web検索ツール"""

    name: str = "web_search"
    description: str = (
        "インターネットを検索して情報を取得します。"
        "Borderlandsの世界に関する質問や、一般的な知識を調べるのに役立ちます。"
    )
    args_schema: type[BaseModel] = WebSearchInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client: TavilyClient | None = None

    @property
    def client(self) -> TavilyClient:
        """Tavilyクライアントを取得します。"""
        if self._client is None:
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise ValueError(
                    "TAVILY_API_KEYが設定されていません。"
                    ".envファイルに設定してください。"
                )
            self._client = TavilyClient(api_key=api_key)
        return self._client

    def _run(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        """
        Webを検索して結果を返します。

        Args:
            query: 検索クエリ
            max_results: 最大取得件数

        Returns:
             検索結果を整形した文字列
        """
        print(f"Web検索: {query}")
        try:
            # Tavily APIで検索
            response = self.client.search(
                query=query,
                search_depth="basic",
                max_results=max_results,
                include_answer=True,
                include_raw_content=False,
            )

            # 結果をフォーマット
            formatted_results = []

            # 回答があれば追加
            if response.get("answer"):
                formatted_results.append(f"▶ 要約: {response['answer']}")
                formatted_results.append("")

            # 検索結果
            if response.get("results"):
                formatted_results.append("▶ 検索結果:")
                for i, result in enumerate(response["results"][:max_results], 1):
                    title = result.get("title", "タイトルなし")
                    url = result.get("url", "")
                    content = result.get("content", "")

                    formatted_results.append(f"{i}. **{title}**")
                    if content:
                        # コンテンツのプレビュー
                        content_preview = (
                            content[:200] + "..." if len(content) > 200 else content
                        )
                        formatted_results.append(f"   {content_preview}")
                    if url:
                        formatted_results.append(f"   URL: {url}")
                    formatted_results.append("")

            if not formatted_results:
                return "検索結果が見つかりませんでした。"

            return "\n".join(formatted_results)

        except Exception as e:
            error_msg = f"エラーが発生しました: {str(e)}"
            print(f"Web検索エラー: {e}")  # ログ出力
            return error_msg

    async def _arun(self, query: str, max_results: int = 5, **kwargs: Any) -> str:
        """非同期でWebを検索します（現在は同期版を呼び出すだけです）。"""
        return self._run(query, max_results, **kwargs)


def create_web_search_tool() -> WebSearchTool:
    """Web検索ツールを作成するヘルパー関数"""
    return WebSearchTool()


# LangGraph等で直接呼び出すための関数
def web_search(query: str, max_results: int = 5) -> str:
    """
    Web検索を実行する関数

    Args:
        query: 検索クエリ
        max_results: 最大取得件数

    Returns:
         検索結果
    """
    tool = create_web_search_tool()
    return tool._run(query=query, max_results=max_results)
