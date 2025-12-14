"""LangGraph agent implementation for ClapTrap bot."""

import operator
import os
import traceback
from typing import Annotated, Any, Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel
from typing_extensions import TypedDict

from .memory import ConversationMemory
from .prompts import CLAPTRAP_SYSTEM_PROMPT, FUNCTION_CALLING_INSTRUCTIONS
from .tools.image_generation import create_image_generation_tool
from .tools.web_search import create_web_search_tool
from .tools.youtube_summary import create_youtube_summary_tool, extract_youtube_urls

# 環境変数をロード
load_dotenv()


class ImageInfo(BaseModel):
    """生成された画像の情報"""

    prompt: str
    size: str
    quality: str
    b64_data: str  # base64エンコードされた画像データ


class AgentResponse(BaseModel):
    """エージェントからの応答データ"""

    text: str
    has_image: bool = False
    image_info: ImageInfo | None = None
    channel_id: str
    user_id: str


class AgentState(TypedDict):
    """エージェントの状態を定義"""

    messages: Annotated[list[BaseMessage], operator.add]
    channel_id: str
    user_id: str
    next_action: Literal["respond", "use_tools", "end"]
    tool_calls_made: int  # ツール呼び出し回数をトラッキング
    generated_image: dict[str, str] | None  # 生成された画像データ


class ClapTrapAgent:
    """ClapTrapボットのメインエージェント"""

    def __init__(self, memory_db_path: str = "./data/claptrap.db"):
        """
        エージェントを初期化

        Args:
            memory_db_path: メモリデータベースのパス
        """
        self.memory = ConversationMemory(memory_db_path)
        # 2つのLLMを作成: チャット用とツール用
        self.llm_chat = self._create_llm_chat()
        self.llm_tools = self._create_llm_tools()
        self.tools = self._create_tools()
        self.graph = self._create_graph()

    def _create_venice_llm(self, model: str) -> ChatOpenAI:
        """Venice LLMを作成するヘルパー"""
        venice_key = os.getenv("VENICE_API_KEY")
        if not venice_key:
            raise ValueError("VENICE_API_KEY が設定されていません")
        return ChatOpenAI(
            api_key=venice_key,
            base_url="https://api.venice.ai/api/v1",
            model=model,
            temperature=0.9,
            max_tokens=1000,
        )

    def _create_llm_chat(self) -> ChatAnthropic | ChatOpenAI:
        """チャット用LLMを初期化（アンセンサード優先）"""
        if os.getenv("VENICE_API_KEY"):
            # Venice uncensored model (Function Calling非対応)
            return self._create_venice_llm(
                os.getenv("VENICE_MODEL_CHAT", "venice-uncensored")
            )
        elif model := os.getenv("CLAUDE_AGENT_MODEL"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                return ChatAnthropic(
                    api_key=api_key,
                    model=model,
                    temperature=0.9,
                    max_tokens=2000,
                )
            raise ValueError("ANTHROPIC_API_KEY が設定されていません")
        elif model := os.getenv("OPENAI_AGENT_MODEL"):
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                return ChatOpenAI(
                    api_key=api_key,
                    model=model,
                    temperature=0.9,
                    max_tokens=2000,
                )
        raise ValueError(
            "VENICE_API_KEY, CLAUDE_AGENT_MODEL, OPENAI_AGENT_MODEL のいずれかが必要です"
        )

    def _create_llm_tools(self) -> ChatAnthropic | ChatOpenAI:
        """ツール呼び出し用LLMを初期化（Function Calling対応）"""
        if os.getenv("VENICE_API_KEY"):
            # Venice with function calling support
            return self._create_venice_llm(
                os.getenv("VENICE_MODEL_TOOLS", "zai-org-glm-4.6")
            )
        # 他のプロバイダーはチャット用と同じ
        return self._create_llm_chat()

    def _create_tools(self) -> list[BaseTool]:
        """利用可能なツールを作成"""
        return [
            create_web_search_tool(),
            create_image_generation_tool(),
            create_youtube_summary_tool(),
        ]

    def _create_graph(self) -> Any:
        """LangGraphを構築"""
        # ツール用LLMにツールをバインド（Function Calling対応モデル）
        llm_with_tools = self.llm_tools.bind_tools(self.tools)

        # チャット用LLM（アンセンサード、ツールなし）
        llm_chat = self.llm_chat

        # ツールノードを作成
        tool_node = ToolNode(self.tools)

        def tool_wrapper(state: AgentState) -> dict[str, Any]:
            """ツール実行回数をカウントし、画像データをキャプチャするラッパー"""
            from .tools.image_generation import (
                clear_last_generated_image,
                get_last_generated_image,
            )

            result = tool_node.invoke(state)
            print(result)

            # ツール呼び出し回数を増加
            result["tool_calls_made"] = state.get("tool_calls_made", 0) + 1

            # 画像生成ツールが使われた場合、画像データを状態に保存
            for message in result.get("messages", []):
                if (
                    isinstance(message, ToolMessage)
                    and message.name == "generate_image"
                ):
                    image_data = get_last_generated_image()
                    if image_data:
                        result["generated_image"] = image_data
                        clear_last_generated_image()
                        print(f"画像データを状態に保存: {image_data['prompt']}")
                    break

            print(f"ツール実行完了。回数: {result['tool_calls_made']}")
            return result

        def should_continue(state: AgentState) -> Literal["use_tools", "end"]:
            """ツールを使うかどうかを判定"""
            messages = state["messages"]
            last_message = messages[-1]
            tool_calls_made = state.get("tool_calls_made", 0)

            # デバッグ出力
            print(
                f"should_continue: メッセージ数={len(messages)}, "
                f"最後のメッセージタイプ={type(last_message).__name__}, "
                f"ツール呼び出し回数={tool_calls_made}"
            )

            # 既に2回ツールを呼び出している場合は終了
            if tool_calls_made >= 3:
                print("ツール呼び出し上限に達したため終了")
                return "end"

            # ツール呼び出しがある場合
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                print(f"ツール呼び出し検出: {len(last_message.tool_calls)}個")
                return "use_tools"

            print("会話終了")
            return "end"

        def call_model(state: AgentState) -> dict[str, Any]:
            """LLMを呼び出し"""
            channel_id = state["channel_id"]
            messages = state["messages"]
            tool_calls_made = state.get("tool_calls_made", 0)
            print("=======")
            print(messages)

            # ツール実行済みかどうかを判定
            has_tool_results = any(isinstance(m, ToolMessage) for m in messages)

            # メモリからコンテキストを取得
            memory_context = self.memory.get_context_for_channel(channel_id)

            # システムプロンプトを構築
            system_prompt = CLAPTRAP_SYSTEM_PROMPT
            if memory_context:
                system_prompt += f"\n\n=== 会話履歴 ===\n{memory_context}"

            # ツールが必要な場合のみツール関連の指示を追加
            if not has_tool_results:
                system_prompt += f"\n\n{FUNCTION_CALLING_INSTRUCTIONS}"

                # YouTube URLの検出をシステムプロンプトに含める
                last_user_message = None
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        last_user_message = msg
                        break

                if last_user_message:
                    youtube_urls = extract_youtube_urls(last_user_message.content)
                    if youtube_urls:
                        system_prompt += (
                            f"\n\n重要: ユーザーのメッセージにYouTube URL "
                            f"({youtube_urls[0]}) が含まれています。"
                            "summarize_youtube_videoツールを使って動画を要約してください。"
                        )

            # ツール使用後の応答を保証する指示を追加
            if has_tool_results:
                system_prompt += (
                    "\n\n**重要**: ツールの結果を元に、ユーザーに対して"
                    "ClapTrapの性格で応答してください。"
                    "絶対に空の応答や無言は避けてください。"
                )

            # メッセージを構築（LangChainメッセージ形式を維持）
            formatted_messages = []

            # ツール結果がある場合（アンセンサードモデル用）
            # -> ToolMessageはサポートされないので、ユーザーメッセージとして変換
            if has_tool_results:
                # 最初のユーザーメッセージを取得
                user_content = ""
                tool_results = []
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        user_content = msg.content
                    elif isinstance(msg, ToolMessage):
                        tool_results.append(msg.content)

                # ツール結果をシステムプロンプトに追加済みなので、
                # シンプルにユーザーメッセージ + ツール結果を含める
                combined_content = user_content
                if tool_results:
                    combined_content += "\n\n=== ツール実行結果 ===\n"
                    combined_content += "\n".join(tool_results)

                formatted_messages = [HumanMessage(content=combined_content)]
            else:
                # 通常のメッセージ処理（ツール対応モデル用）
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        formatted_messages.append(HumanMessage(content=msg.content))
                    elif isinstance(msg, AIMessage):
                        # AIMessageのtool_callsも保持
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            formatted_messages.append(
                                AIMessage(
                                    content=msg.content, tool_calls=msg.tool_calls
                                )
                            )
                        else:
                            formatted_messages.append(AIMessage(content=msg.content))
                    elif isinstance(msg, ToolMessage):
                        # ToolMessageのtool_call_idを保持
                        if hasattr(msg, "tool_call_id"):
                            formatted_messages.append(
                                ToolMessage(
                                    content=msg.content, tool_call_id=msg.tool_call_id
                                )
                            )

            # 空のメッセージリストの場合はダミーメッセージを追加
            if not formatted_messages:
                formatted_messages.append(HumanMessage(content="やあ"))

            # 使用するLLMを選択
            # - ツール結果がある場合: llm_chat (アンセンサード) で応答生成
            # - ツール結果がない場合: llm_with_tools (FC対応) でツール判定
            if has_tool_results:
                # アンセンサードモデルで応答生成
                print(
                    f"LLM選択: チャット用モデル (ツール実行済み: {tool_calls_made}回)"
                )
                messages_with_system = [
                    SystemMessage(content=system_prompt)
                ] + formatted_messages
                response = llm_chat.invoke(messages_with_system)
            else:
                # ツール対応モデルでツール判定
                print("LLM選択: ツール用モデル (初回呼び出し)")
                if isinstance(self.llm_tools, ChatOpenAI):
                    messages_with_system = [
                        SystemMessage(content=system_prompt)
                    ] + formatted_messages
                    response = llm_with_tools.invoke(messages_with_system)
                else:
                    # Anthropicの場合はsystemパラメータを使用
                    llm_with_system = llm_with_tools.bind(system=system_prompt)
                    response = llm_with_system.invoke(formatted_messages)

            # レスポンスの検証
            if not hasattr(response, "content") or not response.content:
                print(f"警告: LLMが空の応答を返しました: {response}")

            return {"messages": [response]}

        # グラフを構築
        workflow = StateGraph(AgentState)

        # ノードを追加
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", tool_wrapper)

        # エントリーポイントを設定
        workflow.set_entry_point("agent")

        # 条件分岐を追加
        workflow.add_conditional_edges(
            "agent", should_continue, {"use_tools": "tools", "end": END}
        )

        # ツール実行後はエージェントに戻る
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    async def process_message(
        self, message: str, channel_id: str, user_id: str
    ) -> AgentResponse:
        """
        メッセージを処理して応答を生成

        Args:
            message: ユーザーメッセージ
            channel_id: Discordチャンネル ID
            user_id: ユーザー ID

        Returns:
            AgentResponse: 構造化された応答データ
        """
        # ユーザーメッセージをメモリに追加
        user_message = HumanMessage(content=message)
        self.memory.add_message(channel_id, user_message)

        # エージェント実行
        initial_state: AgentState = {
            "messages": [user_message],
            "channel_id": channel_id,
            "user_id": user_id,
            "next_action": "respond",
            "tool_calls_made": 0,
            "generated_image": None,
        }

        try:
            # グラフを実行（無限ループを防ぐため、適切な実行回数上限を設定）
            final_state = await self.graph.ainvoke(
                initial_state, config={"recursion_limit": 10}
            )

            # 最後のAIメッセージを取得
            ai_response = None

            for msg in reversed(final_state["messages"]):
                if isinstance(msg, AIMessage):
                    # 空文字列も有効な応答として扱う
                    ai_response = msg.content if msg.content is not None else ""
                    break

            # ai_responseがNoneの場合のみデフォルトメッセージを使用
            if ai_response is None:
                # ログ出力で問題調査を支援
                print(f"警告: AI応答が見つかりません。final_state: {final_state}")
                ai_response = (
                    "うーん、何だか調子が悪いみたいだよぅ…"
                    "もう一度話しかけてみてほしいのだー！"
                )
            elif ai_response == "":
                # 空文字列の場合は、ツールエラーを示すメッセージに変換
                print("警告: AI応答が空文字列です。ツールエラーの可能性があります。")
                ai_response = (
                    "あれれ？何だかツールがうまく動かなかったみたいなのだ〜💦\n"
                    "もう一度試してみるか、別の方法で聞いてみてほしいのだー！"
                )

            # 応答をメモリに追加
            ai_message = AIMessage(content=ai_response)
            self.memory.add_message(channel_id, ai_message)

            # 状態から画像情報を取得
            image_info = self._get_image_info_from_state(final_state)

            return AgentResponse(
                text=ai_response,
                has_image=image_info is not None,
                image_info=image_info,
                channel_id=channel_id,
                user_id=user_id,
            )

        except Exception as e:
            error_response = (
                "にゃにゃっ！？クラトラちゃんの回路がちょっとバグったのだ〜〜💦 "
            )
            print(f"エージェント実行エラー: {e}")
            print(
                f"メッセージ: '{message}', チャンネル: {channel_id}, "
                f"ユーザー: {user_id}"
            )
            traceback.print_exc()

            # エラー応答もメモリに追加
            try:
                error_message = AIMessage(content=error_response)
                self.memory.add_message(channel_id, error_message)
            except Exception as memory_error:
                print(f"メモリ追加エラー: {memory_error}")

            return AgentResponse(
                text=error_response,
                has_image=False,
                image_info=None,
                channel_id=channel_id,
                user_id=user_id,
            )

    def _get_image_info_from_state(self, state: AgentState) -> ImageInfo | None:
        """AgentStateから画像情報を取得します。"""
        image_data = state.get("generated_image")
        if image_data:
            return ImageInfo(
                prompt=image_data["prompt"],
                size=image_data["size"],
                quality=image_data["quality"],
                b64_data=image_data["b64_data"],
            )
        return None

    def clear_memory(self, channel_id: str) -> None:
        """指定チャンネルのメモリをクリア（デバッグ用）"""
        self.memory.clear_channel_memory(channel_id)


# グローバルエージェントインスタンス
_agent_instance: ClapTrapAgent | None = None


def get_agent() -> ClapTrapAgent:
    """グローバルエージェントインスタンスを取得"""
    global _agent_instance  # noqa: PLW0603
    if _agent_instance is None:
        _agent_instance = ClapTrapAgent()
    return _agent_instance


async def process_user_message(
    message: str, channel_id: str, user_id: str
) -> AgentResponse:
    """
    ユーザーメッセージを処理する便利関数

    Args:
        message: ユーザーメッセージ
        channel_id: Discordチャンネル ID
        user_id: ユーザー ID

    Returns:
        AgentResponse: 構造化された応答データ
    """
    agent = get_agent()
    return await agent.process_message(message, channel_id, user_id)
