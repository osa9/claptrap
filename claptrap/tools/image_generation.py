"""Image generation tool using OpenAI GPT-4 Vision (gpt-image-1)."""

import os
from typing import Any

import aiohttp
from langchain_core.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

# 最後に生成された画像データ（一時的保存）
_last_generated_image: dict[str, str] | None = None


class ImageGenerationInput(BaseModel):
    """画像生成ツールの入力スキーマ"""

    prompt: str = Field(description="画像生成のプロンプト")
    size: str = Field(
        default="1024x1024",
        description="画像のサイズ (1024x1024, 1792x1024, 1024x1792)",
    )
    quality: str = Field(
        default="high", description="画像の品質 (low, medium, high, auto)"
    )


class ImageGenerationTool(BaseTool):
    """GPT-4 Vision (gpt-image-1) を使用した画像生成ツール"""

    name: str = "generate_image"
    description: str = (
        "GPT-4 Vision (gpt-image-1) を使用して、指定されたプロンプトから"
        "画像を生成します。ClapTrapの世界観に合わせたスタイル調整も自動的に行います。"
    )
    args_schema: type[BaseModel] = ImageGenerationInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        """OpenAIクライアントを初期化または返します。"""
        if self._client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEYが環境変数に設定されていません。"
                    ".envファイルにキーを追加してください。"
                )
            self._client = OpenAI(api_key=api_key)
        return self._client

    def _run(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "high",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        画像を生成します。

        Args:
            prompt: 画像生成のプロンプト
            size: 画像のサイズ
            quality: 画像の品質

        Returns:
            生成結果を含む辞書
        """
        try:
            # プロンプトを強化してスタイルを追加
            enhanced_prompt = self._enhance_prompt(prompt)

            # GPT-4 Vision で画像を生成 (gpt-image-1 model)
            response = self.client.images.generate(
                model="gpt-image-1",
                prompt=enhanced_prompt,
                size=size,
                quality=quality,
                n=1,
                # response_format="b64_json"  # Not supported by gpt-image-1
            )

            if not response.data:
                return "画像データの受信に失敗しました。"

            image_data = response.data[0]

            # 最後の画像データを保存（キャッシュなし）
            global _last_generated_image
            _last_generated_image = {
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "b64_data": image_data.b64_json,
            }

            # LLM用のクリーンな応答（画像データは一切含まない）
            return (
                f"やったのだー！画像を生成したよ〜✨\n"
                f"プロンプト: {prompt}\n"
                f"サイズ: {size} | 品質: {quality}"
            )

        except Exception as e:
            error_msg = f"画像生成中にエラーが発生しました: {str(e)}"
            print(f"画像生成エラー: {e}")  # デバッグ用
            return error_msg

    def _enhance_prompt(self, prompt: str) -> str:
        """
        プロンプトを強化し、Borderlandsのスタイルを追加します。

        Args:
            prompt: 元のプロンプト

        Returns:
            強化されたプロンプト
        """
        # Borderlands関連のキーワード
        borderlands_keywords = [
            "borderlands",
            "claptrap",
            "pandora",
            "maya",
            "salvador",
            "axton",
            "zer0",
            "gaige",
            "krieg",
            "vault hunter",
        ]

        prompt_lower = prompt.lower()
        is_borderlands = any(
            keyword in prompt_lower for keyword in borderlands_keywords
        )

        if is_borderlands:
            # Borderlandsスタイルを適用
            enhanced = (
                f"{prompt}, borderlands video game art style, cel-shaded, "
                "comic book style, vibrant colors, post-apocalyptic"
            )
        else:
            # 一般的な高品質スタイル
            enhanced = f"{prompt}, high quality, detailed, beautiful"

        # 不適切なコンテンツをフィルタリング
        enhanced = self._filter_inappropriate_content(enhanced)

        return enhanced

    def _filter_inappropriate_content(self, prompt: str) -> str:
        """
        プロンプトから不適切なコンテンツをフィルタリングします。

        Args:
            prompt: フィルタリング前のプロンプト

        Returns:
            フィルタリング後のプロンプト
        """
        # 不適切な単語のリスト
        inappropriate_words = [
            "violence",
            "blood",
            "gore",
            "weapon",
            "gun",
            "knife",
            "violent",
            "explicit",
            "nsfw",
        ]

        filtered_prompt = prompt
        for word in inappropriate_words:
            if word in filtered_prompt.lower():
                # 単語をより安全なものに置換
                if word in ["weapon", "gun", "knife"]:
                    filtered_prompt = filtered_prompt.replace(word, "tool")
                elif word in ["violence", "violent"]:
                    filtered_prompt = filtered_prompt.replace(word, "action")
                else:
                    filtered_prompt = filtered_prompt.replace(word, "")

        return filtered_prompt

    async def _arun(
        self,
        prompt: str,
        size: str = "1024x1024",
        quality: str = "high",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """非同期で画像生成を実行します。"""
        return self._run(prompt, size, quality, **kwargs)


async def download_image_as_bytes(url: str) -> bytes | None:
    """
    URLから画像をバイトデータとしてダウンロードします。

    Args:
        url: 画像のURL

    Returns:
        画像のバイトデータ、またはダウンロード失敗時はNone
    """
    try:
        async with aiohttp.ClientSession() as session, session.get(url) as response:
            if response.status == 200:
                return await response.read()
            return None
    except Exception as e:
        print(f"画像ダウンロードエラー: {e}")
        return None




def get_last_generated_image() -> dict[str, str] | None:
    """最後に生成された画像データを取得します。"""
    return _last_generated_image


def clear_last_generated_image() -> None:
    """最後に生成された画像データをクリアします。"""
    global _last_generated_image
    _last_generated_image = None


def create_image_generation_tool() -> ImageGenerationTool:
    """画像生成ツールを初期化します。"""
    return ImageGenerationTool()


# LangGraphから直接呼び出すためのラッパー関数
def generate_image(
    prompt: str, size: str = "1024x1024", quality: str = "high"
) -> dict[str, Any]:
    """
    画像を生成するラッパー関数

    Args:
        prompt: 画像生成のプロンプト
        size: 画像のサイズ
        quality: 画像の品質

    Returns:
        生成結果の辞書
    """
    tool = create_image_generation_tool()
    return tool._run(prompt=prompt, size=size, quality=quality)
