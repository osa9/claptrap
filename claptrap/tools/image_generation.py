"""Image generation tool using OpenAI (DALL-E 3) and Google Gemini (Nano Banana Pro)."""

import base64
import os
from typing import Any, Literal

from langchain_core.tools import BaseTool
from openai import OpenAI
from pydantic import BaseModel, Field

# 最後に生成された画像データ（一時的保存）
_last_generated_image: dict[str, str] | None = None


class ImageGenerationInput(BaseModel):
    """画像生成ツールの入力スキーマ"""

    prompt: str = Field(description="画像生成のプロンプト")
    style: Literal["illustration", "realistic"] = Field(
        default="illustration",
        description="画像のスタイル (illustration: ChatGPT/DALL-E 3, realistic: Nano Banana Pro/Gemini)",
    )
    size: str = Field(
        default="1024x1024",
        description="画像のサイズ (1024x1024 など)",
    )
    quality: str = Field(default="standard", description="画像の品質 (standard, hd)")


class ImageGenerationTool(BaseTool):
    """画像生成ツール (DALL-E 3 / Nano Banana Pro)"""

    name: str = "generate_image"
    description: str = (
        "指定されたプロンプトから画像を生成します。"
        "スタイルとして 'illustration' (イラスト) または 'realistic' (写実的/スライド) を選択できます。"
    )
    args_schema: type[BaseModel] = ImageGenerationInput

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._openai_client: OpenAI | None = None

    @property
    def openai_client(self) -> OpenAI:
        """OpenAIクライアントを取得します。"""
        if self._openai_client is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEYが設定されていません。")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def _run(
        self,
        prompt: str,
        style: str = "illustration",
        size: str = "1024x1024",
        quality: str = "standard",
        **kwargs: Any,
    ) -> dict[str, Any] | str:
        """画像を生成します。"""
        print(f"画像生成リクエスト: {prompt} (Style: {style})")

        try:
            enhanced_prompt = self._enhance_prompt(prompt, style)
            image_b64 = None

            if style == "illustration":
                # OpenAI DALL-E 3
                print("Using OpenAI DALL-E 3...")
                response = self.openai_client.images.generate(
                    model=os.getenv("IMAGE_MODEL_ILLUSTRATION", "dall-e-3"),
                    prompt=enhanced_prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    response_format="b64_json",
                )
                if response.data:
                    image_b64 = response.data[0].b64_json

            elif style == "realistic":
                # Google GenAI (Nano Banana Pro)
                print("Using Google Nano Banana Pro (Gemini)...")
                import google.generativeai as genai

                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    return "エラー: GEMINI_API_KEYが設定されていません。"

                genai.configure(api_key=api_key)
                model_name = os.getenv(
                    "IMAGE_MODEL_REALISTIC", "nano-banana-pro-preview"
                )

                # Geminiモデルを使って画像を生成
                # 注: 通常のGeminiモデルは画像を返さない可能性がありますが、
                # Nano Banana Proが画像生成能力を持つと仮定してgenerate_contentを使用します
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(enhanced_prompt)

                # レスポンスから画像データを抽出
                # レスポンスから画像データを抽出
                if response.parts:
                    for part in response.parts:
                        # inline_dataがあるか確認
                        if hasattr(part, "inline_data") and part.inline_data:
                            if part.inline_data.mime_type.startswith("image/"):
                                # 画像データ(bytes)をbase64に変換
                                image_b64 = base64.b64encode(
                                    part.inline_data.data
                                ).decode("utf-8")
                                break

                if not image_b64:
                    return f"Nano Banana Proからの画像生成に失敗しました (Model: {model_name})。レスポンスに画像が含まれていません。"

            else:
                return f"不明なスタイルです: {style}"

            if not image_b64:
                return "画像データの生成に失敗しました。"

            # 最後の画像データを保存
            global _last_generated_image
            _last_generated_image = {
                "prompt": prompt,
                "size": size,
                "quality": quality,
                "b64_data": image_b64,
            }

            return f"画像を生成しました！ ({style})\nプロンプト: {prompt}"

        except Exception as e:
            error_msg = f"画像生成エラー: {str(e)}"
            print(error_msg)
            return error_msg

    def _enhance_prompt(self, prompt: str, style: str) -> str:
        """スタイルに応じてプロンプトを強化します。"""
        if style == "illustration":
            return f"{prompt}, high quality illustration, beautiful, detailed"
        elif style == "realistic":
            return f"{prompt}, photorealistic, high resolution, 4k, professional photography"
        return prompt

    async def _arun(
        self, prompt: str, style: str = "illustration", **kwargs: Any
    ) -> str:
        """非同期ラッパー (同期実行)"""
        return self._run(prompt, style, **kwargs)


def get_last_generated_image() -> dict[str, str] | None:
    return _last_generated_image


def clear_last_generated_image() -> None:
    global _last_generated_image
    _last_generated_image = None


def create_image_generation_tool() -> ImageGenerationTool:
    return ImageGenerationTool()


def generate_image(prompt: str, style: str = "illustration") -> Any:
    tool = create_image_generation_tool()
    return tool._run(prompt=prompt, style=style)
