"""Video-language model utilities."""

from __future__ import annotations

from http import HTTPStatus
import os
from pathlib import Path
from typing import Any


class QwenVideoJudge:
    """DashScope Qwen yes/no judge for local video files."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "qwen3-vl-plus",
        fps: int = 8,
        base_url: str | None = None,
    ) -> None:
        """Initializes the judge.

        Args:
            api_key: DashScope API key. Falls back to DASHSCOPE_API_KEY.
            model: DashScope VLM model name.
            fps: Video sampling FPS for DashScope.
            base_url: Optional DashScope base URL.

        Raises:
            ValueError: If no API key is provided.
        """
        import dashscope
        from dashscope import MultiModalConversation

        if base_url:
            dashscope.base_http_api_url = base_url
        self._client = MultiModalConversation
        self._api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not self._api_key:
            raise ValueError("Missing DASHSCOPE_API_KEY or api_key.")
        self._model = model
        self._fps = fps

    @staticmethod
    def _response_value(response: Any, key: str) -> Any:
        if isinstance(response, dict):
            return response.get(key)
        return getattr(response, key, None)

    def _raise_for_bad_response(self, response: Any) -> None:
        status_code = self._response_value(response, "status_code")
        if status_code in (None, HTTPStatus.OK, int(HTTPStatus.OK)):
            return
        code = self._response_value(response, "code") or "unknown"
        message = self._response_value(response, "message") or "no message"
        raise RuntimeError(
            f"DashScope request failed: status_code={status_code}, "
            f"code={code}, message={message}"
        )

    def _extract_text(self, response: Any) -> str:
        self._raise_for_bad_response(response)
        output = self._response_value(response, "output")
        if not output:
            code = self._response_value(response, "code") or "unknown"
            message = self._response_value(response, "message") or "no message"
            raise RuntimeError(
                "DashScope response did not contain output: "
                f"code={code}, message={message}"
            )

        choices = output.get("choices") if isinstance(output, dict) else None
        if not choices:
            raise RuntimeError("DashScope response did not contain choices.")

        message = choices[0].get("message", {})
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if not content:
            raise RuntimeError("DashScope response did not contain content.")

        first_content = content[0]
        if isinstance(first_content, dict):
            text = first_content.get("text")
        else:
            text = getattr(first_content, "text", None)
        if not text:
            raise RuntimeError("DashScope response did not contain text.")
        return str(text)

    def answer_yes_no(self, question: str, video_path: Path) -> str:
        """Asks one yes/no question about a video.

        Args:
            question: Question text.
            video_path: Local video path.

        Returns:
            Raw answer text from the VLM.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": f"file://{video_path.resolve()}",
                        "fps": self._fps,
                    },
                    {
                        "text": (
                            "Answer the following question. "
                            f"Only output yes or no.\n{question}"
                        ),
                    },
                ],
            }
        ]
        response = self._client.call(
            api_key=self._api_key,
            model=self._model,
            messages=messages,
        )
        return self._extract_text(response)

