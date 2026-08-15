"""
Cloud LLM backends: OpenAI (GPT) and Google (Gemini).

Each requires the respective SDK and API key set in the environment:
  OpenAI  -> pip install openai        + OPENAI_API_KEY
  Google  -> pip install google-generativeai + GOOGLE_API_KEY
"""

from __future__ import annotations

from typing import Optional

from src.lei.llm import LLMBackend


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str, api_key: str):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise RuntimeError("Run: pip install openai")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=1024
        )
        return resp.choices[0].message.content.strip()


class GeminiBackend(LLMBackend):
    def __init__(self, model: str, api_key: str):
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            raise RuntimeError("Run: pip install google-generativeai")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        return self._model.generate_content(full).text.strip()
