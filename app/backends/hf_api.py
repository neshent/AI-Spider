"""
HuggingFace Inference API backend.

Uses the serverless HF Inference API. Requires a free HF token.
Free accounts get $0.10/month in credits — enough for thousands of small requests.
Get a token at https://huggingface.co/settings/tokens

Install deps: pip install requests
"""

from __future__ import annotations

from typing import Optional

from src.lei.llm import LLMBackend

_BASE_CHAT = "https://router.huggingface.co/hf-inference/models"
_BASE_TEXT = "https://api-inference.huggingface.co/models"


class HFAPIBackend(LLMBackend):
    def __init__(self, model_id: str, token: str, max_new_tokens: int = 512):
        try:
            import requests  # noqa: F401
        except ImportError:
            raise RuntimeError("Run: pip install requests")
        self._model_id = model_id
        self._token = token
        self._max_tokens = max_new_tokens

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        import requests

        headers = {"Authorization": f"Bearer {self._token}"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            # Try chat-completions endpoint first
            resp = requests.post(
                f"{_BASE_CHAT}/{self._model_id}/v1/chat/completions",
                headers=headers,
                json={
                    "model": self._model_id,
                    "messages": messages,
                    "max_tokens": self._max_tokens,
                },
                timeout=60,
            )
            if resp.ok:
                return resp.json()["choices"][0]["message"]["content"].strip()

            # Fallback: text-generation endpoint
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            resp2 = requests.post(
                f"{_BASE_TEXT}/{self._model_id}",
                headers=headers,
                json={
                    "inputs": full_prompt,
                    "parameters": {"max_new_tokens": self._max_tokens},
                },
                timeout=60,
            )
            if resp2.ok:
                data = resp2.json()
                if isinstance(data, list):
                    return data[0].get("generated_text", "").strip()
            raise RuntimeError(f"HF API error {resp2.status_code}: {resp2.text[:300]}")

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach HuggingFace API — no internet connection. "
                "Use a HF Local model instead (runs fully offline)."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("HuggingFace API request timed out. Try again or use a local model.")
