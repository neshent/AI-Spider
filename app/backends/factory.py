"""
Backend factory — maps a model_id string to the correct LLMBackend instance.

Model ID conventions:
  mock              -> MockLLMBackend (offline)
  lmstudio          -> LMStudioBackend (local server)
  lmstudio/<name>   -> LMStudioBackend with a specific model name
  hf-local/<id>     -> HFLocalBackend  (on-device, no internet after download)
  hf-api/<id>       -> HFAPIBackend    (HF Inference API, free token)
  claude-*          -> AnthropicLLMBackend
  gpt-* / o1* / o3* -> OpenAIBackend
  gemini-*          -> GeminiBackend
"""

from __future__ import annotations

import os

from src.lei.llm import LLMBackend, MockLLMBackend


def make_backend(
    model_id: str,
    hf_token: str = "",
    lmstudio_url: str = "",
    lmstudio_model: str = "",
) -> LLMBackend:
    if not model_id or model_id == "mock":
        return MockLLMBackend()

    if model_id == "lmstudio" or model_id.startswith("lmstudio/"):
        from .lmstudio import LMStudioBackend
        base_url = lmstudio_url or "http://localhost:1234"
        model = lmstudio_model or (
            model_id[len("lmstudio/"):] if "/" in model_id else ""
        )
        return LMStudioBackend(model=model, base_url=base_url)

    if model_id.startswith("hf-local/"):
        from .hf_local import HFLocalBackend
        return HFLocalBackend(model_id[len("hf-local/"):])

    if model_id.startswith("hf-api/"):
        from .hf_api import HFAPIBackend
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError(
                "A HuggingFace token is required for HF API models. "
                "Get one free at https://huggingface.co/settings/tokens"
            )
        return HFAPIBackend(model_id[len("hf-api/"):], token=token)

    if model_id.startswith("claude"):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
        from src.lei.llm import AnthropicLLMBackend
        return AnthropicLLMBackend(model=model_id)

    if model_id.startswith(("gpt", "o1", "o3")):
        from .cloud import OpenAIBackend
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
        return OpenAIBackend(model=model_id, api_key=api_key)

    if model_id.startswith("gemini"):
        from .cloud import GeminiBackend
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set the GOOGLE_API_KEY environment variable.")
        return GeminiBackend(model=model_id, api_key=api_key)

    raise RuntimeError(f"Unknown model '{model_id}'.")
