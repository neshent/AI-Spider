"""
HuggingFace Local backend — runs a model on-device via transformers + torch.
No account or internet required after the first model download.
Models are cached in ~/.cache/huggingface/hub by default.

Install deps: pip install transformers torch
"""

from __future__ import annotations

import threading
from typing import Optional

from src.lei.llm import LLMBackend

_pipeline_cache: dict = {}
_pipeline_lock = threading.Lock()


class HFLocalBackend(LLMBackend):
    def __init__(self, model_id: str, max_new_tokens: int = 512):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError:
            raise RuntimeError("Run: pip install transformers torch")

        with _pipeline_lock:
            if model_id not in _pipeline_cache:
                import torch  # type: ignore
                device = 0 if torch.cuda.is_available() else -1
                _pipeline_cache[model_id] = pipeline(
                    "text-generation",
                    model=model_id,
                    device=device,
                    trust_remote_code=True,
                )
            self._pipe = _pipeline_cache[model_id]

        self._model_id = model_id
        self._max_tokens = max_new_tokens

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            out = self._pipe(
                messages,
                max_new_tokens=self._max_tokens,
                do_sample=False,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
            )
            generated = out[0]["generated_text"]
            if isinstance(generated, list):
                for turn in reversed(generated):
                    if isinstance(turn, dict) and turn.get("role") == "assistant":
                        return turn["content"].strip()
                return str(generated[-1]).strip()
            return str(generated).strip()
        except Exception:
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            out = self._pipe(
                full_prompt,
                max_new_tokens=self._max_tokens,
                do_sample=False,
                return_full_text=False,
            )
            return out[0]["generated_text"].strip()
