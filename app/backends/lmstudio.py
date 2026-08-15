"""
LM Studio backend — talks to LM Studio's local OpenAI-compatible server.

Start LM Studio -> load any model -> enable Local Server (default port 1234).
No account, no API key, no internet needed.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Generator, Optional, Tuple

from src.lei.llm import LLMBackend


def normalize_lmstudio_url(base_url: str) -> str:
    """
    Normalise any user-supplied LM Studio URL to http://host:port/v1.

    Accepted inputs -> normalised output:
      http://host:1234          -> http://host:1234/v1
      http://host:1234/v1       -> http://host:1234/v1  (unchanged)
      http://host:1234/api/v1   -> http://host:1234/v1  (corrected)
    """
    url = base_url.rstrip("/")
    url = re.sub(r"/(api/)?v\d+$", "", url)
    return url + "/v1"


def lmstudio_probe(base_url: str = "http://localhost:1234") -> dict:
    """Return {'running': bool, 'models': [str]}."""
    api_root = normalize_lmstudio_url(base_url)
    try:
        req = urllib.request.Request(
            f"{api_root}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            data = json.loads(r.read().decode())
            models = [m["id"] for m in data.get("data", [])]
            return {"running": True, "models": models}
    except Exception:
        return {"running": False, "models": []}


class LMStudioBackend(LLMBackend):
    """Synchronous completion via LM Studio's OpenAI-compatible /chat/completions."""

    def __init__(
        self,
        model: str = "",
        base_url: str = "http://localhost:1234",
        max_tokens: int = 1024,
    ):
        self._base_url = normalize_lmstudio_url(base_url)
        self._model = model
        self._max_tokens = max_tokens

    def _build_payload(self, prompt: str, system: Optional[str], stream: bool) -> bytes:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict = {
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": 0.7,
            "stream": stream,
        }
        if self._model:
            payload["model"] = self._model
        return json.dumps(payload).encode()

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        data = self._build_payload(prompt, system, stream=False)
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            raise RuntimeError(
                f"LM Studio HTTP {exc.code}. Make sure a model is loaded.\n{raw[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach LM Studio at {self._base_url}. "
                "Open LM Studio -> Local Server tab -> Start Server."
            ) from exc

        if "choices" not in body:
            raise RuntimeError(
                f"LM Studio response missing 'choices'. Keys: {list(body.keys())}"
            )
        msg = body["choices"][0]["message"]
        text = (msg.get("content") or "").strip()
        if not text:
            text = (msg.get("reasoning_content") or "").strip()
        if not text:
            raise RuntimeError("LM Studio returned an empty response.")
        return text

    def complete_stream(
        self, prompt: str, system: Optional[str] = None
    ) -> Generator[Tuple[str, str], None, None]:
        """
        Yields (event_type, chunk) tuples.
        event_type: 'thinking' | 'content' | 'done' | 'error'
        """
        data = self._build_payload(prompt, system, stream=True)
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                for raw_line in resp:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if payload_str == "[DONE]":
                        yield ("done", "")
                        return
                    try:
                        chunk = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    thinking = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""
                    if thinking:
                        yield ("thinking", thinking)
                    if content:
                        yield ("content", content)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            yield ("error", f"LM Studio HTTP {exc.code}: {raw[:300]}")
        except urllib.error.URLError as exc:
            yield ("error", f"Cannot reach LM Studio at {self._base_url}: {exc.reason}")
        except Exception as exc:
            yield ("error", str(exc))
