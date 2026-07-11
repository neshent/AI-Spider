"""
Web UI for the Reasoning Agent Harness.

Run with:
    python web_app.py

Then open http://localhost:5000 in your browser.

Supported model groups:
  mock        - offline deterministic MockLLMBackend (zero deps)
  lmstudio    - LM Studio local server (OpenAI-compatible, default http://localhost:1234/v1)
  hf-local/*  - HuggingFace model run LOCALLY via transformers + torch
                (no account, no internet after first download)
  hf-api/*    - HuggingFace Inference API (free tier; needs HF_TOKEN env var
                OR token entered in the UI)
  claude-*    - Anthropic  (needs ANTHROPIC_API_KEY)
  gpt-* / o*  - OpenAI     (needs OPENAI_API_KEY)
  gemini-*    - Google      (needs GOOGLE_API_KEY)
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Optional

# Load .env from the same directory as this file (if present)
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from agent_harness import Orchestrator
from agent_harness.llm import LLMBackend, MockLLMBackend

logging.getLogger("agent_harness.workflow").setLevel(logging.WARNING)

app = Flask(__name__, template_folder="templates")

# Accumulated knowledge docs added by the user via /api/knowledge
_extra_knowledge: list[tuple[str, str]] = []

# Cache loaded local HF pipelines so we don't reload on every request
_hf_pipeline_cache: dict[str, object] = {}
_hf_pipeline_lock = threading.Lock()


# ── LM Studio backend ────────────────────────────────────────────────────

def _lms_normalize_url(base_url: str) -> str:
    """
    Normalize the user-supplied LM Studio URL to the correct API root.

    LM Studio exposes OpenAI-compatible endpoints at /v1/...
    The /api/v1/ path exists but only for LM Studio's own management API,
    NOT for chat completions.

    Accepted inputs → normalized output (all → http://host:port/v1):
      http://host:1234            → http://host:1234/v1
      http://host:1234/           → http://host:1234/v1
      http://host:1234/v1         → http://host:1234/v1   (unchanged)
      http://host:1234/api/v1     → http://host:1234/v1   (corrected)
      http://host:1234/api/v0     → http://host:1234/v1   (corrected)
    """
    url = base_url.rstrip("/")
    # Strip any /api/vN or /vN suffix so we always re-append /v1
    import re
    url = re.sub(r"/(api/)?v\d+$", "", url)
    return url + "/v1"


class LMStudioBackend(LLMBackend):
    """
    Talks to LM Studio's local OpenAI-compatible server.
    Start LM Studio → load any model → enable "Local Server" (default port 1234).
    No account, no API key, no internet needed — completely free.
    """

    def __init__(self, model: str = "", base_url: str = "http://localhost:1234",
                 max_tokens: int = 1024):
        self._base_url   = _lms_normalize_url(base_url)
        self._model      = model  # empty string → LM Studio uses whatever is loaded
        self._max_tokens = max_tokens

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        import urllib.request
        import urllib.error
        import json

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "messages":    messages,
            "max_tokens":  self._max_tokens,
            "temperature": 0.7,
            "stream":      False,
        }
        if self._model:
            payload["model"] = self._model

        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
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
                f"LM Studio returned HTTP {exc.code}. "
                f"Make sure a model is loaded and the server is running.\nDetail: {raw[:300]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach LM Studio at {self._base_url}. "
                "Make sure LM Studio is open, a model is loaded, and the Local Server is running "
                "(LM Studio → Local Server tab → Start Server)."
            ) from exc

        # Parse response — be explicit so errors are easy to diagnose
        if "choices" not in body:
            raise RuntimeError(
                f"LM Studio response missing 'choices'. Got keys: {list(body.keys())}. "
                f"Full response: {json.dumps(body)[:400]}"
            )
        try:
            msg = body["choices"][0]["message"]
            # Some reasoning models return content in 'reasoning_content' when
            # 'content' is empty (e.g. thinking/R1-style models).
            text = (msg.get("content") or "").strip()
            if not text:
                text = (msg.get("reasoning_content") or "").strip()
            if not text:
                raise RuntimeError(
                    f"LM Studio returned an empty response. "
                    f"Message keys: {list(msg.keys())}. The model may still be loading."
                )
            return text
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Unexpected LM Studio response structure: {exc}. "
                f"Response: {json.dumps(body)[:400]}"
            ) from exc


    def complete_stream(self, prompt: str, system: Optional[str] = None):
        """
        Generator that yields (event_type, data) tuples from a streaming LM Studio call.
        event_type is one of: 'thinking', 'content', 'done', 'error'
        """
        import urllib.request, urllib.error, json

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "messages":    messages,
            "max_tokens":  self._max_tokens,
            "temperature": 0.7,
            "stream":      True,
        }
        if self._model:
            payload["model"] = self._model

        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
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
                    # reasoning_content = thinking tokens (R1/QwQ style)
                    thinking_delta = delta.get("reasoning_content") or ""
                    content_delta  = delta.get("content") or ""
                    if thinking_delta:
                        yield ("thinking", thinking_delta)
                    if content_delta:
                        yield ("content", content_delta)
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode(errors="replace")
            yield ("error", f"LM Studio HTTP {exc.code}: {raw[:300]}")
        except urllib.error.URLError as exc:
            yield ("error", f"Cannot reach LM Studio at {self._base_url}: {exc.reason}")
        except Exception as exc:
            yield ("error", str(exc))

def _lmstudio_probe(base_url: str = "http://localhost:1234") -> dict:
    """Return {'running': bool, 'models': [str]}."""
    import urllib.request, urllib.error, json
    api_root = _lms_normalize_url(base_url)
    try:
        req = urllib.request.Request(
            f"{api_root}/models",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as r:
            data   = json.loads(r.read().decode())
            models = [m["id"] for m in data.get("data", [])]
            return {"running": True, "models": models}
    except Exception:
        return {"running": False, "models": []}




class HFLocalBackend(LLMBackend):
    """
    Runs a HuggingFace model LOCALLY using the transformers pipeline.
    No account or internet required after the first model download.
    Models are cached in ~/.cache/huggingface/hub by default.
    """

    def __init__(self, model_id: str, max_new_tokens: int = 512):
        try:
            from transformers import pipeline  # type: ignore
        except ImportError:
            raise RuntimeError("Run: pip install transformers torch")

        with _hf_pipeline_lock:
            if model_id not in _hf_pipeline_cache:
                import torch  # type: ignore
                device = 0 if torch.cuda.is_available() else -1
                _hf_pipeline_cache[model_id] = pipeline(
                    "text-generation",
                    model=model_id,
                    device=device,
                    trust_remote_code=True,
                )
            self._pipe = _hf_pipeline_cache[model_id]

        self._model_id   = model_id
        self._max_tokens = max_new_tokens

    def complete(self, prompt: str, system: Optional[str] = None) -> str:
        # Build a chat-style prompt when the model supports it, otherwise
        # pass the raw text.
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
            # transformers chat output: list[{generated_text: list[{role,content}]}]
            generated = out[0]["generated_text"]
            if isinstance(generated, list):
                # find the last assistant turn
                for turn in reversed(generated):
                    if isinstance(turn, dict) and turn.get("role") == "assistant":
                        return turn["content"].strip()
                return str(generated[-1]).strip()
            return str(generated).strip()
        except Exception:
            # Fallback: plain text generation
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            out = self._pipe(
                full_prompt,
                max_new_tokens=self._max_tokens,
                do_sample=False,
                return_full_text=False,
            )
            return out[0]["generated_text"].strip()


# ── HuggingFace Inference API backend ────────────────────────────────────

class HFAPIBackend(LLMBackend):
    """
    Uses the HuggingFace Inference API (serverless). Requires a free HF token.
    Free accounts get $0.10/month in credits — enough for thousands of small requests.
    Get a token at https://huggingface.co/settings/tokens (free account).
    """

    BASE_URL = "https://router.huggingface.co/hf-inference/models"

    def __init__(self, model_id: str, token: str, max_new_tokens: int = 512):
        try:
            import requests  # noqa: F401
        except ImportError:
            raise RuntimeError("Run: pip install requests")
        self._model_id   = model_id
        self._token      = token
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
            url = f"{self.BASE_URL}/{self._model_id}/v1/chat/completions"
            resp = requests.post(
                url,
                headers=headers,
                json={"model": self._model_id, "messages": messages, "max_tokens": self._max_tokens},
                timeout=60,
            )
            if resp.ok:
                data = resp.json()
                return data["choices"][0]["message"]["content"].strip()

            # Fallback to text-generation endpoint
            url2 = f"https://api-inference.huggingface.co/models/{self._model_id}"
            full_prompt = f"{system}\n\n{prompt}" if system else prompt
            resp2 = requests.post(
                url2,
                headers=headers,
                json={"inputs": full_prompt, "parameters": {"max_new_tokens": self._max_tokens}},
                timeout=60,
            )
            if resp2.ok:
                data2 = resp2.json()
                if isinstance(data2, list):
                    return data2[0].get("generated_text", "").strip()
            raise RuntimeError(f"HF API error {resp2.status_code}: {resp2.text[:300]}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot reach HuggingFace API — no internet connection detected. "
                "Use a HF Local model instead (they run fully offline on your machine)."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError("HuggingFace API request timed out. Try again or use a local model.")


# ── Other cloud backends ───────────────────────────────────────────────────

class _OpenAIBackend(LLMBackend):
    def __init__(self, model: str, api_key: str):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model  = model

    def complete(self, prompt: str, system=None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self._model, messages=messages, max_tokens=1024
        )
        return resp.choices[0].message.content.strip()


class _GeminiBackend(LLMBackend):
    def __init__(self, model: str, api_key: str):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    def complete(self, prompt: str, system=None) -> str:
        full = f"{system}\n\n{prompt}" if system else prompt
        return self._model.generate_content(full).text.strip()


# ── Backend factory ────────────────────────────────────────────────────────

def _make_backend(model_id: str, hf_token: str = "",
                  lmstudio_url: str = "", lmstudio_model: str = "") -> LLMBackend:
    if not model_id or model_id == "mock":
        return MockLLMBackend()

    # LM Studio  (lmstudio or lmstudio/<model-name>)
    if model_id == "lmstudio" or model_id.startswith("lmstudio/"):
        base_url = lmstudio_url or "http://localhost:1234/v1"
        model    = lmstudio_model or (model_id[len("lmstudio/"):] if "/" in model_id else "")
        return LMStudioBackend(model=model, base_url=base_url)

    # Local HuggingFace model  (hf-local/org/model-name)
    if model_id.startswith("hf-local/"):
        hf_model = model_id[len("hf-local/"):]
        return HFLocalBackend(hf_model)

    # HF Inference API  (hf-api/org/model-name)
    if model_id.startswith("hf-api/"):
        token = hf_token or os.environ.get("HF_TOKEN", "")
        if not token:
            raise RuntimeError(
                "A free HuggingFace token is required for HF API models. "
                "Get one at https://huggingface.co/settings/tokens and paste it in the Token field."
            )
        hf_model = model_id[len("hf-api/"):]
        return HFAPIBackend(hf_model, token=token)

    if model_id.startswith("claude"):
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise RuntimeError("Run: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable.")
        from agent_harness.llm import AnthropicLLMBackend
        return AnthropicLLMBackend(model=model_id)

    if model_id.startswith("gpt") or model_id.startswith("o1") or model_id.startswith("o3"):
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError:
            raise RuntimeError("Run: pip install openai")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set the OPENAI_API_KEY environment variable.")
        return _OpenAIBackend(model=model_id, api_key=api_key)

    if model_id.startswith("gemini"):
        try:
            import google.generativeai  # noqa: F401
        except ImportError:
            raise RuntimeError("Run: pip install google-generativeai")
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set the GOOGLE_API_KEY environment variable.")
        return _GeminiBackend(model=model_id, api_key=api_key)

    raise RuntimeError(f"Unknown model '{model_id}'.")


# ── Pipeline runner ────────────────────────────────────────────────────────

def _run(user_message: str, model_id: str, hf_token: str = "",
         lmstudio_url: str = "", lmstudio_model: str = "") -> dict:
    llm  = _make_backend(model_id, hf_token=hf_token,
                         lmstudio_url=lmstudio_url, lmstudio_model=lmstudio_model)
    orch = Orchestrator(llm=llm)
    orch.add_knowledge(
        "stack_notes",
        "FastAPI is a modern, fast Python web framework for building APIs. "
        "SQLite and PostgreSQL are common relational databases.",
    )
    for doc_id, text in _extra_knowledge:
        orch.add_knowledge(doc_id, text)
    trace = orch.handle(user_message)
    return {
        "final_response":   trace.final_response,
        "intent":           trace.intent,
        "handled_reactively": trace.handled_reactively,
        "plan":             [{"index": s.index, "description": s.description} for s in (trace.plan or [])],
        "retrieved_context": trace.retrieved_context or [],
        "tool_used":        trace.tool_used,
        "tool_output":      trace.tool_output,
        "multi_agent_report": trace.multi_agent_report.summary() if trace.multi_agent_report else None,
        "workflow": (
            {"attempts": trace.workflow_result.attempts, "success": trace.workflow_result.success}
            if trace.workflow_result else None
        ),
    }


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def list_models():
    hf_token_set = bool(os.environ.get("HF_TOKEN"))
    lms = _lmstudio_probe()
    return jsonify({
        "models": [
            # ── Offline ──────────────────────────────────────────────────
            {"id": "mock",        "label": "Mock (offline)",   "provider": "mock",      "free": True},
            # ── LM Studio ────────────────────────────────────────────────
            {"id": "lmstudio",    "label": "LM Studio (local server)", "provider": "lmstudio", "free": True},
            # ── HF Local (truly free, no account) ────────────────────────
            {"id": "hf-local/HuggingFaceTB/SmolLM2-135M-Instruct",  "label": "SmolLM2 135M (local)",   "provider": "hf-local", "free": True},
            {"id": "hf-local/HuggingFaceTB/SmolLM2-360M-Instruct",  "label": "SmolLM2 360M (local)",   "provider": "hf-local", "free": True},
            {"id": "hf-local/Qwen/Qwen2.5-0.5B-Instruct",           "label": "Qwen2.5 0.5B (local)",   "provider": "hf-local", "free": True},
            {"id": "hf-local/Qwen/Qwen2.5-1.5B-Instruct",           "label": "Qwen2.5 1.5B (local)",   "provider": "hf-local", "free": True},
            {"id": "hf-local/microsoft/phi-2",                       "label": "Phi-2 2.7B (local)",     "provider": "hf-local", "free": True},
            {"id": "hf-local/TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "label": "TinyLlama 1.1B (local)", "provider": "hf-local", "free": True},
            # ── HF Inference API (free token, optional) ───────────────────
            {"id": "hf-api/HuggingFaceTB/SmolLM2-1.7B-Instruct",   "label": "SmolLM2 1.7B (HF API)",  "provider": "hf-api",   "free": True},
            {"id": "hf-api/Qwen/Qwen2.5-7B-Instruct",               "label": "Qwen2.5 7B (HF API)",    "provider": "hf-api",   "free": True},
            {"id": "hf-api/meta-llama/Llama-3.2-3B-Instruct",       "label": "Llama 3.2 3B (HF API)",  "provider": "hf-api",   "free": True},
            {"id": "hf-api/mistralai/Mistral-7B-Instruct-v0.3",     "label": "Mistral 7B (HF API)",    "provider": "hf-api",   "free": True},
            {"id": "hf-api/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","label": "DeepSeek-R1 7B (HF API)","provider": "hf-api",   "free": True},
            # ── Commercial ───────────────────────────────────────────────
            {"id": "claude-sonnet-4-5",   "label": "Claude Sonnet 4.5",  "provider": "anthropic", "free": False},
            {"id": "claude-haiku-3-5",    "label": "Claude Haiku 3.5",   "provider": "anthropic", "free": False},
            {"id": "gpt-4o",              "label": "GPT-4o",              "provider": "openai",    "free": False},
            {"id": "gpt-4o-mini",         "label": "GPT-4o Mini",         "provider": "openai",    "free": False},
            {"id": "gemini-2.0-flash",    "label": "Gemini 2.0 Flash",    "provider": "google",    "free": False},
        ],
        "keys": {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openai":    bool(os.environ.get("OPENAI_API_KEY")),
            "google":    bool(os.environ.get("GOOGLE_API_KEY")),
            "hf-api":    hf_token_set,
            "lmstudio":  lms["running"],
        },
        "lmstudio": lms,
    })


@app.route("/api/lmstudio/probe", methods=["POST"])
def lmstudio_probe():
    body     = request.get_json(force=True, silent=True) or {}
    base_url = (body.get("base_url") or "http://localhost:1234/v1").strip()
    return jsonify(_lmstudio_probe(base_url))


@app.route("/api/chat", methods=["POST"])
def chat():
    body             = request.get_json(force=True, silent=True) or {}
    user_message     = (body.get("message")        or "").strip()
    model_id         = (body.get("model")          or "mock").strip()
    hf_token         = (body.get("hf_token")       or "").strip()
    lmstudio_url     = (body.get("lmstudio_url")   or "").strip()
    lmstudio_model   = (body.get("lmstudio_model") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        payload = _run(user_message, model_id, hf_token=hf_token,
                       lmstudio_url=lmstudio_url, lmstudio_model=lmstudio_model)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(payload)


@app.route("/api/knowledge", methods=["POST"])
def add_knowledge():
    body   = request.get_json(force=True, silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()
    text   = (body.get("text")   or "").strip()
    if not doc_id or not text:
        return jsonify({"error": "doc_id and text are required"}), 400
    _extra_knowledge.append((doc_id, text))
    return jsonify({"status": "ok", "doc_id": doc_id})


@app.route("/api/stream", methods=["POST"])
def stream_chat():
    """
    SSE endpoint. Emits these event types:
      pipeline  – a pipeline stage completed (JSON: {stage, data})
      thinking  – a reasoning/thinking token chunk (plain text)
      content   – a response token chunk (plain text)
      done      – generation finished
      error     – an error occurred (plain text)
    """
    body             = request.get_json(force=True, silent=True) or {}
    user_message     = (body.get("message")        or "").strip()
    model_id         = (body.get("model")          or "mock").strip()
    hf_token         = (body.get("hf_token")       or "").strip()
    lmstudio_url     = (body.get("lmstudio_url")   or "").strip()
    lmstudio_model   = (body.get("lmstudio_model") or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    def generate():
        import json as _json

        def sse(event: str, data: str) -> str:
            # Emit each line of data without an extra space after "data:"
            # so token spaces are preserved verbatim in the browser.
            data_lines = "\n".join(f"data:{line}" for line in data.split("\n"))
            return f"event: {event}\n{data_lines}\n\n"

        try:
            llm  = _make_backend(model_id, hf_token=hf_token,
                                 lmstudio_url=lmstudio_url, lmstudio_model=lmstudio_model)
            orch = Orchestrator(llm=llm)
            orch.add_knowledge(
                "stack_notes",
                "FastAPI is a modern, fast Python web framework for building APIs. "
                "SQLite and PostgreSQL are common relational databases.",
            )
            for doc_id, text in _extra_knowledge:
                orch.add_knowledge(doc_id, text)

            # ── For LM Studio: stream the raw LLM call first, then run the
            #    rest of the pipeline non-streaming so we get trace data.
            if isinstance(llm, LMStudioBackend):
                # 1. Stream thinking + content tokens directly from LM Studio
                full_thinking = []
                full_content  = []
                for evt, chunk in llm.complete_stream(user_message):
                    if evt == "thinking":
                        full_thinking.append(chunk)
                        yield sse("thinking", chunk)
                    elif evt == "content":
                        full_content.append(chunk)
                        yield sse("content", chunk)
                    elif evt == "error":
                        yield sse("error", chunk)
                        return
                    elif evt == "done":
                        break

                assembled = "".join(full_content) or "".join(full_thinking)

                # 2. Now run the full pipeline for trace data, but inject the
                #    already-generated response so we don't call LM Studio twice.
                #    We use a pre-baked backend that returns the cached response.
                class _CachedBackend(LLMBackend):
                    def complete(self, p, system=None):
                        return assembled

                orch2 = Orchestrator(llm=_CachedBackend())
                orch2.add_knowledge("stack_notes",
                    "FastAPI is a modern, fast Python web framework. "
                    "SQLite and PostgreSQL are common databases.")
                for doc_id, text in _extra_knowledge:
                    orch2.add_knowledge(doc_id, text)
                trace = orch2.handle(user_message)

            else:
                # Non-streaming path: run full pipeline, then animate the response
                # in the browser by sending the full text as one content event.
                # The frontend will typewriter-animate it client-side.
                trace = orch.handle(user_message)
                assembled = trace.final_response
                yield sse("content", assembled)

            # 3. Emit pipeline trace as a single event
            trace_payload = _json.dumps({
                "intent":           trace.intent,
                "handled_reactively": trace.handled_reactively,
                "plan":             [{"index": s.index, "description": s.description}
                                     for s in (trace.plan or [])],
                "retrieved_context": trace.retrieved_context or [],
                "tool_used":        trace.tool_used,
                "tool_output":      trace.tool_output,
                "multi_agent_report": (trace.multi_agent_report.summary()
                                       if trace.multi_agent_report else None),
                "workflow": ({"attempts": trace.workflow_result.attempts,
                              "success":  trace.workflow_result.success}
                             if trace.workflow_result else None),
            })
            yield sse("pipeline", trace_payload)
            yield sse("done", "")

        except RuntimeError as exc:
            yield sse("error", str(exc))
        except Exception as exc:
            yield sse("error", str(exc))

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering if behind proxy
        },
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Reasoning Agent Harness  —  http://localhost:{port}\n")
    app.run(debug=False, port=port)
