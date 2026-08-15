"""Routes: model listing and LM Studio probe."""

from __future__ import annotations

import os

from flask import Blueprint, jsonify, request

from app.backends.lmstudio import lmstudio_probe

models_bp = Blueprint("models", __name__)

_MODEL_LIST = [
    # ── Offline ──────────────────────────────────────────────────────────
    {"id": "mock",       "label": "Mock (offline)",          "provider": "mock",      "free": True},
    # ── LM Studio ────────────────────────────────────────────────────────
    {"id": "lmstudio",   "label": "LM Studio (local server)","provider": "lmstudio",  "free": True},
    # ── HF Local ─────────────────────────────────────────────────────────
    {"id": "hf-local/HuggingFaceTB/SmolLM2-135M-Instruct",  "label": "SmolLM2 135M (local)",   "provider": "hf-local", "free": True},
    {"id": "hf-local/HuggingFaceTB/SmolLM2-360M-Instruct",  "label": "SmolLM2 360M (local)",   "provider": "hf-local", "free": True},
    {"id": "hf-local/Qwen/Qwen2.5-0.5B-Instruct",           "label": "Qwen2.5 0.5B (local)",   "provider": "hf-local", "free": True},
    {"id": "hf-local/Qwen/Qwen2.5-1.5B-Instruct",           "label": "Qwen2.5 1.5B (local)",   "provider": "hf-local", "free": True},
    {"id": "hf-local/microsoft/phi-2",                       "label": "Phi-2 2.7B (local)",     "provider": "hf-local", "free": True},
    {"id": "hf-local/TinyLlama/TinyLlama-1.1B-Chat-v1.0",  "label": "TinyLlama 1.1B (local)", "provider": "hf-local", "free": True},
    # ── HF API ───────────────────────────────────────────────────────────
    {"id": "hf-api/HuggingFaceTB/SmolLM2-1.7B-Instruct",   "label": "SmolLM2 1.7B (HF API)",  "provider": "hf-api",   "free": True},
    {"id": "hf-api/Qwen/Qwen2.5-7B-Instruct",               "label": "Qwen2.5 7B (HF API)",    "provider": "hf-api",   "free": True},
    {"id": "hf-api/meta-llama/Llama-3.2-3B-Instruct",       "label": "Llama 3.2 3B (HF API)",  "provider": "hf-api",   "free": True},
    {"id": "hf-api/mistralai/Mistral-7B-Instruct-v0.3",     "label": "Mistral 7B (HF API)",    "provider": "hf-api",   "free": True},
    {"id": "hf-api/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","label": "DeepSeek-R1 7B (HF API)","provider": "hf-api",   "free": True},
    # ── Commercial ───────────────────────────────────────────────────────
    {"id": "claude-sonnet-4-5", "label": "Claude Sonnet 4.5", "provider": "anthropic", "free": False},
    {"id": "claude-haiku-3-5",  "label": "Claude Haiku 3.5",  "provider": "anthropic", "free": False},
    {"id": "gpt-4o",            "label": "GPT-4o",             "provider": "openai",    "free": False},
    {"id": "gpt-4o-mini",       "label": "GPT-4o Mini",        "provider": "openai",    "free": False},
    {"id": "gemini-2.0-flash",  "label": "Gemini 2.0 Flash",   "provider": "google",    "free": False},
]


@models_bp.route("/api/models")
def list_models():
    lms = lmstudio_probe()
    return jsonify({
        "models": _MODEL_LIST,
        "keys": {
            "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY")),
            "openai":    bool(os.environ.get("OPENAI_API_KEY")),
            "google":    bool(os.environ.get("GOOGLE_API_KEY")),
            "hf-api":    bool(os.environ.get("HF_TOKEN")),
            "lmstudio":  lms["running"],
        },
        "lmstudio": lms,
    })


@models_bp.route("/api/lmstudio/probe", methods=["POST"])
def probe_lmstudio():
    body = request.get_json(force=True, silent=True) or {}
    base_url = (body.get("base_url") or "http://localhost:1234").strip()
    return jsonify(lmstudio_probe(base_url))
