"""
web_app.py — Flask web interface for Lei.

Talks exclusively to backend.core. Does not import from src/lei or app/ directly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Load .env
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from backend import AgentRequest, backend

app = Flask(__name__, template_folder="templates", static_folder="static")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------------
# API: models
# ---------------------------------------------------------------------------

@app.route("/api/models")
def list_models():
    # No models registered yet — returns empty list so select stays grayed out
    return jsonify({"models": []})


# ---------------------------------------------------------------------------
# API: knowledge
# ---------------------------------------------------------------------------

@app.route("/api/knowledge", methods=["POST"])
def add_knowledge():
    body   = request.get_json(force=True, silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()
    text   = (body.get("text")   or "").strip()
    if not doc_id or not text:
        return jsonify({"error": "doc_id and text are required"}), 400
    backend.add_knowledge(doc_id, text)
    return jsonify({"status": "ok", "doc_id": doc_id})


# ---------------------------------------------------------------------------
# API: streaming chat
# ---------------------------------------------------------------------------

@app.route("/api/stream", methods=["POST"])
def stream_chat():
    body    = request.get_json(force=True, silent=True) or {}
    message = (body.get("message") or "").strip()
    model   = (body.get("model")   or "").strip()

    if not message:
        return jsonify({"error": "message is required"}), 400

    def generate():
        def sse(event: str, data: str) -> str:
            lines = "\n".join(f"data:{l}" for l in data.split("\n"))
            return f"event: {event}\n{lines}\n\n"

        if not model:
            yield sse("error", "No model selected. Add a model first.")
            return

        req = AgentRequest(message=message, model_id=model)
        for chunk in backend.stream(req):
            if chunk.startswith("[error]"):
                yield sse("error", chunk[len("[error]"):].strip())
                return
            yield sse("content", chunk)
        yield sse("done", "")

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Lei  —  http://localhost:{port}\n")
    app.run(debug=False, port=port)
