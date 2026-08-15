"""Routes: /api/stream (SSE) and /api/chat (JSON)."""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify, request, stream_with_context

from app.pipeline import build_orchestrator

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True, silent=True) or {}
    user_message = (body.get("message") or "").strip()
    model_id     = (body.get("model")   or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400
    if not model_id:
        return jsonify({"error": "No model selected. Add a model first."}), 400

    try:
        from app.backends.factory import make_backend
        llm   = make_backend(model_id)
        orch  = build_orchestrator(llm)
        trace = orch.handle(user_message)
        return jsonify({"response": trace.final_response})
    except NotImplementedError as exc:
        return jsonify({"error": str(exc)}), 501
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@chat_bp.route("/api/stream", methods=["POST"])
def stream_chat():
    body = request.get_json(force=True, silent=True) or {}
    user_message = (body.get("message") or "").strip()
    model_id     = (body.get("model")   or "").strip()

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    def generate():
        def sse(event: str, data: str) -> str:
            lines = "\n".join(f"data:{l}" for l in data.split("\n"))
            return f"event: {event}\n{lines}\n\n"

        if not model_id:
            yield sse("error", "No model selected. Add a model first.")
            return

        try:
            from app.backends.factory import make_backend
            llm   = make_backend(model_id)
            orch  = build_orchestrator(llm)
            trace = orch.handle(user_message)
            yield sse("content", trace.final_response or "")
            yield sse("done", "")
        except NotImplementedError as exc:
            yield sse("error", str(exc))
        except Exception as exc:
            yield sse("error", str(exc))

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
