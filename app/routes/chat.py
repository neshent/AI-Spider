"""
Routes: /api/chat (sync JSON) and /api/stream (SSE streaming).
"""

from __future__ import annotations

import json

from flask import Blueprint, Response, jsonify, request, stream_with_context

from app.backends.lmstudio import LMStudioBackend
from app.pipeline import build_orchestrator, run_pipeline, trace_to_dict
from src.lei.llm import LLMBackend

chat_bp = Blueprint("chat", __name__)


def _parse_body(body: dict) -> tuple:
    return (
        (body.get("message")        or "").strip(),
        (body.get("model")          or "mock").strip(),
        (body.get("hf_token")       or "").strip(),
        (body.get("lmstudio_url")   or "").strip(),
        (body.get("lmstudio_model") or "").strip(),
    )


@chat_bp.route("/api/chat", methods=["POST"])
def chat():
    body = request.get_json(force=True, silent=True) or {}
    user_message, model_id, hf_token, lmstudio_url, lmstudio_model = _parse_body(body)

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    try:
        payload = run_pipeline(
            user_message, model_id,
            hf_token=hf_token,
            lmstudio_url=lmstudio_url,
            lmstudio_model=lmstudio_model,
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(payload)


@chat_bp.route("/api/stream", methods=["POST"])
def stream_chat():
    """
    SSE endpoint. Emits events:
      thinking  – reasoning token chunk (plain text)
      content   – response token chunk (plain text)
      pipeline  – full pipeline trace (JSON)
      done      – generation finished
      error     – error message (plain text)
    """
    body = request.get_json(force=True, silent=True) or {}
    user_message, model_id, hf_token, lmstudio_url, lmstudio_model = _parse_body(body)

    if not user_message:
        return jsonify({"error": "message is required"}), 400

    def generate():
        def sse(event: str, data: str) -> str:
            data_lines = "\n".join(f"data:{line}" for line in data.split("\n"))
            return f"event: {event}\n{data_lines}\n\n"

        try:
            from app.backends.factory import make_backend
            llm = make_backend(
                model_id, hf_token=hf_token,
                lmstudio_url=lmstudio_url, lmstudio_model=lmstudio_model,
            )
            orch = build_orchestrator(llm)

            if isinstance(llm, LMStudioBackend):
                # 1. Stream tokens directly from LM Studio
                full_thinking, full_content = [], []
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

                # 2. Re-run pipeline with cached response for trace data
                class _CachedBackend(LLMBackend):
                    def complete(self, p, system=None):
                        return assembled

                trace = build_orchestrator(_CachedBackend()).handle(user_message)
            else:
                # Non-streaming: run pipeline, animate client-side
                trace = orch.handle(user_message)
                yield sse("content", trace.final_response)

            yield sse("pipeline", json.dumps(trace_to_dict(trace)))
            yield sse("done", "")

        except RuntimeError as exc:
            yield sse("error", str(exc))
        except Exception as exc:
            yield sse("error", str(exc))

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
