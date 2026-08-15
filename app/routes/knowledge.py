"""Route: add documents to the shared RAG knowledge base."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from app.pipeline import _extra_knowledge

knowledge_bp = Blueprint("knowledge", __name__)


@knowledge_bp.route("/api/knowledge", methods=["POST"])
def add_knowledge():
    body = request.get_json(force=True, silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip()
    text = (body.get("text") or "").strip()
    if not doc_id or not text:
        return jsonify({"error": "doc_id and text are required"}), 400
    _extra_knowledge.append((doc_id, text))
    return jsonify({"status": "ok", "doc_id": doc_id})
