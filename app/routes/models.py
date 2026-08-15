"""Routes: model listing. No models registered yet — add them one by one."""

from __future__ import annotations

from flask import Blueprint, jsonify

models_bp = Blueprint("models", __name__)

# Empty until models are added one by one
_MODEL_LIST = []


@models_bp.route("/api/models")
def list_models():
    return jsonify({"models": _MODEL_LIST})
