"""
Web UI entry point for the Lei Reasoning Agent.

Run with:
    python web_app.py

Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# Load .env from project root if present
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from flask import Flask, render_template

from app.routes import chat_bp, knowledge_bp, models_bp

logging.getLogger("lei.workflow").setLevel(logging.WARNING)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.register_blueprint(chat_bp)
app.register_blueprint(models_bp)
app.register_blueprint(knowledge_bp)


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Lei Reasoning Agent  —  http://localhost:{port}\n")
    app.run(debug=False, port=port)
