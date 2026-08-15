#!/usr/bin/env python3
"""
main.py — unified entry point for Lei.

Usage:
    python main.py web              Start the web UI  (http://localhost:5000)
    python main.py cli "message"    Send a single message via CLI
    python main.py cli --interactive  Start interactive CLI REPL
    python main.py cli --help       Show CLI options

Both modes talk exclusively to backend/core.py.
They never call each other.
"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0

    mode = sys.argv.pop(1).lower()   # remove mode arg so sub-parsers see clean argv

    if mode == "web":
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

        import web_app
        port = int(os.environ.get("PORT", 5000))
        print(f"\n  Lei  —  http://localhost:{port}\n")
        web_app.app.run(debug=False, port=port)
        return 0

    if mode == "cli":
        import cli
        return cli.main()

    print(f"Unknown mode '{mode}'. Use: web | cli")
    return 1


if __name__ == "__main__":
    sys.exit(main())
