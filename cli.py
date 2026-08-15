#!/usr/bin/env python3
"""
cli.py — Command-line interface for Lei.

Talks exclusively to backend.core. Does not import from src/lei or app/ directly.

Usage:
    python cli.py "What is machine learning?"
    python cli.py --interactive
    python cli.py --knowledge doc1="FastAPI is a Python web framework." "Build an API."
"""

from __future__ import annotations

import argparse
import sys

from backend import AgentRequest, AgentResponse, backend


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_response(response: AgentResponse, verbose: bool) -> None:
    if not response.success:
        print(f"\n[error] {response.error}\n")
        return
    if verbose and response.intent:
        print(f"\n--- intent: {response.intent} ---")
        if response.handled_reactively:
            print("[handled reactively]")
        if response.plan:
            print("plan:")
            for step in response.plan:
                print(f"  {step['index']}. {step['description']}")
        if response.tool_used:
            print(f"tool: {response.tool_used} -> {response.tool_output}")
        print("---")
    print(f"\n{response.text}\n")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_interactive(verbose: bool) -> int:
    print("Lei — interactive mode. Type 'exit' to quit.\n")
    try:
        while True:
            message = input("> ").strip()
            if message.lower() in ("exit", "quit"):
                break
            if not message:
                continue
            req      = AgentRequest(message=message)
            response = backend.handle(req)
            print_response(response, verbose=verbose)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Lei CLI")
    parser.add_argument("request",       nargs="?",        help="Message to send.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive REPL.")
    parser.add_argument("--quiet",       action="store_true", help="Hide pipeline details.")
    parser.add_argument(
        "--knowledge", action="append", default=[],
        metavar="id=text",
        help="Seed RAG knowledge: --knowledge doc1='some text'",
    )
    args = parser.parse_args()

    # Seed knowledge from CLI flags
    for entry in args.knowledge:
        if "=" in entry:
            doc_id, text = entry.split("=", 1)
            backend.add_knowledge(doc_id.strip(), text.strip())

    if args.interactive:
        return run_interactive(verbose=not args.quiet)

    if not args.request:
        parser.print_help()
        return 1

    req      = AgentRequest(message=args.request)
    response = backend.handle(req)
    print_response(response, verbose=not args.quiet)
    return 0 if response.success else 1


if __name__ == "__main__":
    sys.exit(main())
