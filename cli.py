#!/usr/bin/env python3
"""
Command-line interface for the Lei Reasoning Agent.

Usage:
    python cli.py "Build a weather application."
    python cli.py "What's today's weather?"
    python cli.py --interactive
    python cli.py --knowledge doc1="FastAPI is a Python web framework." "Build an API."
"""

from __future__ import annotations

import argparse
import sys

from src.lei import Orchestrator


def print_trace(trace, verbose: bool) -> None:
    if verbose:
        print(f"\n--- intent: {trace.intent} ---")
        if trace.handled_reactively:
            print("[handled reactively]")
        if trace.plan:
            print("plan:")
            for step in trace.plan:
                print(f"  {step.index}. {step.description}")
        if trace.retrieved_context:
            print(f"retrieved {len(trace.retrieved_context)} context chunk(s)")
        if trace.tool_used:
            print(f"tool used: {trace.tool_used}\n  -> {trace.tool_output}")
        if trace.multi_agent_report:
            print("multi-agent report:")
            print(trace.multi_agent_report.summary())
        if trace.workflow_result:
            wr = trace.workflow_result
            print(f"workflow attempts: {wr.attempts}, success: {wr.success}")
        print("---")
    print(f"\n{trace.final_response}\n")


def build_orchestrator(knowledge_args: list) -> Orchestrator:
    orch = Orchestrator()
    # Default knowledge seed
    orch.add_knowledge(
        "stack_notes",
        "FastAPI is a modern, fast Python web framework for building APIs. "
        "SQLite and PostgreSQL are common relational databases.",
    )
    for entry in knowledge_args:
        if "=" in entry:
            doc_id, text = entry.split("=", 1)
            orch.add_knowledge(doc_id.strip(), text.strip())
    return orch


def run_interactive(orch: Orchestrator, verbose: bool) -> int:
    print("Lei Reasoning Agent — interactive mode. Ctrl+C or 'exit' to quit.\n")
    try:
        while True:
            request = input("> ").strip()
            if request.lower() in ("exit", "quit"):
                break
            if not request:
                continue
            trace = orch.handle(request)
            print_trace(trace, verbose=verbose)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Lei Reasoning Agent CLI")
    parser.add_argument("request", nargs="?", help="The request to process.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive REPL.")
    parser.add_argument("--quiet", action="store_true", help="Suppress pipeline trace output.")
    parser.add_argument(
        "--knowledge", action="append", default=[],
        help="Seed the RAG knowledge base: --knowledge doc1='some text'",
    )
    args = parser.parse_args()

    orch = build_orchestrator(args.knowledge)

    if args.interactive:
        return run_interactive(orch, verbose=not args.quiet)

    if not args.request:
        parser.print_help()
        return 1

    trace = orch.handle(args.request)
    print_trace(trace, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
