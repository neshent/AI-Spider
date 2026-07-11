#!/usr/bin/env python3
"""
Command-line harness for the Reasoning Agent architecture.

Usage:
    python cli.py "Build a weather application."
    python cli.py "What's today's weather?"
    python cli.py --interactive
"""

from __future__ import annotations

import argparse
import sys

from agent_harness import Orchestrator


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
            print(f"workflow attempts: {trace.workflow_result.attempts}, success: {trace.workflow_result.success}")
        print("---")

    print(f"\n{trace.final_response}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Reasoning Agent harness.")
    parser.add_argument("request", nargs="?", help="The user request / goal to process.")
    parser.add_argument("--interactive", action="store_true", help="Start an interactive REPL.")
    parser.add_argument("--quiet", action="store_true", help="Suppress pipeline trace output.")
    parser.add_argument(
        "--knowledge",
        action="append",
        default=[],
        help="path=text pairs to seed the RAG knowledge base, e.g. --knowledge doc1='FastAPI is a Python web framework.'",
    )
    args = parser.parse_args()

    orchestrator = Orchestrator()

    for entry in args.knowledge:
        if "=" in entry:
            doc_id, text = entry.split("=", 1)
            orchestrator.add_knowledge(doc_id, text)

    # Seed a little default knowledge so RAG has something to retrieve.
    orchestrator.add_knowledge(
        "stack_notes",
        "FastAPI is a modern, fast Python web framework for building APIs. "
        "SQLite and PostgreSQL are common relational databases for small "
        "and large deployments respectively.",
    )

    if args.interactive:
        print("Reasoning Agent Harness — interactive mode. Ctrl+C or 'exit' to quit.\n")
        try:
            while True:
                request = input("> ").strip()
                if request.lower() in ("exit", "quit"):
                    break
                if not request:
                    continue
                trace = orchestrator.handle(request)
                print_trace(trace, verbose=not args.quiet)
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
        return 0

    if not args.request:
        parser.print_help()
        return 1

    trace = orchestrator.handle(args.request)
    print_trace(trace, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    sys.exit(main())
