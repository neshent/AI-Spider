"""
Lei — Reasoning Agent
=====================
Wires together Input Processing, Reasoning Engine, Planning, Reactive,
Tool-Using, RAG, Multi-Agent, Workflow, and Memory into one pipeline.

Quick start (offline, no API key needed):
    from lei import Orchestrator
    trace = Orchestrator().handle("Build a weather app.")
    print(trace.final_response)
"""

from .orchestrator import Orchestrator

__all__ = ["Orchestrator"]
__version__ = "0.1.0"
