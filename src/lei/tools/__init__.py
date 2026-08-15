"""
Tool Layer: Web Search, Python, SQL, REST API, GitHub, Email, Calendar,
File System stubs. Each tool implements run(request: str) -> str.

All tools are safe offline stubs — no real network calls or shell execution.
Replace run() in each with a real integration as needed.
"""

from __future__ import annotations

from typing import Dict, Protocol


class Tool(Protocol):
    name: str

    def run(self, request: str) -> str: ...


class WebSearchTool:
    name = "web_search"

    def run(self, request: str) -> str:
        return (
            f"[stub web_search] Would search the web for: '{request}'. "
            "Wire up to a real search API (Bing, Serper, Tavily, etc.)."
        )


class PythonTool:
    name = "python"

    def run(self, request: str) -> str:
        return (
            f"[stub python] Would execute a Python snippet for: '{request}'. "
            "Wire up to a sandboxed interpreter before enabling real execution."
        )


class SQLTool:
    name = "sql"

    def run(self, request: str) -> str:
        return (
            f"[stub sql] Would run a SQL query for: '{request}'. "
            "Wire up to a real DB connection (psycopg2, sqlite3, SQLAlchemy)."
        )


class RestApiTool:
    name = "api"

    def run(self, request: str) -> str:
        return f"[stub api] Would call a REST API for: '{request}'."


class FileSystemTool:
    name = "file_system"

    def run(self, request: str) -> str:
        return f"[stub file_system] Would read/write files for: '{request}'."


class EmailTool:
    name = "email"

    def run(self, request: str) -> str:
        return f"[stub email] Would send an email for: '{request}'."


class CalendarTool:
    name = "calendar"

    def run(self, request: str) -> str:
        return f"[stub calendar] Would schedule an event for: '{request}'."


class GitHubTool:
    name = "github"

    def run(self, request: str) -> str:
        return f"[stub github] Would perform a source-code operation for: '{request}'."


def get_tool_registry() -> Dict[str, Tool]:
    tools = [
        WebSearchTool(),
        PythonTool(),
        SQLTool(),
        RestApiTool(),
        FileSystemTool(),
        EmailTool(),
        CalendarTool(),
        GitHubTool(),
    ]
    return {tool.name: tool for tool in tools}
