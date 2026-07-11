"""
Tool Layer: Web Search, Python, SQL, REST API, GitHub, Email, Calendar,
File System, etc. Each tool implements `run(request: str) -> str`.

All tools here are safe, offline stubs (no real network calls, no shell
execution) so the harness runs anywhere. Replace `run()` in each tool with
a real integration (requests, subprocess+sandbox, boto3, etc.) as needed.
"""

from __future__ import annotations

from typing import Dict, Protocol


class Tool(Protocol):
    name: str

    def run(self, request: str) -> str:
        ...


class WebSearchTool:
    name = "web_search"

    def run(self, request: str) -> str:
        return (
            f"[stub web_search] Would search the web for: '{request}'. "
            "Wire this up to a real search API (e.g. Bing, Serper, Tavily) "
            "to get live results."
        )


class PythonTool:
    name = "python"

    def run(self, request: str) -> str:
        return (
            f"[stub python] Would execute a Python snippet to satisfy: '{request}'. "
            "Wire this up to a sandboxed interpreter (e.g. a Docker-isolated "
            "subprocess) before enabling real code execution."
        )


class SQLTool:
    name = "sql"

    def run(self, request: str) -> str:
        return (
            f"[stub sql] Would run a SQL query against a database to satisfy: "
            f"'{request}'. Wire this up to a real DB connection (psycopg2, "
            "sqlite3, SQLAlchemy, ...)."
        )


class RestApiTool:
    name = "api"

    def run(self, request: str) -> str:
        return f"[stub api] Would call a REST API to satisfy: '{request}'."


class FileSystemTool:
    name = "file_system"

    def run(self, request: str) -> str:
        return f"[stub file_system] Would read/write files to satisfy: '{request}'."


class EmailTool:
    name = "email"

    def run(self, request: str) -> str:
        return f"[stub email] Would send an email to satisfy: '{request}'."


class CalendarTool:
    name = "calendar"

    def run(self, request: str) -> str:
        return f"[stub calendar] Would schedule an event to satisfy: '{request}'."


class GitHubTool:
    name = "github"

    def run(self, request: str) -> str:
        return f"[stub github] Would perform a source-code operation to satisfy: '{request}'."


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
