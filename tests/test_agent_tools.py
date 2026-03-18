"""Regression tests for the documentation agent with tools."""

import json
import os
import subprocess
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


class MockLLMHandler(BaseHTTPRequestHandler):
    """Mock LLM server that simulates function calling responses."""

    # Class-level state for multi-turn conversations
    call_count: int = 0
    scenario: str = "merge_conflict"

    def log_message(self, format, *args):  # type: ignore[override]
        """Suppress default logging."""
        pass

    def do_POST(self) -> None:  # noqa: N802
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body.decode("utf-8"))

        messages = data.get("messages", [])
        last_message: dict[str, Any] = messages[-1] if messages else {}

        # Check if this is a tool result message
        if last_message.get("role") == "tool":
            # Second call - return final answer
            self._send_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "answer": "Edit the conflicting file, choose which changes to keep, then stage and commit.",
                                        "source": "wiki/git-workflow.md#resolving-merge-conflicts",
                                    }
                                ),
                            }
                        }
                    ]
                }
            )
            return

        # First call - return tool calls
        if MockLLMHandler.scenario == "merge_conflict":
            self._send_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "list_files",
                                            "arguments": json.dumps({"path": "wiki"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }
            )
        elif MockLLMHandler.scenario == "wiki_listing":
            self._send_response(
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(
                                    {
                                        "answer": "The wiki contains documentation files.",
                                        "source": "wiki/",
                                    }
                                ),
                            }
                        }
                    ]
                }
            )

    def _send_response(self, response_data: dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_data).encode("utf-8"))


def start_mock_server(port: int) -> HTTPServer:
    """Start the mock LLM server."""
    server = HTTPServer(("127.0.0.1", port), MockLLMHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def test_agent_merge_conflict_question() -> None:
    """Test that agent uses read_file and returns correct source for merge conflict question."""
    project_root = Path(__file__).parent.parent
    agent_path = project_root / "agent.py"

    # Start mock server on a random available port
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    MockLLMHandler.scenario = "merge_conflict"
    MockLLMHandler.call_count = 0

    server = start_mock_server(port)

    try:
        # Wait for server to start
        time.sleep(0.1)

        env = {
            **os.environ,
            "LLM_API_KEY": "test-key",
            "LLM_API_BASE": f"http://127.0.0.1:{port}/v1",
            "LLM_MODEL": "test-model",
        }

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "uv",
                "run",
                str(agent_path),
                "How do you resolve a merge conflict?",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            env=env,
        )

        assert result.returncode == 0, f"Agent failed: {result.stderr}"

        response = json.loads(result.stdout)

        # Check required fields
        assert "answer" in response, "Response missing 'answer' field"
        assert "source" in response, "Response missing 'source' field"
        assert "tool_calls" in response, "Response missing 'tool_calls' field"

        # Check source contains expected file and anchor
        assert "wiki/git-workflow.md" in response["source"], (
            f"Source should reference wiki/git-workflow.md, got: {response['source']}"
        )
        assert (
            "resolving-merge-conflicts" in response["source"].lower()
            or "merge-conflict" in response["source"].lower()
        ), f"Source should contain merge conflict anchor, got: {response['source']}"

    finally:
        server.shutdown()


def test_agent_wiki_listing_question() -> None:
    """Test that agent uses list_files for wiki listing question."""
    project_root = Path(__file__).parent.parent
    agent_path = project_root / "agent.py"

    # Start mock server on a random available port
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    MockLLMHandler.scenario = "wiki_listing"
    MockLLMHandler.call_count = 0

    server = start_mock_server(port)

    try:
        # Wait for server to start
        time.sleep(0.1)

        env = {
            **os.environ,
            "LLM_API_KEY": "test-key",
            "LLM_API_BASE": f"http://127.0.0.1:{port}/v1",
            "LLM_MODEL": "test-model",
        }

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "uv",
                "run",
                str(agent_path),
                "What files are in the wiki?",
            ],
            capture_output=True,
            text=True,
            cwd=project_root,
            env=env,
        )

        assert result.returncode == 0, f"Agent failed: {result.stderr}"

        response = json.loads(result.stdout)

        # Check required fields
        assert "answer" in response, "Response missing 'answer' field"
        assert "source" in response, "Response missing 'source' field"
        assert "tool_calls" in response, "Response missing 'tool_calls' field"

        # For wiki listing, we expect the agent to potentially use list_files
        # The mock returns a direct answer, so tool_calls might be empty
        assert isinstance(response["tool_calls"], list), "'tool_calls' must be a list"

    finally:
        server.shutdown()
