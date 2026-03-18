#!/usr/bin/env python3
"""CLI agent that calls an LLM with function calling and executes tools."""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import httpx

# Maximum number of tool calls per question
MAX_TOOL_CALLS = 10

# Project root directory
PROJECT_ROOT = Path(__file__).parent


def load_env_file(env_path: Path) -> dict[str, str]:
    """Parse a simple KEY=VALUE env file."""
    env_vars: dict[str, str] = {}
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars


def ensure_env_vars() -> None:
    """Load .env.agent.secret and populate missing LLM_* env vars."""
    env_path = Path(__file__).parent / ".env.agent.secret"
    env_vars = load_env_file(env_path)

    for key in ("LLM_API_KEY", "LLM_API_BASE", "LLM_MODEL"):
        if key not in os.environ and key in env_vars:
            os.environ[key] = env_vars[key]


def validate_env() -> tuple[str, str, str]:
    """Validate required env vars are present."""
    api_key = os.environ.get("LLM_API_KEY")
    api_base = os.environ.get("LLM_API_BASE")
    model = os.environ.get("LLM_MODEL")

    if not api_key:
        print("Error: LLM_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    if not api_base:
        print("Error: LLM_API_BASE not set", file=sys.stderr)
        sys.exit(1)
    if not model:
        print("Error: LLM_MODEL not set", file=sys.stderr)
        sys.exit(1)

    return api_key, api_base, model


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return the tool schemas for function calling."""
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files and directories at a given path in the project repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative directory path from project root (e.g., 'wiki').",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the project repository.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path from project root (e.g., 'wiki/git.md').",
                        }
                    },
                    "required": ["path"],
                },
            },
        },
    ]


def validate_path(path_str: str) -> tuple[bool, str, Path]:
    """
    Validate a path string for security.

    Returns:
        (is_valid, error_message, resolved_path)
    """
    # Reject absolute paths
    if os.path.isabs(path_str):
        return False, "Absolute paths are not allowed", Path()

    # Reject paths with .. segments
    if ".." in path_str.split(os.sep):
        return False, "Path traversal with '..' is not allowed", Path()

    # Resolve the path relative to project root
    candidate = PROJECT_ROOT / path_str

    # Check if path exists
    if not candidate.exists():
        return False, f"Path does not exist: {path_str}", Path()

    # Resolve to absolute path and verify it's within project root
    try:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
            return False, "Path is outside project directory", Path()
        return True, "", resolved
    except (ValueError, OSError) as e:
        return False, f"Invalid path: {e}", Path()


def tool_list_files(args: dict[str, Any]) -> str:
    """List files and directories at a given path."""
    path_str = args.get("path", "")

    is_valid, error_msg, resolved_path = validate_path(path_str)
    if not is_valid:
        return f"Error: {error_msg}"

    if not resolved_path.is_dir():
        return f"Error: Not a directory: {path_str}"

    entries: list[str] = []
    for entry in resolved_path.iterdir():
        rel_path = entry.relative_to(PROJECT_ROOT)
        suffix = "/" if entry.is_dir() else ""
        entries.append(f"{rel_path}{suffix}")

    return "\n".join(sorted(entries))


def tool_read_file(args: dict[str, Any]) -> str:
    """Read a file from the project repository."""
    path_str = args.get("path", "")

    is_valid, error_msg, resolved_path = validate_path(path_str)
    if not is_valid:
        return f"Error: {error_msg}"

    if not resolved_path.is_file():
        return f"Error: Not a file: {path_str}"

    try:
        with open(resolved_path, encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError) as e:
        return f"Error reading file: {e}"


def execute_tool(tool_name: str, args: dict[str, Any]) -> str:
    """Execute a tool by name and return the result."""
    if tool_name == "list_files":
        return tool_list_files(args)
    elif tool_name == "read_file":
        return tool_read_file(args)
    else:
        return f"Error: Unknown tool: {tool_name}"


def call_llm(
    messages: list[dict[str, Any]],
    api_key: str,
    api_base: str,
    model: str,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Call the LLM and return the response data."""
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"

    with httpx.Client(timeout=40.0) as client:
        response = client.post(url, headers=headers, json=body)
        response.raise_for_status()
        return response.json()


def get_system_prompt() -> str:
    """Return the system prompt for the documentation agent."""
    return """You are a documentation agent that helps users find information in the project wiki.

You have two tools:
1. list_files(path) - List files in a directory
2. read_file(path) - Read a file's contents

Strategy:
1. Use list_files to discover what files exist in the wiki directory
2. Use read_file to read relevant files
3. When you have enough information, respond with a JSON object:
   {
     "answer": "<your answer to the user's question>",
     "source": "<file-path>#<section-anchor>"
   }

Rules:
- Always include the source field with the file path and section anchor
- Use tools to gather information before answering
- Keep answers concise and accurate
- The source should be in format: wiki/filename.md#section-anchor
"""


def run_agent_loop(
    question: str,
    api_key: str,
    api_base: str,
    model: str,
) -> dict[str, Any]:
    """Run the agentic loop and return the final result."""
    tools = get_tool_schemas()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": question},
    ]

    tool_calls_log: list[dict[str, Any]] = []
    tool_call_count = 0

    while tool_call_count < MAX_TOOL_CALLS:
        response_data = call_llm(messages, api_key, api_base, model, tools)

        choice = response_data.get("choices", [{}])[0]
        assistant_message = choice.get("message", {})

        # Check for tool calls
        tool_calls = assistant_message.get("tool_calls", [])

        if not tool_calls:
            # No tool calls - this is the final answer
            content = assistant_message.get("content", "")
            answer, source = parse_final_answer(content)
            return {
                "answer": answer,
                "source": source,
                "tool_calls": tool_calls_log,
            }

        # Process tool calls
        for tool_call in tool_calls:
            if tool_call_count >= MAX_TOOL_CALLS:
                break

            tool_call_id = tool_call.get("id", str(uuid.uuid4()))
            function = tool_call.get("function", {})
            tool_name = function.get("name", "")

            try:
                tool_args: dict[str, Any] = json.loads(function.get("args", "{}"))
            except json.JSONDecodeError:
                tool_args = {}

            # Execute the tool
            result = execute_tool(tool_name, tool_args)

            # Log the tool call
            tool_calls_log.append(
                {
                    "tool": tool_name,
                    "args": tool_args,
                    "result": result,
                }
            )

            # Add tool result as a tool message
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": result,
                }
            )

            tool_call_count += 1

        # Add assistant message to conversation
        messages.append(assistant_message)

    # Max tool calls reached - return what we have
    return {
        "answer": "Maximum tool calls reached. Partial information available.",
        "source": "",
        "tool_calls": tool_calls_log,
    }


def parse_final_answer(content: str) -> tuple[str, str]:
    """Parse the final answer from LLM content."""
    answer: str = content.strip()
    source: str = ""

    # Try to parse as JSON
    try:
        data: dict[str, Any] = json.loads(content)
        answer_val = data.get("answer")
        source_val = data.get("source")
        answer = str(answer_val) if answer_val is not None else content.strip()
        source = str(source_val) if source_val is not None else ""
    except json.JSONDecodeError:
        pass

    return answer, source


def main() -> None:
    """Main entry point."""
    if len(sys.argv) < 2:
        print('Usage: uv run agent.py "<question>"', file=sys.stderr)
        print("Error: No question provided", file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]

    ensure_env_vars()
    api_key, api_base, model = validate_env()

    try:
        result = run_agent_loop(question, api_key, api_base, model)
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error: {e.response.status_code}"
        print(
            json.dumps(
                {"answer": "", "source": "", "tool_calls": [], "error": error_msg},
                ensure_ascii=False,
            )
        )
        print(f"Debug: {error_msg}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        error_msg = f"Request error: {e.__class__.__name__}"
        print(
            json.dumps(
                {"answer": "", "source": "", "tool_calls": [], "error": error_msg},
                ensure_ascii=False,
            )
        )
        print(f"Debug: {error_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Unexpected error: {e.__class__.__name__}"
        print(
            json.dumps(
                {"answer": "", "source": "", "tool_calls": [], "error": error_msg},
                ensure_ascii=False,
            )
        )
        print(f"Debug: {error_msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
