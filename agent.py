#!/usr/bin/env python3
"""CLI agent that calls an LLM and returns a structured JSON answer."""

import json
import os
import sys
from pathlib import Path

import httpx


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


def call_llm(question: str, api_key: str, api_base: str, model: str) -> str:
    """Call the LLM and return the answer text."""
    url = f"{api_base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer the question concisely. Do not use tools. Respond in plain text.",
            },
            {"role": "user", "content": question},
        ],
    }

    with httpx.Client(timeout=40.0) as client:
        response = client.post(url, headers=headers, json=body)
        response.raise_for_status()
        data = response.json()

    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip() if content else "No answer received."


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
        answer = call_llm(question, api_key, api_base, model)
        result: dict[str, str | list[object]] = {"answer": answer, "tool_calls": []}
        print(json.dumps(result, ensure_ascii=False))
        sys.exit(0)
    except httpx.HTTPStatusError as e:
        error_msg = f"HTTP error: {e.response.status_code}"
        print(
            json.dumps(
                {"answer": "", "tool_calls": [], "error": error_msg}, ensure_ascii=False
            )
        )
        print(f"Debug: {error_msg}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        error_msg = f"Request error: {e.__class__.__name__}"
        print(
            json.dumps(
                {"answer": "", "tool_calls": [], "error": error_msg}, ensure_ascii=False
            )
        )
        print(f"Debug: {error_msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        error_msg = f"Unexpected error: {e.__class__.__name__}"
        print(
            json.dumps(
                {"answer": "", "tool_calls": [], "error": error_msg}, ensure_ascii=False
            )
        )
        print(f"Debug: {error_msg}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
