"""Regression tests for the agent CLI."""

import json
import subprocess
import sys
from pathlib import Path


def test_agent_returns_valid_json_structure() -> None:
    """Test that agent.py returns valid JSON with required fields."""
    project_root = Path(__file__).parent.parent
    agent_path = project_root / "agent.py"

    result = subprocess.run(
        [sys.executable, "-m", "uv", "run", str(agent_path), "What is API?"],
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root,
    )

    assert result.returncode == 0, f"Agent failed with stderr: {result.stderr}"

    response = json.loads(result.stdout)

    assert "answer" in response, "Response missing 'answer' field"
    assert isinstance(response["answer"], str), "'answer' must be a string"

    assert "tool_calls" in response, "Response missing 'tool_calls' field"
    assert isinstance(response["tool_calls"], list), "'tool_calls' must be a list"
