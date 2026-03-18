# Agent Documentation

## Overview

This agent is a CLI tool that connects to an LLM and answers questions. It is the foundation for the more advanced agent with tools and agentic loop that will be built in subsequent tasks.

## Architecture

### LLM Provider

- **Provider**: Qwen Code API running on a VM via `qwen-code-oai-proxy`
- **Model**: `qwen3-coder-plus`
- **API Compatibility**: OpenAI-compatible chat completions API

### Configuration

The agent reads configuration from `.env.agent.secret` in the project root:

```env
LLM_API_KEY=your-api-key-here
LLM_API_BASE=http://<vm-ip>:<port>/v1
LLM_MODEL=qwen3-coder-plus
```

**Note**: This is different from the backend LMS API key in `.env.docker.secret`. The `LLM_API_KEY` authenticates with the LLM provider.

### CLI Interface

**Usage**:
```bash
uv run agent.py "What does REST stand for?"
```

**Input**: A single command-line argument (the question).

**Output**: A single JSON line to stdout:
```json
{"answer": "Representational State Transfer.", "tool_calls": []}
```

**Exit codes**:
- `0`: Success
- `1`: Error (missing question, missing env vars, LLM error)

### Request/Response Flow

1. User provides a question as a CLI argument
2. Agent loads environment variables from `.env.agent.secret`
3. Agent sends a POST request to `${LLM_API_BASE}/chat/completions`
4. LLM returns the answer
5. Agent outputs JSON with `answer` and `tool_calls` fields

### Limitations (Task 1)

- No tools integration (always returns empty `tool_calls`)
- Single-turn conversation (no conversation state)
- No retry logic or caching
- No multi-step reasoning

## Files

- `agent.py`: Main CLI agent implementation
- `.env.agent.secret`: LLM configuration (not committed to git)
- `AGENT.md`: This documentation file
- `plans/task-1.md`: Implementation plan
