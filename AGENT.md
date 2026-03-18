# Agent Documentation

## Overview

This agent is a CLI tool that connects to an LLM with function calling capabilities and executes tools to answer questions about the project documentation. It implements an agentic loop that can discover wiki files, read their contents, and provide sourced answers.

## Architecture

### LLM Provider

- **Provider**: Qwen Code API running on a VM via `qwen-code-oai-proxy`
- **Model**: `qwen3-coder-plus`
- **API Compatibility**: OpenAI-compatible chat completions API with function calling

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
uv run agent.py "How do you resolve a merge conflict?"
```

**Input**: A single command-line argument (the question).

**Output**: A single JSON line to stdout:
```json
{
  "answer": "Edit the conflicting file, choose which changes to keep, then stage and commit.",
  "source": "wiki/git-workflow.md#resolving-merge-conflicts",
  "tool_calls": [
    {"tool": "list_files", "args": {"path": "wiki"}, "result": "git-workflow.md\n..."},
    {"tool": "read_file", "args": {"path": "wiki/git-workflow.md"}, "result": "..."}
  ]
}
```

**Exit codes**:
- `0`: Success
- `1`: Error (missing question, missing env vars, LLM error)

## Tools

The agent has two tools that it can call via function calling:

### `list_files`

Lists files and directories at a given path in the project repository.

**Parameters**:
- `path` (string, required): Relative directory path from project root (e.g., `"wiki"`)

**Returns**: Newline-separated list of entries with `/` suffix for directories.

**Example**:
```json
{"tool": "list_files", "args": {"path": "wiki"}, "result": "wiki/git.md\nwiki/linux.md\n"}
```

### `read_file`

Reads a file from the project repository.

**Parameters**:
- `path` (string, required): Relative file path from project root (e.g., `"wiki/git.md"`)

**Returns**: File contents as a string, or an error message.

**Example**:
```json
{"tool": "read_file", "args": {"path": "wiki/git.md"}, "result": "# Git\n\nGit is a distributed..."}
```

### Path Security

Both tools enforce security constraints:

1. **No absolute paths**: Paths must be relative to the project root
2. **No path traversal**: Paths containing `..` are rejected
3. **Within project root**: Resolved paths must be within the project directory
4. **Type checking**: `list_files` requires a directory, `read_file` requires a file

## Agentic Loop

The agent implements an agentic loop with the following flow:

1. **Initialize**: Build messages list with system prompt and user question
2. **Call LLM**: Send messages and tool schemas to the LLM
3. **Check for tool calls**:
   - If **no tool calls**: Parse final answer and return
   - If **tool calls present**: Execute each tool, append results, continue loop
4. **Repeat**: Continue until no tool calls or max 10 tool calls reached

### System Prompt Strategy

The system prompt instructs the LLM to:

1. Use `list_files` to discover what files exist in the wiki directory
2. Use `read_file` to read relevant files
3. Respond with a JSON object containing `answer` and `source` fields
4. Include the source in format `wiki/filename.md#section-anchor`

### Output Fields

- **`answer`** (string): The agent's answer to the user's question
- **`source`** (string): Reference to the wiki section (e.g., `wiki/git-workflow.md#merge-conflicts`)
- **`tool_calls`** (array): All tool calls made during the agentic loop, each with:
  - `tool`: Tool name (`list_files` or `read_file`)
  - `args`: Arguments passed to the tool
  - `result`: Result returned by the tool

## Limitations

- Maximum 10 tool calls per question
- Single-turn conversation (no conversation state between runs)
- No retry logic for LLM errors
- No caching of tool results

## Files

- `agent.py`: Main CLI agent implementation with tools and agentic loop
- `.env.agent.secret`: LLM configuration (not committed to git)
- `AGENT.md`: This documentation file
- `plans/task-1.md`: Task 1 implementation plan
- `plans/task-2.md`: Task 2 implementation plan
