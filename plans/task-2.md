# Task 2 Plan – The Documentation Agent

## Goal

Extend `agent.py` from Task 1 into a simple “documentation agent” that can:

- Call an OpenAI-compatible Chat Completions API **with function calling** enabled.
- Execute tool calls (`list_files`, `read_file`) against the local repository.
- Loop until the model returns a final answer (or we hit a max tool-call budget).
- Print **one JSON object** to `stdout` with required fields:
  - `answer` (string)
  - `source` (string; `wiki/<file>.md#<section-anchor>`)
  - `tool_calls` (array of `{tool, args, result}`)

All diagnostics remain on `stderr`.

## Tool schemas (function calling)

Define two tool schemas and include them in every LLM request:

- **`list_files`**
  - Params: `{ "path": "wiki" | "<relative-dir>" }`
  - Returns: newline-separated directory entries or an error string.
- **`read_file`**
  - Params: `{ "path": "<relative-file>" }`
  - Returns: file contents or an error string.

Implementation will follow the OpenAI “tools” format:

- Request field `tools: [{type:"function", function:{name, description, parameters}}]`
- Allow model selection with `tool_choice: "auto"`

## Agentic loop

Implement an agentic loop that:

- Builds a `messages` list:
  - `system`: instruct the model to:
    - Prefer `list_files` to discover wiki files.
    - Use `read_file` to retrieve relevant content.
    - Produce a final response as a JSON object containing `answer` and `source`.
  - `user`: the CLI question.
- Calls the LLM.
- If the assistant message contains `tool_calls`:
  - For each tool call:
    - Execute the requested tool.
    - Append an entry to our *output* `tool_calls` array with `{tool, args, result}`.
    - Append a `tool` role message back to `messages` with the tool result (including `tool_call_id`).
  - Continue looping, counting tool calls, until:
    - The model returns **no tool calls**, or
    - We reach **10 tool calls** total.
- If the assistant message contains final `content`:
  - Parse `content` as JSON to extract `answer` and `source`.
  - If parsing fails, fall back to:
    - `answer = content.strip()`
    - `source = ""` (still required as a string)

## Path security

Both tools must prevent access outside the repository root:

- Reject absolute paths.
- Reject paths containing `..` segments.
- Resolve the candidate path and verify it is within project root (e.g. `resolved.is_relative_to(project_root)`).
- Return an error string (not exceptions) for invalid paths, missing paths, or wrong types (file vs directory).

## Code structure changes in `agent.py`

- Add:
  - Tool schema builders.
  - Tool implementations (`_tool_list_files`, `_tool_read_file`) with path guards.
  - A loop function (e.g. `_run_agent(question)`) returning `{answer, source, tool_calls}`.
- Update the existing LLM call helper to accept:
  - `messages`
  - `tools`
  - (optionally) `tool_choice`
- Keep existing env loading and error handling style from Task 1.

## Documentation updates (`AGENT.md`)

Update `AGENT.md` to describe:

- The new tools, their parameters/returns, and security constraints.
- The agentic loop (max 10 tool calls).
- The system prompt strategy (discover via `list_files`, read via `read_file`, include `source` anchor).
- The updated CLI output shape with `answer`, `source`, `tool_calls`.

## Tests (2 new regression tests)

Add 2 tests that **do not depend on a real LLM** by running a tiny local HTTP stub server that mimics the Chat Completions API:

- **Merge conflict question**
  - Stub returns tool calls: `list_files wiki`, then `read_file wiki/git.md`, then final JSON content with `source: wiki/git.md#merge-conflict`.
  - Assert:
    - Output JSON has `answer`, `source`, `tool_calls`.
    - `tool_calls` contains `list_files` and `read_file`.
    - `source` contains `wiki/git.md#merge-conflict`.
- **Wiki listing question**
  - Stub returns one tool call: `list_files wiki`, then final JSON.
  - Assert:
    - `tool_calls` contains `list_files`.

Tests will set `LLM_API_BASE` to the stub server URL, `LLM_API_KEY`/`LLM_MODEL` to dummy values, and run `agent.py` as a subprocess (matching existing CLI tests).
