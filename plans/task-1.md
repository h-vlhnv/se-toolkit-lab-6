# Task 1 Plan – Call an LLM from Code

## LLM provider and configuration

- **Provider**: Qwen Code API running on the VM via `qwen-code-oai-proxy`.
- **Model**: `qwen3-coder-plus` (OpenAI-compatible chat completions).
- **Endpoint**: `LLM_API_BASE` read from `.env.agent.secret`, e.g. `http://<vm-ip>:42005/v1`.
- **Auth**: `LLM_API_KEY` read from `.env.agent.secret` (value `gochamp` set outside the code).
- **Config loading**: Manually parse `.env.agent.secret` in `agent.py` as a simple `KEY=VALUE`
  file and populate missing `LLM_API_BASE`, `LLM_API_KEY`, and `LLM_MODEL` in `os.environ`
  without overwriting existing environment variables.

## CLI behavior

- **Invocation**: `uv run agent.py "What does REST stand for?"`
- **Arguments**:
  - First positional argument is the user question (required).
  - If missing, print a helpful error message to `stderr` and exit with code `1`.
- **Output**:
  - Print a single line of valid JSON to `stdout` with shape:
    - `{"answer": <string>, "tool_calls": []}`
  - `tool_calls` is always an empty list in Task 1.
  - No extra text before/after the JSON line.

## LLM request/response flow

- **System prompt**:
  - Short, minimal instructions: answer the question concisely, tools disabled, respond in plain text.
- **HTTP request** (using `httpx`):
  - POST to `${LLM_API_BASE}/chat/completions`.
  - Headers: `Authorization: Bearer ${LLM_API_KEY}`, `Content-Type: application/json`.
  - Body:
    - `model`: `${LLM_MODEL}` from env.
    - `messages`: one system message + one user message (the CLI question).
  - Use a client-level timeout around 40 seconds to stay under 60s overall.
- **Response handling**:
  - Parse JSON.
  - Extract `answer` from `choices[0].message.content` (fall back to a sane message if empty).
  - Never include tool calls in this task; always set `"tool_calls": []` in the final JSON.

## Error handling and exit codes

- **Validation errors** (e.g. missing question, missing env vars):
  - Write a clear explanation to `stderr`.
  - Exit with non-zero status (e.g. `sys.exit(1)`).
- **LLM/HTTP errors** (network error, non-2xx response, malformed JSON):
  - Print a structured JSON error to `stdout`:
    - `{"answer": "", "tool_calls": [], "error": "<short description>"}`
  - Also log details to `stderr` for debugging.
  - Exit with non-zero status (per your choice A).
- **Success**:
  - Exit code `0`.

## `agent.py` structure

- `main()` function that:
  - Parses the command-line argument (the question).
  - Loads env variables (`LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL`) from `.env.agent.secret` via environment.
  - Builds and sends the `requests.post` call.
  - Extracts the answer text.
  - Constructs the result dict: `{"answer": answer_text, "tool_calls": []}`.
  - Prints `json.dumps(result, ensure_ascii=False)` to `stdout`.
- `if __name__ == "__main__":` guard to call `main()`.

## Documentation (`AGENT.md`)

- Describe:
  - Which provider and model you use (Qwen Code API, `qwen3-coder-plus`).
  - How configuration is read from `.env.agent.secret`.
  - The CLI interface (how to run, expected input/output format).
  - Limitations in Task 1 (no tools, single-turn, no conversation state).

## Tests

- Add one regression test script (e.g. under `tests/`):
  - Uses `subprocess.run(["uv", "run", "agent.py", "Test question"], capture_output=True, text=True, check=True)`.
  - Parses `stdout` with `json.loads`.
  - Asserts:
    - `answer` key exists and is a string.
    - `tool_calls` key exists and is a list.
  - Ensures the process exits with code `0` for a normal question.