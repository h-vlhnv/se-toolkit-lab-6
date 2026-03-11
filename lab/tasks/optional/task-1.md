# Advanced Agent Features

Extend your agent with advanced capabilities that improve reliability or expand what it can answer.

## [Git workflow](../../../wiki/git-workflow.md)

1. Create an issue titled `[Task] Advanced Agent Features`.
2. Pull latest `main` from `origin` and `upstream`.
3. Create a branch from `main` (e.g., `task/advanced-agent-features`).
4. Work on the branch. Commit as you go using [conventional commits](https://www.conventionalcommits.org/) (e.g., `feat:`, `docs:`, `test:`).
5. Push, create a PR to `main` in **your fork** (not upstream). Link the issue using a keyword (e.g., `Closes #4`).
6. Get a review from your partner, merge (this closes the issue automatically), delete the branch.

## What you will do

Choose **one or more** extensions to implement. Each extension should measurably improve your agent — either by increasing the pass rate, reducing latency, or handling failure modes.

### Extension options

#### Retry logic with backoff

LLM APIs have rate limits. Implement automatic retry with exponential backoff when the API returns 429 (Too Many Requests) or 5xx errors. This makes your agent reliable under load.

#### Caching layer

Avoid re-calling tools for the same arguments within a single conversation. Cache tool results in memory so that if the LLM asks to `read_file("backend/app/main.py")` twice, the second call returns instantly.

#### Direct database tool (`query_db`)

Add a `query_db` tool that runs **read-only** SQL queries against the PostgreSQL database. This lets the agent answer data questions without going through the API. Use a read-only database connection to prevent accidental writes.

#### Multi-step reasoning

Before executing tools, have the agent output a plan (what it needs to find out and which tools to use). Then execute the plan step by step. This improves accuracy on complex questions that require multiple tool calls.

## Deliverables

### 1. Plan (`plans/optional-1.md`)

Document which extension(s) you chose and why. Describe the expected improvement.

Commit:

```text
docs: add advanced features plan
```

### 2. Implementation (update `agent.py`)

Implement your chosen extension(s).

Commit:

```text
feat: add <extension name> to agent
```

### 3. Tests

Write tests that demonstrate the extension works correctly.

Commit:

```text
test: add tests for <extension name>
```

### 4. Documentation (update `AGENT.md`)

Update `AGENT.md` to describe the extension(s) you implemented.

Commit:

```text
docs: document advanced agent features
```

## Acceptance criteria

- [ ] Issue has the correct title.
- [ ] At least one extension is implemented and working.
- [ ] Tests demonstrate the extension works correctly.
- [ ] `AGENT.md` is updated to describe the extension.
- [ ] PR is approved and merged.
- [ ] Issue is closed by the PR.
