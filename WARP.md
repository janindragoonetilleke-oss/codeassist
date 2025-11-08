# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

- Overview
  CodeAssist is a multi-service, containerized system for a local, editor-native coding assistant. The stack includes:
  - Web UI (Next.js, TypeScript) for recording/editing and driving requests
  - State Service (FastAPI, Python) for action selection, prompt building (FIM), LLM calls, and episode storage + background test execution
  - Solution Tester (FastAPI, Python) for running problem test cases against submitted code
  - Policy Models API (FastAPI, Python) that loads trained checkpoints and returns (action, line) decisions
  - Ollama LLM container providing completions
  Persistent data is under persistent-data/, including episodes and model checkpoints. The full workflow is orchestrated by run.py (Docker SDK) or compose.yml (Docker Compose), and training is coordinated by training_loop.py.

- Ports and services
  - web UI: 3000 (prod containers), 3001 (local dev), 3002 (simulation dev), 3003 (zero-style UI)
  - state-service: 8000
  - solution-tester: 8008
  - policy-models: 8001
  - ollama: 11434
  Note: root README reserves 8000, 8080, 8001, 8008, 3003, 11434. If 3000 is taken, you can change run.py with -p/--port.

- Run the full system (recommended)
  Requires Docker and uv.
  - Start everything with Docker SDK orchestration:
    uv run run.py          # opens the Web UI at http://localhost:3000 by default
    # Options: --no-train, --train-only, --no-pull, --no-telemetry, -p <port>, etc.
  - Alternatively, using Docker Compose from repo root:
    docker compose build
    docker compose up -d
    # Open http://localhost:3000 (or simulation UI at :3002, zero-style UI at :3003)
    # Stop: docker compose down

- Service-by-service development
  Web UI (web-ui/)
  - Install deps: npm ci
  - Dev servers:
    # standard dev (defaults to :3001)
    NEXT_PUBLIC_TESTER_URL=http://localhost:8008 \
    NEXT_PUBLIC_STATE_SERVICE_URL=http://localhost:8000 \
    NEXT_PUBLIC_POLICY_MODELS_URL=http://localhost:8001 \
    npm run dev

    # simulation mode (:3002)
    NEXT_PUBLIC_TESTER_URL=http://localhost:8008 \
    NEXT_PUBLIC_STATE_SERVICE_URL=http://localhost:8000 \
    NEXT_PUBLIC_POLICY_MODELS_URL=http://localhost:8001 \
    npm run dev:simulation

    # zero-style mode (:3003)
    NEXT_PUBLIC_TESTER_URL=http://localhost:8008 \
    NEXT_PUBLIC_STATE_SERVICE_URL=http://localhost:8000 \
    NEXT_PUBLIC_POLICY_MODELS_URL=http://localhost:8001 \
    npm run dev:zero-style
  - Build: npm run build
  - Lint/format: npm run lint, npm run lint:fix, npm run type-check
  - Tests:
    npm test                          # run all Vitest tests
    npm test -- src/path/to/file.test.ts   # single file
    npm test -- -t "test name substring"  # single spec by name

  State Service (state-service/)
  - Run locally (reload from DEBUG):
    uv run python main.py
    # or: uv run uvicorn src.api.server:app --reload --port 8000
  - Key env (see src/config.py): OLLAMA_BASE_URL, OLLAMA_MODEL, SOLUTION_TESTER_BASE_URL, POLICY_MODEL_BASE_URL, PERSISTENT_DATA_DIR, REQUEST_TIMEOUT, TEST_WORKER_CONCURRENCY, MAX_TEST_CASES
  - Smoke check: curl http://localhost:8000/health

  Solution Tester (solution-tester/)
  - Run locally:
    uv run python main.py   # serves on :8008
  - Smoke check: curl http://localhost:8008/health

  Policy Models API (policy_models/)
  - Run API locally:
    uv run python api_server.py   # serves on :8001
  - Model files are expected under persistent-data/trainer/models (paths configurable via env). Health: curl http://localhost:8001/health
  - Verification (unit/e2e sanity for policy code):
    uv run python -m verify.run_all --device cpu

- Training
  - From the main launcher, press Ctrl+C in the run.py terminal to trigger the training phase, or run training-only:
    uv run run.py --train-only
  - The training loop uses training_config.json. To run it directly:
    uv run python training_loop.py    # reads training_config.json

- Persistent data structure (created by run.py)
  - persistent-data/state-service/episodes: episode raw logs and snapshots
  - persistent-data/state-service/simulated-episodes and shallow-zero-style-episodes
  - persistent-data/trainer/models: checkpoints (asm_assistant_model.pt, asm_human_model.pt, asm_featurizer.pt)

- High-level architecture
  - Web UI issues state updates and inference requests to State Service. UI sets NEXT_PUBLIC_* URLs for services.
  - State Service flow (src/api/server.py):
    1) Validate request. If action provided, use it; else query Policy Models API (/infer or /infer_human) to get (action, target_line).
    2) Build FIM prompt via TextPreprocessor and call Ollama asynchronously (ollama library) with streaming.
    3) Post-process stream to produce editor-appliable unified diff; compute per-line assistant attribution.
    4) Episode persistence: append states to JSONL under persistent-data; /episodes/end enqueues background tests.
    5) Background worker reads episode states, injects a stdin harness using dataset entry_point, calls Solution Tester, and writes test summaries back into episode snapshots (env.compiled/tests/execution_time_ms). Test queue status exposed at /test-queue/status.
  - Solution Tester executes code against problem I/O cases and returns pass/fail per test.
  - Policy Models API loads checkpoints and featurizer on startup, serves /infer and /infer_human with strategy/top-k/temperature/epsilon options. Returns action_idx and 1-based line_idx.
  - Training loop (training_loop.py) runs BC/PPO from episodes, then launches zero-style UI recordings, waits for test queue drain, and runs a follow-up PPO phase. Models are persisted under persistent-data/trainer/models and consumed by the Policy Models API.

- Lint/format conventions
  - Python: ruff format . && ruff check .  # repository expects ruff formatting (see README)
  - Web UI: eslint and prettier via package.json scripts

This file intentionally omits generic engineering advice and redundant file listings.
