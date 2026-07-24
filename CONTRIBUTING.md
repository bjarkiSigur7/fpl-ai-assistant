# Contributing to PITCHSIDE

Thanks for taking a look. Small, focused PRs are the easiest to review.

## Ground rules

1. **Everything green before you push.** From the repo root:

   ```bash
   make test    # backend offline suite (844 tests) — must pass
   make lint    # ruff (backend) + eslint (frontend) — must be clean
   ```

   Frontend changes must also survive a production build:
   `cd frontend && npm run build`.

2. **Offline-test discipline.** The default test suite runs with **no
   network**. New data-touching code gets fixture-based tests: record one small
   real payload into `backend/tests/fixtures/` and test against that. Tests
   that genuinely need the network are marked `@pytest.mark.live` (excluded by
   default, run via `make test-live`). A PR whose tests flake without Wi-Fi
   will be sent back.

3. **No new dependencies without discussion.** Backend deps live in
   `backend/pyproject.toml` (managed by uv), frontend in
   `frontend/package.json`. Open an issue first if you think one is needed.

4. **No secrets, no data.** Never commit API keys (`.env` is gitignored — keep
   it that way), and never commit anything under `data/` or `site-data/` —
   raw pulls, parquet, model artifacts, and bundles are all regenerable and
   all gitignored. If `git status` shows a parquet file, something is wrong.

5. **Respect the upstream APIs.** All HTTP goes through the shared throttled
   helper in `fplai/data/fpl_api.py`. Do not add request paths that bypass the
   throttle or the on-disk cache.

6. **Conventions**: Python 3.12, type hints everywhere, `ruff` clean at line
   length 100; module ownership and data contracts are documented in
   [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — read the relevant section
   before changing an interface. Numbers quoted in docs must be reproducible
   from the code or cite their source.

## Where to start

- [docs/STATUS.md](docs/STATUS.md) — what works today and the known-gaps list
  (a ready-made source of well-scoped issues).
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — module map and contracts.
- [docs/MODEL_DESIGN_INPUTS.md](docs/MODEL_DESIGN_INPUTS.md) — the model spec,
  benchmarks, and the leakage traps every feature change must respect.
