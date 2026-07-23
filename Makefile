# fpl-ai-assistant — repo-root entry points. All backend commands run via uv from
# backend/; frontend via npm from frontend/. Pre-launch (2026-27 not live) the
# predict/optimize demo window defaults to 2025-26 GW34 — override with SEASON/GW/GWS,
# e.g. `make demo GW=30` or `make backtest GWS=30..38`.

.PHONY: refresh train predict optimize simulate demo backtest api web dev test test-live lint

SEASON ?= 2025
GW ?= 34
GWS ?=

# Full weekly cycle: snapshot -> pulls -> build -> predict -> optimize (exit-0 data-safe)
refresh:
	cd backend && uv run fplai refresh

# Retrain all models from scratch (~70 s; run after each gameweek or when stale)
train:
	cd backend && uv run fplai train

# Live mode (no args). Between seasons this degrades to a message — use `make demo`.
predict:
	cd backend && uv run fplai predict

optimize:
	cd backend && uv run fplai optimize

# Monte Carlo chip-timing simulation over the full chip window (needs
# `fplai predict --through-gw 19` coverage; refresh handles both automatically)
simulate:
	cd backend && uv run fplai simulate

# Pre-launch demo chain: predict + optimize the SEASON/GW backtest window
demo:
	cd backend && uv run fplai predict --season $(SEASON) --gw $(GW)
	cd backend && uv run fplai optimize --season $(SEASON) --gw $(GW)

# Walk-forward policy backtest (stage 4 harness; GWS narrows the window, e.g. GWS=30..38)
backtest:
	cd backend && uv run fplai backtest --season $(SEASON) $(if $(GWS),--gws $(GWS))

# Run the FastAPI backend (http://localhost:8000)
api:
	cd backend && uv run uvicorn fplai.api.app:app --reload --port 8000

# Run the Next.js dashboard (http://localhost:3000; expects the API on :8000)
web:
	cd frontend && npm run dev

# Both, for development (parallel; Ctrl-C stops both)
dev:
	$(MAKE) -j2 api web

# Offline test suite (live/network tests excluded; run those with `make test-live`)
test:
	cd backend && uv run pytest -q -m "not live"

test-live:
	cd backend && uv run pytest -q -m live

lint:
	cd backend && uv run ruff check src tests
	cd frontend && npm run lint
