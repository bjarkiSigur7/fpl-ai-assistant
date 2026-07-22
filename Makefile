.PHONY: refresh train predict optimize api web dev test lint

# Full weekly cycle: pull fresh data, re-predict, re-optimize
refresh:
	cd backend && uv run fplai refresh

# Retrain all models from scratch (slow; run after each gameweek or when stale)
train:
	cd backend && uv run fplai train

predict:
	cd backend && uv run fplai predict

optimize:
	cd backend && uv run fplai optimize

# Run the FastAPI backend (http://localhost:8000)
api:
	cd backend && uv run uvicorn fplai.api.app:app --reload --port 8000

# Run the Next.js dashboard (http://localhost:3000)
web:
	cd frontend && npm run dev

# Both, for development
dev:
	$(MAKE) -j2 api web

test:
	cd backend && uv run pytest

lint:
	cd backend && uv run ruff check src tests
	cd frontend && npm run lint
