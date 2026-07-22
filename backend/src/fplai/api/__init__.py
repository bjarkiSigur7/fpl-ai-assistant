"""FastAPI layer (stage 5): serves predictions, plans and recommendations under /api.

Entry point: ``uv run uvicorn fplai.api.app:app --port 8000`` (see
:mod:`fplai.api.app`).  Submodules: :mod:`~fplai.api.app` (routes),
:mod:`~fplai.api.schemas` (response models re-using the optimizer contracts),
:mod:`~fplai.api.cache` (mtime-aware artifact loaders) and :mod:`~fplai.api.jobs`
(single-flight background jobs for the refresh pipeline and my-team solves).
"""
