"""Central configuration and paths."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
CACHE_DIR = DATA_DIR / "cache"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=REPO_ROOT / ".env", env_prefix="FPLAI_", extra="ignore")

    # Optional API keys / external services
    odds_api_key: str = ""
    gemini_api_key: str = ""

    # The user's FPL entry (team) id, once they have one. 0 = not set.
    entry_id: int = 0

    # Planning
    horizon_gws: int = 8
    ev_decay: float = 0.87


settings = Settings()

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
