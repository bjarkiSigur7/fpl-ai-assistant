"""Mtime-aware file loaders for artifacts that refresh under the API's feet.

The refresh pipeline rewrites the processed parquet/JSON artifacts while the API is
serving them, so every read goes through :class:`FileCache`: the parsed value is cached
in memory and transparently reloaded whenever the file's ``(mtime_ns, size)`` signature
changes.  A ``stat()`` per request is the only steady-state cost.

Thread-safe (uvicorn runs sync endpoints in a thread pool).  Missing files raise
``FileNotFoundError`` — endpoint handlers translate that into a 404 with guidance.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["FileCache", "cache"]


@dataclass
class _Entry:
    """One cached file: its stat signature and the parsed value."""

    mtime_ns: int
    size: int
    value: Any


class FileCache:
    """In-memory cache of parsed files, invalidated on mtime/size change."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = threading.Lock()

    def get(self, path: Path, reader: Callable[[Path], Any]) -> Any:
        """Return ``reader(path)``, cached until the file's mtime or size changes.

        Raises:
            FileNotFoundError: when ``path`` does not exist (stale entries are dropped).
        """
        key = str(path)
        try:
            stat = path.stat()
        except FileNotFoundError:
            with self._lock:
                self._entries.pop(key, None)
            raise
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None and entry.mtime_ns == stat.st_mtime_ns and entry.size == stat.st_size:
                return entry.value
        value = reader(path)  # parse outside the lock; worst case two threads both parse
        with self._lock:
            self._entries[key] = _Entry(mtime_ns=stat.st_mtime_ns, size=stat.st_size, value=value)
        return value

    def load_parquet(self, path: Path) -> pd.DataFrame:
        """Load a parquet file through the cache (reloaded on mtime change)."""
        import pandas as pd

        return self.get(path, pd.read_parquet)

    def load_json(self, path: Path) -> Any:
        """Load a JSON file through the cache (reloaded on mtime change)."""
        return self.get(path, lambda p: json.loads(p.read_text()))

    def clear(self) -> None:
        """Drop every cached entry (test hook)."""
        with self._lock:
            self._entries.clear()


#: Process-wide singleton used by the API endpoints.
cache = FileCache()
