"""Feature engineering — leakage-safe multi-horizon window features (stage 2)."""

from fplai.features.windows import (
    FEATURE_PREFIX,
    LABEL_COLUMNS,
    STARTS_WINDOWS,
    WINDOWS,
    assert_no_leakage,
    build_feature_frame,
    get_label_columns,
)

__all__ = [
    "FEATURE_PREFIX",
    "LABEL_COLUMNS",
    "STARTS_WINDOWS",
    "WINDOWS",
    "assert_no_leakage",
    "build_feature_frame",
    "get_label_columns",
]
