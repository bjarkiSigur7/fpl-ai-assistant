"""Static site-data bundle publisher — the $0 release architecture's data plane.

:func:`build_bundle` turns the pipeline artifacts under ``data/processed/`` into the
static JSON bundle the GitHub-Pages frontend consumes (the shared contract locked with
the owner)::

    meta.json             {generated_utc, season, next_gw, next_deadline_utc,
                           model_version, window: {from_gw, through_gw}, missing: [...]}
    players.json          [{player_code, web_name, position, team_short, team_code,
                            price, status, q0_gw1}]                      (the pool)
    xp.json               {gws: [...], players: {"<code>": [xp per gw, same order]}}
    predictions_gw1.json  [{player_code, xp, components..., fixtures: [...]}]
    recommendation.json   the on-disk Recommendation JSON, verbatim structure
    dream_team.json       verbatim structure
    chip_curves.json      chip_curves.parquet as records (v2 sim columns included)
    rating.json           {floor_metric, optimal_metric, window_gws, captain_bonus}
    history/              optional, created empty

Guarantees:

* **Deterministic**: sorted JSON keys, compact separators, explicitly ordered lists,
  every float rounded to 3 dp and ``-0.0`` normalised — two runs over the same inputs
  are byte-identical.  ``generated_utc`` derives from the newest input artifact's
  mtime, never from the wall clock.
* **Self-consistent rating anchors**: ``rating.json``'s ``floor_metric`` /
  ``optimal_metric`` are computed with :func:`fplai.optimizer.rating.squad_metric` and
  :func:`fplai.optimizer.rating.cheapest_legal_squad` (the exact server-side engine —
  zero duplication) over the *rounded* xP table the bundle ships, so a client
  re-running the same greedy formula over ``xp.json`` reproduces them bit-for-bit.
* **Graceful degradation**: optional artifacts (recommendation, dream team, chip
  curves, teams) that are absent produce ``null`` / empty-records files plus their
  name in ``meta.json``'s ``missing`` list; the REQUIRED artifacts
  (``predictions_gw.parquet``, ``predictions.parquet``, ``live_roster.parquet``)
  raise a clear :class:`FileNotFoundError` naming every missing file.
* **Size guard**: the bundle must stay under :data:`MAX_BUNDLE_BYTES` uncompressed;
  per-file byte sizes are reported in the returned :class:`BundleReport`.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from fplai.optimizer.rating import cheapest_legal_squad, squad_metric

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["MAX_BUNDLE_BYTES", "BundleReport", "build_bundle"]

#: Uncompressed bundle budget (shared-contract target: < 2 MB).
MAX_BUNDLE_BYTES = 2_000_000

#: Decimal places every float in the bundle is rounded to.
ROUND_DP = 3

#: Artifacts without which no meaningful bundle exists.
_REQUIRED = ("predictions_gw.parquet", "predictions.parquet", "live_roster.parquet")

#: Optional artifacts: absent -> degraded output + a ``missing`` flag in meta.json.
_OPTIONAL = (
    "recommendation.json",
    "dream_team.json",
    "chip_curves.parquet",
    "chip_sim_report.json",
    "teams.parquet",
    "team_fixtures.parquet",
    "captaincy.parquet",
)

#: chip_curves.parquet v2 season-sim columns (backfilled from chip_sim_report.json
#: when a v1 file lacks them).
_SIM_COLS = ("sd", "p_best_week", "p_beats_hold", "n_rollouts")

#: Bundle files in report order.
_BUNDLE_FILES = (
    "meta.json",
    "players.json",
    "xp.json",
    "predictions_gw1.json",
    "recommendation.json",
    "dream_team.json",
    "chip_curves.json",
    "rating.json",
    "fixtures.json",
    "set_pieces.json",
    "captaincy.json",
)


@dataclass(frozen=True)
class BundleReport:
    """What :func:`build_bundle` wrote: per-file byte sizes and provenance."""

    out_dir: Path
    season: int
    window: tuple[int, int]
    files: dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0
    missing: tuple[str, ...] = ()


# --------------------------------------------------------------------------------------
# JSON canonicalisation (determinism + 3 dp rounding)
# --------------------------------------------------------------------------------------


def _round3(x: float) -> float:
    r = round(float(x), ROUND_DP)
    return 0.0 if r == 0 else r  # normalise -0.0


def _cell(v: Any) -> Any:
    """One parquet scalar -> a JSON-safe python value (NaN/NA -> None, 3 dp floats)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if pd.isna(v):
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else _round3(f)
    if isinstance(v, (dt.datetime, pd.Timestamp)):
        return _iso(v)
    return str(v)


def _round_tree(obj: Any) -> Any:
    """Recursively round floats to 3 dp in a ``json.loads``-produced tree."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else _round3(obj)
    if isinstance(obj, dict):
        return {k: _round_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_tree(v) for v in obj]
    return obj


def _iso(t: dt.datetime | pd.Timestamp) -> str:
    t = t if isinstance(t, dt.datetime) else t.to_pydatetime()
    t = t.astimezone(dt.UTC) if t.tzinfo is not None else t.replace(tzinfo=dt.UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _dump(path: Path, payload: Any) -> int:
    """Write canonical JSON (sorted keys, compact, trailing newline); return byte size."""
    text = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                   allow_nan=False)
        + "\n"
    )
    data = text.encode("utf-8")
    path.write_bytes(data)
    return len(data)


# --------------------------------------------------------------------------------------
# Optional-input helpers
# --------------------------------------------------------------------------------------


def _load_json_opt(path: Path) -> Any | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _season_state(raw_dir: Path | None) -> Any | None:
    """SeasonState from the newest on-disk snapshot's bootstrap.json (best-effort)."""
    if raw_dir is None:
        return None
    root = Path(raw_dir) / "fpl_api" / "snapshots"
    if not root.is_dir():
        return None
    for snap in sorted((d for d in root.iterdir() if d.is_dir()), reverse=True):
        bootstrap = snap / "bootstrap.json"
        if not bootstrap.exists():
            continue
        from fplai.data.fpl_api import parse_season_state

        try:
            return parse_season_state(json.loads(bootstrap.read_text(encoding="utf-8")))
        except (ValueError, json.JSONDecodeError):
            return None
    return None


def _model_version(models_dir: Path | None) -> str | None:
    """The trained-model batch id: ``manifest.json``'s ``created_utc`` (best-effort)."""
    if models_dir is None:
        return None
    try:
        manifest = _load_json_opt(Path(models_dir) / "manifest.json")
    except (json.JSONDecodeError, OSError):
        return None
    if isinstance(manifest, dict) and manifest.get("created_utc"):
        return str(manifest["created_utc"])
    return None


def _unwrap(data: Any, key: str) -> Any:
    """Accept both a bare model dump and a ``{key: {...}}`` wrapper (API convention)."""
    if isinstance(data, dict) and key in data and isinstance(data[key], dict):
        return data[key]
    return data


# --------------------------------------------------------------------------------------
# Section builders
# --------------------------------------------------------------------------------------


def _players_records(
    roster: pd.DataFrame,
    shorts: Mapping[int, str],
    q0_gw1: Mapping[int, float],
    returns: Mapping[int, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    returns = returns or {}
    out = []
    for row in roster.sort_values("player_code", kind="stable").itertuples(index=False):
        code = int(row.player_code)
        q0 = q0_gw1.get(code)
        ret = returns.get(code) or {}
        news = getattr(row, "news", None)
        out.append(
            {
                "player_code": code,
                "web_name": str(row.web_name),
                "position": str(row.position),
                "team_short": shorts.get(int(row.team_code)),
                "team_code": int(row.team_code),
                "price": int(row.price),
                "status": _cell(getattr(row, "status", None)),
                "q0_gw1": None if q0 is None else _round3(q0),
                "ownership": _cell(getattr(row, "selected_by_percent", None)),
                "news": (str(news) or None) if isinstance(news, str) else None,
                "return_gw": _cell(ret.get("return_gw")),
                "pen_order": _cell(getattr(row, "penalties_order", None)),
            }
        )
    return out


def _fixture_records(
    team_fx: pd.DataFrame | None, season: int, shorts: Mapping[int, str]
) -> list[dict[str, Any]]:
    """team_fixtures.parquet -> per-fixture outlook records (window season only)."""
    if team_fx is None:
        return []
    rows = team_fx[team_fx["season"] == season]
    records = []
    for r in rows.sort_values(["gw", "fpl_fixture_id"], kind="stable").itertuples(index=False):
        records.append(
            {
                "gw": int(r.gw),
                "fpl_fixture_id": int(r.fpl_fixture_id),
                "kickoff_utc": _cell(getattr(r, "kickoff_utc", None)),
                "home_code": int(r.home_team_code),
                "home_short": shorts.get(int(r.home_team_code)),
                "away_code": int(r.away_team_code),
                "away_short": shorts.get(int(r.away_team_code)),
                "home_xg": _cell(getattr(r, "home_lambda", None)),
                "away_xg": _cell(getattr(r, "away_lambda", None)),
                "p_cs_home": _cell(getattr(r, "p_cs_home", None)),
                "p_cs_away": _cell(getattr(r, "p_cs_away", None)),
                "p_home_win": _cell(getattr(r, "p_home_win", None)),
                "p_draw": _cell(getattr(r, "p_draw", None)),
                "p_away_win": _cell(getattr(r, "p_away_win", None)),
                "odds_blended": bool(getattr(r, "odds_blended", False)),
            }
        )
    return records


#: Roster set-piece columns -> compact bundle keys.
_SET_PIECE_FIELDS = (
    ("penalties_order", "pen"),
    ("direct_freekicks_order", "fk"),
    ("corners_and_indirect_freekicks_order", "corner"),
)


def _set_piece_records(roster: pd.DataFrame, shorts: Mapping[int, str]) -> list[dict[str, Any]]:
    """Players holding any set-piece duty, with per-duty order + editorial note."""
    if not any(col for col, _ in _SET_PIECE_FIELDS if col in roster.columns):
        return []
    records = []
    for row in roster.sort_values("player_code", kind="stable").itertuples(index=False):
        duties: dict[str, Any] = {}
        for col, key in _SET_PIECE_FIELDS:
            order = getattr(row, col, None)
            if order is not None and not pd.isna(order):
                duties[key] = int(order)
                note = getattr(row, f"{col.rsplit('_order', 1)[0]}_text", None)
                if isinstance(note, str) and note:
                    duties[f"{key}_note"] = note
        if not duties:
            continue
        records.append(
            {
                "player_code": int(row.player_code),
                "web_name": str(row.web_name),
                "team_code": int(row.team_code),
                "team_short": shorts.get(int(row.team_code)),
                "position": str(row.position),
                **duties,
            }
        )
    return records


def _captaincy_records(cap: pd.DataFrame | None) -> list[dict[str, Any]]:
    if cap is None:
        return []
    records = [
        {k: _cell(v) for k, v in rec.items()} for rec in cap.to_dict(orient="records")
    ]
    records.sort(key=lambda r: (-(r.get("xp") or 0.0), r["player_code"]))
    return records


def _xp_table(
    gws: list[int], codes: list[int], xp_of: Mapping[tuple[int, int], float]
) -> dict[str, Any]:
    return {
        "gws": gws,
        "players": {str(c): [xp_of.get((gw, c), 0.0) for gw in gws] for c in sorted(codes)},
    }


def _gw1_records(
    gw1: pd.DataFrame, fixtures: pd.DataFrame, shorts: Mapping[int, str]
) -> list[dict[str, Any]]:
    """Next-GW detail records: predictions_gw row + per-fixture legs from predictions."""
    legs: dict[int, list[dict[str, Any]]] = {}
    for row in fixtures.sort_values(
        ["player_code", "fpl_fixture_id"], kind="stable"
    ).itertuples(index=False):
        rec = {
            "fpl_fixture_id": int(row.fpl_fixture_id),
            "opponent_code": int(row.opponent_code),
            "opponent_short": shorts.get(int(row.opponent_code)),
            "was_home": bool(row.was_home),
            "xp": _cell(row.xp),
            "q0": _cell(row.q0),
            "q1": _cell(getattr(row, "q1", None)),
            "q2": _cell(getattr(row, "q2", None)),
            "mu1": _cell(getattr(row, "mu1", None)),
            "mu2": _cell(getattr(row, "mu2", None)),
        }
        legs.setdefault(int(row.player_code), []).append(rec)

    drop = {"season", "gw"}
    records = []
    for rec in gw1.to_dict(orient="records"):
        out = {k: _cell(v) for k, v in rec.items() if k not in drop}
        code = int(rec["player_code"])
        team_code = rec.get("team_code")
        out["team_short"] = shorts.get(int(team_code)) if team_code is not None else None
        out["fixtures"] = legs.get(code, [])
        records.append(out)
    records.sort(key=lambda r: (-(r.get("xp") or 0.0), r["player_code"]))
    return records


def _chip_curve_records(
    curves: pd.DataFrame | None, sim_report: Any | None
) -> list[dict[str, Any]]:
    """chip_curves.parquet -> records; v1 files get the sim columns backfilled from
    chip_sim_report.json (matched on ``(chip, gw)``) where that report exists."""
    if curves is None:
        return []
    records = [
        {k: _cell(v) for k, v in rec.items()} for rec in curves.to_dict(orient="records")
    ]
    needs_sim = [c for c in _SIM_COLS if c not in curves.columns]
    if needs_sim and isinstance(sim_report, dict):
        n_rollouts = sim_report.get("n_rollouts")
        stats = {
            (str(s.get("chip")), int(s.get("gw"))): s
            for s in sim_report.get("stats", [])
            if isinstance(s, dict) and s.get("chip") is not None and s.get("gw") is not None
        }
        for rec in records:
            stat = stats.get((str(rec.get("chip")), int(rec.get("gw", -1))), {})
            for col in needs_sim:
                v = n_rollouts if col == "n_rollouts" else stat.get(col)
                rec[col] = _cell(v) if isinstance(v, (int, float)) else None
    records.sort(key=lambda r: (str(r.get("chip")), int(r.get("gw", 0))))
    return records


def _rating_payload(
    roster: pd.DataFrame,
    preds: pd.DataFrame,
    dream: Any | None,
    xp_of: Mapping[tuple[int, int], float],
    gws: list[int],
) -> dict[str, Any]:
    """The client-side rating constants, via the EXACT server metric functions."""
    pos_of: dict[int, str] = {
        int(r.player_code): str(r.position) for r in roster.itertuples(index=False)
    }
    floor_metric = squad_metric(cheapest_legal_squad(roster), pos_of, xp_of, gws)

    optimal_metric: float | None = None
    if dream is not None:
        dream_squad = [int(c) for c in _unwrap(dream, "dream_team")["squad"]]
        missing = [c for c in dream_squad if c not in pos_of]
        if missing and "position" in preds.columns:
            for row in preds[preds["player_code"].isin(missing)].itertuples(index=False):
                pos_of.setdefault(int(row.player_code), str(row.position))
            missing = [c for c in dream_squad if c not in pos_of]
        if missing:
            raise ValueError(
                f"dream squad players unknown to roster/predictions: {sorted(missing)}"
            )
        optimal_metric = squad_metric(dream_squad, pos_of, xp_of, gws)

    return {
        "floor_metric": _round3(floor_metric),
        "optimal_metric": None if optimal_metric is None else _round3(optimal_metric),
        "window_gws": gws,
        "captain_bonus": True,
    }


# --------------------------------------------------------------------------------------
# build_bundle
# --------------------------------------------------------------------------------------


def build_bundle(
    processed_dir: Path | str,
    out_dir: Path | str,
    *,
    raw_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
) -> BundleReport:
    """Build the shared-contract static bundle from on-disk artifacts.

    Args:
        processed_dir: The pipeline's artifact directory (``data/processed``).
        raw_dir: Optional raw-data root; when given, ``meta.json``'s
            ``next_deadline_utc`` comes from the newest FPL snapshot on disk.
        models_dir: Optional model-artifact root; supplies ``model_version``.

    Returns:
        A :class:`BundleReport` with per-file byte sizes.

    Raises:
        FileNotFoundError: when any REQUIRED artifact is absent (all named at once).
        ValueError: when the bundle exceeds :data:`MAX_BUNDLE_BYTES`, or the on-disk
            artifacts are mutually inconsistent (e.g. dream-team players unknown to
            the roster and predictions).
    """
    processed = Path(processed_dir)
    out = Path(out_dir)
    absent_required = [n for n in _REQUIRED if not (processed / n).exists()]
    if absent_required:
        raise FileNotFoundError(
            f"required artifacts missing from {processed}: {', '.join(absent_required)} "
            "— run `fplai predict` / `fplai refresh` first"
        )

    pred_gw = pd.read_parquet(processed / "predictions_gw.parquet")
    pred = pd.read_parquet(processed / "predictions.parquet")
    roster = pd.read_parquet(processed / "live_roster.parquet")
    if pred_gw.empty:
        raise ValueError("predictions_gw.parquet is empty — nothing to publish")

    missing = [n for n in _OPTIONAL if not (processed / n).exists()]
    recommendation = _load_json_opt(processed / "recommendation.json")
    dream = _load_json_opt(processed / "dream_team.json")
    sim_report = _load_json_opt(processed / "chip_sim_report.json")
    curves_path = processed / "chip_curves.parquet"
    curves = pd.read_parquet(curves_path) if curves_path.exists() else None
    team_fx_path = processed / "team_fixtures.parquet"
    team_fx = pd.read_parquet(team_fx_path) if team_fx_path.exists() else None
    cap_path = processed / "captaincy.parquet"
    captaincy = pd.read_parquet(cap_path) if cap_path.exists() else None

    # ---- window --------------------------------------------------------------------
    season = int(pred_gw["season"].max())
    sg = pred_gw[pred_gw["season"] == season]
    gws = sorted(int(g) for g in sg["gw"].unique())
    from_gw, through_gw = gws[0], gws[-1]

    #: (gw, player_code) -> ROUNDED xp — the single table xp.json AND the rating
    #: anchors are computed from, so client-side recomputation matches exactly.
    xp_of: dict[tuple[int, int], float] = {
        (int(gw), int(code)): _round3(xp)
        for gw, code, xp in zip(sg["gw"], sg["player_code"], sg["xp"], strict=True)
    }
    gw1_rows = sg[sg["gw"] == from_gw]
    q0_gw1: dict[int, float] = {
        int(code): float(q0)
        for code, q0 in zip(gw1_rows["player_code"], gw1_rows["q0"], strict=True)
        if pd.notna(q0)
    }

    shorts: dict[int, str] = {}
    teams_path = processed / "teams.parquet"
    if teams_path.exists():
        teams = pd.read_parquet(teams_path)
        rows = teams[teams["season"] == season]
        shorts = {
            int(c): str(s)
            for c, s in zip(rows["team_code"], rows["short_name"], strict=True)
        }

    # ---- provenance (deterministic: newest INPUT mtime, never the wall clock) --------
    consumed = [
        processed / n
        for n in (*_REQUIRED, *_OPTIONAL, f"availability_{season}.json")
        if (processed / n).exists()
    ]
    generated_utc = _iso(
        dt.datetime.fromtimestamp(max(p.stat().st_mtime for p in consumed), tz=dt.UTC)
    )
    state = _season_state(Path(raw_dir) if raw_dir is not None else None)
    next_deadline_utc: str | None = None
    if (
        state is not None
        and state.season == season
        and state.next_gw == from_gw
        and state.next_deadline_utc is not None
    ):
        next_deadline_utc = _iso(state.next_deadline_utc)

    # ---- write ------------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    (out / "history").mkdir(exist_ok=True)
    sizes: dict[str, int] = {}

    availability = _load_json_opt(processed / f"availability_{season}.json")
    returns: dict[int, Any] = {}
    if isinstance(availability, dict) and isinstance(availability.get("returns"), dict):
        returns = {int(c): r for c, r in availability["returns"].items()}
    sizes["players.json"] = _dump(
        out / "players.json", _players_records(roster, shorts, q0_gw1, returns)
    )
    sizes["fixtures.json"] = _dump(
        out / "fixtures.json", _fixture_records(team_fx, season, shorts)
    )
    sizes["set_pieces.json"] = _dump(
        out / "set_pieces.json", _set_piece_records(roster, shorts)
    )
    sizes["captaincy.json"] = _dump(out / "captaincy.json", _captaincy_records(captaincy))
    codes = sorted({c for _, c in xp_of})
    sizes["xp.json"] = _dump(out / "xp.json", _xp_table(gws, codes, xp_of))
    gw1_fixtures = pred[(pred["season"] == season) & (pred["gw"] == from_gw)]
    sizes["predictions_gw1.json"] = _dump(
        out / "predictions_gw1.json", _gw1_records(gw1_rows, gw1_fixtures, shorts)
    )
    sizes["recommendation.json"] = _dump(
        out / "recommendation.json", _round_tree(recommendation)
    )
    sizes["dream_team.json"] = _dump(out / "dream_team.json", _round_tree(dream))
    sizes["chip_curves.json"] = _dump(
        out / "chip_curves.json", _chip_curve_records(curves, sim_report)
    )
    sizes["rating.json"] = _dump(
        out / "rating.json", _rating_payload(roster, sg, dream, xp_of, gws)
    )
    sizes["meta.json"] = _dump(
        out / "meta.json",
        {
            "generated_utc": generated_utc,
            "season": season,
            "next_gw": from_gw,
            "next_deadline_utc": next_deadline_utc,
            "model_version": _model_version(
                Path(models_dir) if models_dir is not None else None
            ),
            "window": {"from_gw": from_gw, "through_gw": through_gw},
            "missing": sorted(missing),
        },
    )

    ordered = {name: sizes[name] for name in _BUNDLE_FILES}
    total = sum(ordered.values())
    if total > MAX_BUNDLE_BYTES:
        detail = ", ".join(f"{n}={s:,}B" for n, s in ordered.items())
        raise ValueError(
            f"bundle size {total:,} B exceeds the {MAX_BUNDLE_BYTES:,} B budget ({detail})"
        )
    return BundleReport(
        out_dir=out,
        season=season,
        window=(from_gw, through_gw),
        files=ordered,
        total_bytes=total,
        missing=tuple(sorted(missing)),
    )
