"""Squad state for the optimizer: what a manager owns going into the next deadline.

Implements the ``optimizer/state.py`` contract from ``docs/ARCHITECTURE.md``:
:class:`SquadState` (pydantic) plus :meth:`SquadState.from_entry`, which reconstructs the
state from the public FPL API (picks + transfers + history replay).

Conventions: prices are £0.1m units; ``season`` is the start year; chip instance ids are
lowercase ``"<kind><set>"`` strings (``"wc1"``, ``"fh2"``, ...) derived from
:func:`fplai.rules.chip_windows`.

Pre-deadline limitation (documented per the contract)
-----------------------------------------------------
The public API only exposes *picks* (squad, captain, bench order, active chip of that GW)
for gameweeks whose deadline has passed. ``from_entry`` therefore starts from the picks of
the last **finished** GW and then applies every transfer logged after it (the transfer log
and chip log ARE public immediately), so pending transfers and a pre-played WC/FH for the
upcoming GW are reflected. What cannot be recovered pre-deadline is the user's pending
*lineup/captaincy* (irrelevant here — the optimizer re-picks those) and, before GW1, any
squad at all (use ``state=None`` initial-build mode instead).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from fplai import rules
from fplai.data.fpl_api import FplApiClient, parse_season_state

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)

#: Chip instance id: chip kind + set number, e.g. ``"wc1"``, ``"tc2"`` (``"aoa1"``/``"am1"``
#: only exist for historical seasons and are not plannable by the MILP).
ChipId = Annotated[str, StringConstraints(to_lower=True, pattern=r"^(wc|fh|bb|tc|aoa|am)[12]$")]

#: FPL API chip name -> our chip kind.
API_CHIP_NAMES: dict[str, str] = {
    "wildcard": "wc",
    "freehit": "fh",
    "bboost": "bb",
    "3xc": "tc",
    "manager": "am",
    "assistant_manager": "am",
}


def all_chip_ids(season: int) -> list[str]:
    """Return every chip instance id playable in ``season``, per :func:`rules.chip_windows`."""
    return [f"{w.name.lower()}{w.set}" for w in rules.chip_windows(season)]


def chip_instance_for(
    season: int, kind: str, gw: int, available: Iterable[str] | None = None
) -> str | None:
    """Resolve which instance of chip ``kind`` ("wc"/"fh"/"bb"/"tc") covers ``gw``.

    Returns the lowest-set instance whose window contains ``gw`` and (when ``available``
    is given) whose id is in ``available``; ``None`` if no instance matches.
    """
    avail = None if available is None else set(available)
    for window in sorted(rules.chip_windows(season), key=lambda w: w.set):
        cid = f"{window.name.lower()}{window.set}"
        if window.name.lower() != kind:
            continue
        if not (window.first_gw <= gw <= window.last_gw):
            continue
        if avail is not None and cid not in avail:
            continue
        return cid
    return None


class OwnedPlayer(BaseModel):
    """One owned player: identity plus the two prices the sell formula needs (£0.1m units)."""

    model_config = ConfigDict(frozen=True)

    player_code: int = Field(ge=1, description="Stable FPL element.code")
    purchase_price: int = Field(ge=0, description="Price paid, £0.1m units")
    current_price: int = Field(ge=0, description="Current price, £0.1m units")

    def sell_price(self) -> int:
        """Selling price per FPL_KNOWLEDGE §1.8 (50% sell-on fee, floored to £0.1m)."""
        return rules.sell_price(self.purchase_price, self.current_price)


class SquadState(BaseModel):
    """A manager's squad state going into the next deadline (ARCHITECTURE stage-3 contract).

    ``None`` in place of a ``SquadState`` means initial-squad-build mode: GW1 semantics,
    £100.0m budget, unlimited transfers into the first GW, all chips available per their
    windows (handled by :func:`fplai.optimizer.milp.solve_plan`).
    """

    model_config = ConfigDict(frozen=True)

    season: int = Field(description="Season start year, e.g. 2026 for 2026-27")
    current_gw: int = Field(ge=1, le=38, description="The next GW to be played")
    squad: list[OwnedPlayer] = Field(description="The 15 owned players")
    bank: int = Field(ge=0, description="In the bank, £0.1m units")
    free_transfers: int = Field(ge=1, le=5, description="FTs going into current_gw")
    chips_available: list[ChipId] = Field(default_factory=list)
    active_chip: ChipId | None = Field(
        default=None, description="Chip already played for current_gw (e.g. an active WC)"
    )

    @field_validator("season")
    @classmethod
    def _season_known(cls, v: int) -> int:
        if v not in rules.SEASONS:
            raise ValueError(f"season {v} outside supported range {rules.SEASONS}")
        return v

    def sell_value(self) -> int:
        """Total selling value of the squad, £0.1m units."""
        return sum(p.sell_price() for p in self.squad)

    @classmethod
    def from_entry(cls, entry_id: int, *, client: FplApiClient | None = None) -> SquadState:
        """Build the state for a public FPL entry (team) id from the official API.

        Reconstruction recipe (public endpoints only):

        * squad: picks at the last finished GW, then every logged transfer after it
          applied in chronological order (pending next-GW transfers are public);
        * purchase prices: latest transfer-in cost per element from the transfer log
          (Free-Hit-week transfers excluded — those squads reverted), falling back to
          the season start price (``now_cost - cost_change_start``) for players held
          since GW1 (prices are frozen until the GW1 deadline, so start price = paid);
        * bank: the history row of the last finished GW, adjusted for applied transfers;
        * free transfers: full replay of ``rules.next_free_transfers`` from GW2 using
          per-GW transfer counts and WC/FH chip weeks (plus the 2025-26 AFCON GW16
          top-up flagged in ``rules.SEASON_FLAGS``), minus any pending transfers
          already confirmed for the upcoming GW (they consumed FTs when made; floored
          at 1 with a warning if hits were already committed);
        * chips: season inventory from ``rules.chip_windows`` minus played instances;
          a chip logged for the upcoming GW becomes ``active_chip``.

        See the module docstring for the pre-deadline limitation. If the season has
        fully finished, ``current_gw`` is clamped to 38 (planning is then moot).

        Raises:
            ValueError: before GW1 has finished (no public picks exist yet — build an
                initial squad with ``state=None`` instead), or on malformed payloads.
        """
        api = client if client is not None else FplApiClient()
        bootstrap = api.bootstrap()
        season_state = parse_season_state(bootstrap)
        season = season_state.season
        events: list[dict[str, Any]] = bootstrap.get("events") or []
        finished = [int(e["id"]) for e in events if e.get("finished")]
        if not finished:
            raise ValueError(
                "no finished gameweek yet — pre-season picks are not public; "
                "build an initial squad with state=None instead"
            )
        last_finished = max(finished)
        next_gw = season_state.next_gw or min(last_finished + 1, 38)
        if season_state.next_gw is None:
            logger.warning(
                "season %s appears finished; clamping current_gw to %d", season, next_gw
            )
        elements = {int(el["id"]): el for el in bootstrap.get("elements") or []}

        history = api.entry_history(entry_id)
        picks_payload = api.entry_picks(entry_id, last_finished)
        transfers = api.entry_transfers(entry_id)  # newest first

        # chip log: gw -> kind
        chip_events: dict[int, str] = {}
        for row in history.get("chips") or []:
            kind = API_CHIP_NAMES.get(str(row.get("name", "")).lower())
            if kind is not None:
                chip_events[int(row["event"])] = kind
        fh_gws = {gw for gw, kind in chip_events.items() if kind == "fh"}

        # purchase prices: latest (non-Free-Hit) transfer-in cost per element id
        last_in_cost: dict[int, int] = {}
        for t in transfers:
            if int(t.get("event", 0)) in fh_gws:
                continue  # FH transfers reverted; they never set a real purchase price
            last_in_cost.setdefault(int(t["element_in"]), int(t["element_in_cost"]))

        current_rows = {int(r["event"]): r for r in history.get("current") or []}
        bank = int(current_rows.get(last_finished, {}).get("bank", 0))

        squad_ids = [int(p["element"]) for p in picks_payload.get("picks") or []]

        # apply transfers logged after the last finished GW (pending ones are public),
        # except Free-Hit weeks whose squads reverted
        later = [
            t
            for t in transfers
            if last_finished < int(t.get("event", -1)) <= next_gw
            and int(t["event"]) not in fh_gws
        ]
        for t in reversed(later):  # transfer log is newest-first; apply chronologically
            out_id, in_id = int(t["element_out"]), int(t["element_in"])
            if out_id in squad_ids:
                squad_ids.remove(out_id)
                bank += int(t["element_out_cost"])
            squad_ids.append(in_id)
            bank -= int(t["element_in_cost"])

        squad: list[OwnedPlayer] = []
        for eid in squad_ids:
            el = elements.get(eid)
            if el is None:
                raise ValueError(f"entry {entry_id}: picked element id {eid} not in bootstrap")
            now = int(el["now_cost"])
            purchase = last_in_cost.get(eid, now - int(el.get("cost_change_start") or 0))
            squad.append(
                OwnedPlayer(player_code=int(el["code"]), purchase_price=purchase, current_price=now)
            )

        # free-transfer replay: 1 FT going into GW2 (GW1 is unlimited), then the rules fn
        flags = rules.SEASON_FLAGS[season]
        ft = 1
        for g in range(2, next_gw):
            used = int(current_rows.get(g, {}).get("event_transfers", 0) or 0)
            wc_fh = chip_events.get(g) in ("wc", "fh")
            ft = rules.next_free_transfers(ft, used, season, wc_fh)
            if flags.amnesty == "topped_to_5_at_GW16" and g + 1 == 16:
                ft = flags.max_free_transfers  # 2025-26 AFCON one-off top-up
        # pending transfers already logged for the upcoming GW consumed FTs the moment
        # they were confirmed (unless a WC/FH is active, when transfers are free) — the
        # optimizer must see the REMAINING count, not the deadline-time accrual.
        if chip_events.get(next_gw) not in ("wc", "fh"):
            pending_used = sum(1 for t in later if int(t["event"]) == next_gw)
            if pending_used:
                if ft - pending_used < 1:
                    logger.warning(
                        "entry %d: %d pending transfer(s) exceed the %d banked FT(s) "
                        "(hits already committed); clamping remaining FTs to 1",
                        entry_id,
                        pending_used,
                        ft,
                    )
                ft = max(1, ft - pending_used)

        available = all_chip_ids(season)
        active_chip: str | None = None
        for gw, kind in sorted(chip_events.items()):
            inst = chip_instance_for(season, kind, gw, available)
            if inst is None:
                logger.warning("cannot map played chip %s@GW%d to an instance", kind, gw)
                continue
            available.remove(inst)
            if gw == next_gw:
                active_chip = inst

        return cls(
            season=season,
            current_gw=next_gw,
            squad=squad,
            bank=bank,
            free_transfers=ft,
            chips_available=available,
            active_chip=active_chip,
        )


def from_entry(entry_id: int, *, client: FplApiClient | None = None) -> SquadState:
    """Module-level alias for :meth:`SquadState.from_entry` (ARCHITECTURE contract name)."""
    return SquadState.from_entry(entry_id, client=client)
