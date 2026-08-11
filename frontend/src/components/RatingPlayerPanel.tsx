"use client";

/**
 * RATING PLAYER PANEL — the left column of the AI RATING page, echoing the
 * official FPL squad-selection anatomy in the desk's own chrome: search box,
 * position tabs, club + max-price + sort controls, and a scrollable pool list
 * with a + on every row (disabled with a reason whenever the pick is illegal).
 */

import { useMemo, useState } from "react";
import { moneyBare, xp2 } from "@/lib/format";
import type { IndexedPlayer, PlayerIndex } from "@/lib/playerIndex";
import { POSITIONS, type Position } from "@/lib/types";
import { Card, CardHead } from "./ui";

type SortKey = "xp" | "xph" | "price" | "name";

const SORT_LABEL: Record<SortKey, string> = {
  xp: "xP NEXT GW",
  xph: "HORIZON xP",
  price: "PRICE",
  name: "NAME",
};

/** Max-price steps in 0.1m units: £14.0 down to £4.0. */
const PRICE_STEPS: number[] = (() => {
  const out: number[] = [];
  for (let t = 140; t >= 40; t -= t > 90 ? 10 : 5) out.push(t);
  return out;
})();

function PoolRow({
  p,
  gw,
  isSelected,
  blocked,
  onToggle,
}: {
  p: IndexedPlayer;
  gw: number;
  isSelected: boolean;
  /** Reason the + is disabled, or null when the pick is legal. */
  blocked: string | null;
  onToggle: (code: number) => void;
}) {
  const disabled = !isSelected && blocked !== null;
  return (
    <li className="flex items-center gap-2.5 border-b border-hairline-soft px-3 py-1.5 last:border-b-0">
      <span className="min-w-0 flex-1 truncate text-[12.5px] font-medium text-ink">
        {p.web_name}
      </span>
      <span className="w-9 shrink-0 font-mono text-[10px] text-ink-dim">{p.team_short}</span>
      <span className="w-8 shrink-0 font-mono text-[10px] text-ink-dim">{p.position}</span>
      <span className="w-9 shrink-0 text-right font-mono text-[11.5px] tnum text-ink-mid">
        {moneyBare(p.price)}
      </span>
      <span className="w-11 shrink-0 text-right font-mono text-[12px] font-semibold tnum text-ink">
        {xp2(p.xpByGw.get(gw) ?? 0)}
      </span>
      <button
        onClick={() => onToggle(p.player_code)}
        disabled={disabled}
        aria-label={
          isSelected ? `Remove ${p.web_name} from squad` : `Add ${p.web_name} to squad`
        }
        title={isSelected ? "IN SQUAD — REMOVE" : (blocked ?? "ADD TO SQUAD")}
        className={`hit relative flex h-[22px] w-[22px] shrink-0 items-center justify-center rounded border font-mono text-[12px] leading-none transition-colors ${
          isSelected
            ? "border-pitch-deep bg-pitch/15 text-pitch-bright hover:border-neg hover:bg-transparent hover:text-neg"
            : disabled
              ? "cursor-not-allowed border-hairline-soft text-ink-dim opacity-40"
              : "border-hairline text-ink-mid hover:border-ink-dim hover:bg-raised hover:text-ink"
        }`}
      >
        {isSelected ? "✓" : "+"}
      </button>
    </li>
  );
}

export function RatingPlayerPanel({
  index,
  gw,
  selected,
  canAdd,
  onToggle,
  pos,
  onPosChange,
}: {
  index: PlayerIndex;
  gw: number;
  selected: Set<number>;
  /** null when the player can legally be added; otherwise the blocking rule. */
  canAdd: (p: IndexedPlayer) => string | null;
  onToggle: (code: number) => void;
  /** Position filter is owned by the page — the pitch's empty slots set it too. */
  pos: Position | "ALL";
  onPosChange: (pos: Position | "ALL") => void;
}) {
  const [q, setQ] = useState("");
  const [club, setClub] = useState<string>("ALL");
  const [maxPrice, setMaxPrice] = useState<number>(0); // 0 = no cap
  const [sort, setSort] = useState<SortKey>("xp");

  const clubs = useMemo(
    () => [...index.teams.values()].map((t) => t.short_name).sort(),
    [index],
  );

  const rows = useMemo(() => {
    const needle = q.trim().toLowerCase();
    const pool = [...index.byCode.values()]
      .filter((p) => pos === "ALL" || p.position === pos)
      .filter((p) => club === "ALL" || p.team_short === club)
      .filter((p) => maxPrice === 0 || p.price <= maxPrice)
      .filter(
        (p) =>
          needle === "" ||
          p.web_name.toLowerCase().includes(needle) ||
          p.team_short.toLowerCase().includes(needle),
      );
    const xpAt = (p: IndexedPlayer) => p.xpByGw.get(gw) ?? 0;
    pool.sort((a, b) => {
      switch (sort) {
        case "xp":
          return xpAt(b) - xpAt(a);
        case "xph":
          return b.xpHorizon - a.xpHorizon;
        case "price":
          return b.price - a.price;
        case "name":
          return a.web_name.localeCompare(b.web_name);
      }
    });
    return pool;
  }, [index, q, pos, club, maxPrice, sort, gw]);

  const selectClass =
    "rounded border border-hairline bg-surface px-2 py-1.5 font-mono text-[10.5px] text-ink-mid outline-none focus:border-ink-dim";

  return (
    <Card>
      <CardHead label="PLAYER POOL" right={`xP GW${gw} · ${rows.length} OF ${index.byCode.size}`} />
      <div className="space-y-2 border-b border-hairline-soft px-3 py-3">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search player or club…"
          aria-label="Search players"
          className="w-full rounded border border-hairline bg-surface px-3 py-1.5 text-[13px] text-ink placeholder:text-ink-dim focus:border-ink-dim focus:outline-none"
        />
        <div
          className="inline-flex rounded border border-hairline p-0.5"
          role="tablist"
          aria-label="Position filter"
        >
          {(["ALL", ...POSITIONS] as const).map((p) => (
            <button
              key={p}
              role="tab"
              aria-selected={pos === p}
              onClick={() => onPosChange(p)}
              className={`hit relative rounded px-2.5 py-1 font-mono text-[10.5px] tracking-[0.12em] ${
                pos === p ? "bg-raised text-ink" : "text-ink-dim hover:text-ink-mid"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
        <div className="flex flex-wrap gap-2">
          <select
            value={club}
            onChange={(e) => setClub(e.target.value)}
            aria-label="Club filter"
            className={selectClass}
          >
            <option value="ALL">ALL CLUBS</option>
            {clubs.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <select
            value={maxPrice}
            onChange={(e) => setMaxPrice(Number(e.target.value))}
            aria-label="Max price filter"
            className={selectClass}
          >
            <option value={0}>MAX £ — ANY</option>
            {PRICE_STEPS.map((t) => (
              <option key={t} value={t}>
                ≤ £{moneyBare(t)}m
              </option>
            ))}
          </select>
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            aria-label="Sort pool"
            className={selectClass}
          >
            {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => (
              <option key={k} value={k}>
                SORT — {SORT_LABEL[k]}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="flex items-center gap-2.5 border-b border-hairline px-3 py-1.5">
        <span className="microlabel min-w-0 flex-1">PLAYER</span>
        <span className="microlabel w-9 shrink-0">CLUB</span>
        <span className="microlabel w-8 shrink-0">POS</span>
        <span className="microlabel w-9 shrink-0 text-right">£</span>
        <span className="microlabel w-11 shrink-0 text-right">xP</span>
        <span className="w-[22px] shrink-0" aria-hidden="true" />
      </div>
      <ul className="deskscroll max-h-[544px] overflow-y-auto">
        {rows.map((p) => (
          <PoolRow
            key={p.player_code}
            p={p}
            gw={gw}
            isSelected={selected.has(p.player_code)}
            blocked={selected.has(p.player_code) ? null : canAdd(p)}
            onToggle={onToggle}
          />
        ))}
        {rows.length === 0 ? (
          <li className="px-3 py-6 text-center text-[12.5px] text-ink-dim">
            No players match these filters.
          </li>
        ) : null}
      </ul>
    </Card>
  );
}
