"use client";

/**
 * Fixtures: the model-based difficulty ticker. One row per club, one column per
 * GW; each cell is that club's fixture colored by OUR model's difficulty — not
 * the official 1-5 FDR. Toggle the lens: ATTACK (expected goals FOR — backing
 * attackers), DEFENCE (clean-sheet probability — backing defenders/GKs), or
 * OVERALL (blend). Sort by average difficulty over a GW range to find fixture
 * runs; cells carry the opponent label + venue so color is never the only
 * encoding, and tooltips give the exact model numbers.
 */

import { useMemo, useState } from "react";
import { OFFLINE_HINT, useFixturesOutlook } from "@/lib/api";
import type { FixtureOutlook } from "@/lib/types";
import { ScrollFade } from "@/components/ScrollFade";
import { Card, EmptyState, PageTitle, Skeleton } from "@/components/ui";

type Lens = "overall" | "attack" | "defence";

/** Diverging ramp, easy -> hard (teal-green pole, neutral mid, red-orange pole).
 * Poles are CVD-distinguishable (teal carries blue); labels ride every cell. */
const RAMP = ["#22c987", "#187a50", "#4b4b47", "#a84a2d", "#e65a3c"] as const;
/** Per-step ink: bright poles take dark text, dim middle takes light text. */
const RAMP_INK = ["#0d0d0d", "#ffffff", "#ffffff", "#ffffff", "#0d0d0d"] as const;

interface Cell {
  gw: number;
  opponent: string;
  home: boolean;
  /** Expected goals FOR this team in the fixture. */
  xgFor: number | null;
  /** This team's clean-sheet probability. */
  pCs: number | null;
  oddsBlended: boolean;
  kickoff: string | null;
}

interface TeamRow {
  code: number;
  short: string;
  cells: Map<number, Cell[]>;
}

function buildRows(fixtures: FixtureOutlook[]): { rows: TeamRow[]; gws: number[] } {
  const byTeam = new Map<number, TeamRow>();
  const gwSet = new Set<number>();
  const upsert = (code: number, short: string | null) => {
    let row = byTeam.get(code);
    if (!row) {
      row = { code, short: short ?? String(code), cells: new Map() };
      byTeam.set(code, row);
    }
    return row;
  };
  for (const f of fixtures) {
    gwSet.add(f.gw);
    const home = upsert(f.home_code, f.home_short);
    const away = upsert(f.away_code, f.away_short);
    const base = { gw: f.gw, oddsBlended: f.odds_blended, kickoff: f.kickoff_utc };
    const push = (row: TeamRow, cell: Cell) => {
      const list = row.cells.get(f.gw) ?? [];
      list.push(cell);
      row.cells.set(f.gw, list);
    };
    push(home, {
      ...base,
      opponent: away.short,
      home: true,
      xgFor: f.home_xg,
      pCs: f.p_cs_home,
    });
    push(away, {
      ...base,
      opponent: home.short,
      home: false,
      xgFor: f.away_xg,
      pCs: f.p_cs_away,
    });
  }
  return {
    rows: [...byTeam.values()].sort((a, b) => a.short.localeCompare(b.short)),
    gws: [...gwSet].sort((a, b) => a - b),
  };
}

/** Higher = better fixture for this club under the chosen lens. */
function cellScore(cell: Cell, lens: Lens): number | null {
  if (lens === "attack") return cell.xgFor;
  if (lens === "defence") return cell.pCs;
  if (cell.xgFor === null || cell.pCs === null) return cell.xgFor ?? cell.pCs;
  // Overall: normalize xG (~0..3) and CS prob (0..1) onto comparable scales.
  return cell.xgFor / 3 + cell.pCs;
}

/** Quintile bucket of `v` within `all` (0 = easiest .. 4 = hardest). */
function bucketOf(v: number, all: number[]): number {
  let below = 0;
  for (const x of all) if (x < v) below++;
  const pct = all.length > 1 ? below / (all.length - 1) : 0.5;
  return Math.min(4, Math.floor((1 - pct) * 5));
}

export default function FixturesPage() {
  const { data, isLoading, error } = useFixturesOutlook();
  const [lens, setLens] = useState<Lens>("overall");
  const [fromGw, setFromGw] = useState<number | null>(null);
  const [toGw, setToGw] = useState<number | null>(null);
  const [sortByEase, setSortByEase] = useState(true);

  const built = useMemo(() => (data ? buildRows(data) : null), [data]);

  const view = useMemo(() => {
    if (!built) return null;
    const gws = built.gws.filter(
      (g) => (fromGw === null || g >= fromGw) && (toGw === null || g <= toGw),
    );
    const scores: number[] = [];
    for (const row of built.rows) {
      for (const gw of gws) {
        for (const c of row.cells.get(gw) ?? []) {
          const s = cellScore(c, lens);
          if (s !== null) scores.push(s);
        }
      }
    }
    const rows = built.rows.map((row) => {
      let sum = 0;
      let n = 0;
      for (const gw of gws) {
        for (const c of row.cells.get(gw) ?? []) {
          const s = cellScore(c, lens);
          if (s !== null) {
            sum += s;
            n++;
          }
        }
      }
      return { row, avg: n ? sum / n : -Infinity };
    });
    if (sortByEase) rows.sort((a, b) => b.avg - a.avg);
    return { gws, rows, scores };
  }, [built, lens, fromGw, toGw, sortByEase]);

  if (error) return <EmptyState title="NO FIXTURE OUTLOOK" detail={OFFLINE_HINT} />;

  const gwOptions = built?.gws ?? [];

  return (
    <div>
      <PageTitle
        kicker={
          view ? `MODEL DIFFICULTY · GW${view.gws[0] ?? "?"}–${view.gws[view.gws.length - 1] ?? "?"}` : "MODEL DIFFICULTY"
        }
        title="FIXTURES"
        right={
          <span className="microlabel hidden items-center gap-1.5 sm:inline-flex">
            EASY
            {RAMP.map((c) => (
              <span key={c} className="h-2.5 w-2.5 rounded-sm" style={{ background: c }} />
            ))}
            HARD
          </span>
        }
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded border border-hairline p-0.5" role="tablist" aria-label="Difficulty lens">
          {(
            [
              ["overall", "OVERALL"],
              ["attack", "ATTACK xG"],
              ["defence", "DEFENCE CS%"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              role="tab"
              aria-selected={lens === value}
              onClick={() => setLens(value)}
              className={`hit relative rounded px-2.5 py-1.5 font-mono text-[10.5px] tracking-[0.12em] ${
                lens === value ? "bg-raised text-ink" : "text-ink-dim hover:text-ink-mid"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <select
          value={fromGw ?? ""}
          onChange={(e) => setFromGw(e.target.value === "" ? null : Number(e.target.value))}
          aria-label="From gameweek"
          className="rounded border border-hairline bg-surface px-2 py-1.5 font-mono text-[11px] text-ink-mid outline-none focus:border-ink-dim"
        >
          <option value="">FROM GW</option>
          {gwOptions.map((g) => (
            <option key={g} value={g}>
              GW{g}
            </option>
          ))}
        </select>
        <select
          value={toGw ?? ""}
          onChange={(e) => setToGw(e.target.value === "" ? null : Number(e.target.value))}
          aria-label="To gameweek"
          className="rounded border border-hairline bg-surface px-2 py-1.5 font-mono text-[11px] text-ink-mid outline-none focus:border-ink-dim"
        >
          <option value="">TO GW</option>
          {gwOptions.map((g) => (
            <option key={g} value={g}>
              GW{g}
            </option>
          ))}
        </select>
        <label className="microlabel hit inline-flex cursor-pointer items-center gap-1.5">
          <input
            type="checkbox"
            checked={sortByEase}
            onChange={(e) => setSortByEase(e.target.checked)}
            className="accent-pitch"
          />
          SORT BY BEST RUN
        </label>
      </div>

      <Card>
        {!view || isLoading ? (
          <div className="space-y-2 p-4">
            {Array.from({ length: 10 }, (_, i) => (
              <Skeleton key={i} className="h-8" />
            ))}
          </div>
        ) : view.rows.length === 0 ? (
          <EmptyState
            title="NO FIXTURES YET"
            detail="The daily model run publishes the fixture outlook alongside predictions."
          />
        ) : (
          <ScrollFade fade="surface">
            <table className="w-full border-collapse text-left" style={{ minWidth: 80 + view.gws.length * 58 }}>
              <thead>
                <tr className="border-b border-hairline">
                  <th className="microlabel sticky left-0 z-10 bg-surface px-3 py-2">CLUB</th>
                  {view.gws.map((g) => (
                    <th key={g} className="microlabel px-1 py-2 text-center">
                      GW{g}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {view.rows.map(({ row }) => (
                  <tr key={row.code} className="border-b border-hairline-soft last:border-b-0">
                    <td className="sticky left-0 z-10 bg-surface px-3 py-1 font-mono text-[11.5px] font-medium text-ink">
                      {row.short}
                    </td>
                    {view.gws.map((g) => {
                      const cells = row.cells.get(g) ?? [];
                      if (cells.length === 0) {
                        return (
                          <td key={g} className="px-0.5 py-1 text-center">
                            <span
                              className="block rounded-sm bg-raised px-1 py-1.5 font-mono text-[9.5px] text-ink-dim"
                              title={`GW${g}: blank — no fixture`}
                            >
                              —
                            </span>
                          </td>
                        );
                      }
                      return (
                        <td key={g} className="px-0.5 py-1 text-center">
                          <span className="flex flex-col gap-0.5">
                            {cells.map((c, i) => {
                              const s = cellScore(c, lens);
                              const bucket = s === null ? 2 : bucketOf(s, view.scores);
                              const label = c.home
                                ? c.opponent.toUpperCase()
                                : c.opponent.toLowerCase();
                              const bits = [
                                `${row.short} ${c.home ? "v" : "@"} ${c.opponent} (GW${c.gw})`,
                                c.xgFor !== null ? `xG for ${c.xgFor.toFixed(2)}` : null,
                                c.pCs !== null ? `CS ${(c.pCs * 100).toFixed(0)}%` : null,
                                c.oddsBlended ? "odds-blended" : "model only",
                              ].filter(Boolean);
                              return (
                                <span
                                  key={i}
                                  className="block rounded-sm px-1 py-1.5 font-mono text-[9.5px] font-semibold"
                                  style={{ background: RAMP[bucket], color: RAMP_INK[bucket] }}
                                  title={bits.join(" · ")}
                                >
                                  {label}
                                </span>
                              );
                            })}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollFade>
        )}
      </Card>

      <p className="microlabel mt-3">
        UPPERCASE = HOME · lowercase = away · DIFFICULTY FROM THE SAME MODEL THAT POWERS xP
        (DIXON-COLES + ODDS BLEND), NOT THE OFFICIAL FDR
      </p>
    </div>
  );
}
