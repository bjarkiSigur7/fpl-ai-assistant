"use client";

/**
 * Players: the predictions explorer. Dense sortable/filterable table of next-GW xP
 * (with component sparkbars), horizon xP and availability risk; search, position
 * tabs and club filter. Row click opens the player detail page.
 */

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { OFFLINE_HINT } from "@/lib/api";
import { moneyBare, xp2 } from "@/lib/format";
import { usePlayerIndex, type IndexedPlayer } from "@/lib/playerIndex";
import { POSITIONS, type Position } from "@/lib/types";
import { Sparkbar, SparkbarLegend } from "@/components/charts";
import { ScrollFade } from "@/components/ScrollFade";
import { Card, EmptyState, PageTitle, RiskMark, Skeleton } from "@/components/ui";

type SortKey = "name" | "pos" | "club" | "price" | "xp" | "xph" | "q0" | "own" | "diff";

interface Sort {
  key: SortKey;
  dir: 1 | -1;
}

const DEFAULT_DIR: Record<SortKey, 1 | -1> = {
  name: 1,
  pos: 1,
  club: 1,
  price: -1,
  xp: -1,
  xph: -1,
  q0: 1,
  own: -1,
  diff: -1,
};

/** <sm sort control labels — the column headers are hidden with the table. */
const SORT_LABEL: Record<SortKey, string> = {
  xp: "xP NEXT GW",
  xph: "Σ HORIZON xP",
  price: "PRICE",
  name: "NAME",
  club: "CLUB",
  pos: "POSITION",
  q0: "RISK q0",
  own: "OWNERSHIP %",
  diff: "DIFFERENTIAL",
};

/** Differential score: horizon xP over an ownership floor — the gold quadrant
 * (high xP, low owned) ranks first; template picks sink. */
function differentialScore(p: IndexedPlayer): number {
  return p.xpHorizon / Math.max(p.ownership ?? 100, 2);
}

/** Penalty-duty badge (P1 = first-choice taker), from the official bootstrap. */
function PenBadge({ order }: { order: number | null }) {
  if (order === null) return null;
  return (
    <span
      className="ml-1.5 rounded bg-raised px-1 py-0.5 font-mono text-[9px] text-pitch-bright"
      title={`Penalty taker #${order}`}
    >
      P{order}
    </span>
  );
}

/** Availability tooltip: the FPL news line + the model's parsed return GW. */
function riskTitle(p: IndexedPlayer): string | undefined {
  if (!p.news && p.returnGw === null) return undefined;
  const bits = [p.news, p.returnGw !== null ? `model expects back GW${p.returnGw}` : null];
  return bits.filter(Boolean).join(" · ");
}

function compare(a: IndexedPlayer, b: IndexedPlayer, sort: Sort, gw: number): number {
  const get = (p: IndexedPlayer): string | number => {
    switch (sort.key) {
      case "name":
        return p.web_name.toLowerCase();
      case "pos":
        return POSITIONS.indexOf(p.position);
      case "club":
        return p.team_short;
      case "price":
        return p.price;
      case "xp":
        return p.xpByGw.get(gw) ?? 0;
      case "xph":
        return p.xpHorizon;
      case "q0":
        return p.q0ByGw.get(gw) ?? 1;
      case "own":
        return p.ownership ?? -1;
      case "diff":
        return differentialScore(p);
    }
  };
  const va = get(a);
  const vb = get(b);
  if (va < vb) return -1 * sort.dir;
  if (va > vb) return 1 * sort.dir;
  return 0;
}

function Th({
  label,
  k,
  sort,
  onSort,
  className = "",
}: {
  label: string;
  k: SortKey;
  sort: Sort;
  onSort: (k: SortKey) => void;
  className?: string;
}) {
  const active = sort.key === k;
  return (
    <th
      className={`whitespace-nowrap px-3 py-2 ${className}`}
      aria-sort={active ? (sort.dir === 1 ? "ascending" : "descending") : undefined}
    >
      <button
        onClick={() => onSort(k)}
        className={`microlabel hit relative inline-flex items-center gap-1 ${active ? "!text-ink" : "hover:text-ink-mid"}`}
      >
        {label}
        <span aria-hidden="true" className={active ? "text-ink" : "text-transparent"}>
          {active && sort.dir === 1 ? "▲" : "▼"}
        </span>
      </button>
    </th>
  );
}

export default function PlayersPage() {
  const router = useRouter();
  const { index, isLoading, error } = usePlayerIndex();
  const [q, setQ] = useState("");
  const [pos, setPos] = useState<Position | "ALL">("ALL");
  const [club, setClub] = useState<string>("ALL");
  const [sort, setSort] = useState<Sort>({ key: "xp", dir: -1 });
  const [diffOnly, setDiffOnly] = useState(false);

  const gw = index?.nextGw ?? 0;
  const hasOwnership = useMemo(
    () => (index ? [...index.byCode.values()].some((p) => p.ownership !== null) : false),
    [index],
  );

  const rows = useMemo(() => {
    if (!index) return [];
    const needle = q.trim().toLowerCase();
    return [...index.byCode.values()]
      .filter((p) => pos === "ALL" || p.position === pos)
      .filter((p) => club === "ALL" || p.team_short === club)
      .filter((p) => !diffOnly || ((p.ownership ?? 100) < 15 && (p.q0ByGw.get(gw) ?? 1) < 0.4))
      .filter(
        (p) =>
          needle === "" ||
          p.web_name.toLowerCase().includes(needle) ||
          p.team_short.toLowerCase().includes(needle),
      )
      .sort((a, b) => compare(a, b, diffOnly ? { key: "diff", dir: -1 } : sort, gw));
  }, [index, q, pos, club, sort, gw, diffOnly]);

  const maxXp = useMemo(
    () => (index ? Math.max(...[...index.byCode.values()].map((p) => p.xpByGw.get(gw) ?? 0), 1) : 1),
    [index, gw],
  );

  const clubs = useMemo(
    () => (index ? [...index.teams.values()].map((t) => t.short_name).sort() : []),
    [index],
  );

  const onSort = (k: SortKey) =>
    setSort((s) => (s.key === k ? { key: k, dir: s.dir === 1 ? -1 : 1 } : { key: k, dir: DEFAULT_DIR[k] }));

  if (error) {
    return (
      <EmptyState title="NO PREDICTIONS FEED" detail={OFFLINE_HINT} />
    );
  }

  return (
    <div>
      <PageTitle
        kicker={index ? `SEASON ${index.season}-${String((index.season + 1) % 100).padStart(2, "0")} · GW${gw} NEXT` : "PREDICTIONS"}
        title="PLAYERS"
        right={<SparkbarLegend />}
      />

      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded border border-hairline p-0.5" role="tablist" aria-label="Position">
          {(["ALL", ...POSITIONS] as const).map((p) => (
            <button
              key={p}
              role="tab"
              aria-selected={pos === p}
              onClick={() => setPos(p)}
              className={`hit relative rounded px-2.5 py-1.5 font-mono text-[10.5px] tracking-[0.12em] ${
                pos === p ? "bg-raised text-ink" : "text-ink-dim hover:text-ink-mid"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
        {hasOwnership ? (
          <button
            onClick={() => setDiffOnly((v) => !v)}
            aria-pressed={diffOnly}
            title="Under 15% owned, fit, ranked by horizon xP per ownership point"
            className={`hit relative rounded border px-2.5 py-1.5 font-mono text-[10.5px] tracking-[0.12em] ${
              diffOnly
                ? "border-pitch bg-raised text-pitch-bright"
                : "border-hairline text-ink-dim hover:text-ink-mid"
            }`}
          >
            ◆ DIFFERENTIALS
          </button>
        ) : null}
        <select
          value={club}
          onChange={(e) => setClub(e.target.value)}
          aria-label="Club filter"
          className="rounded border border-hairline bg-surface px-2 py-1.5 font-mono text-[11px] text-ink-mid outline-none focus:border-ink-dim"
        >
          <option value="ALL">ALL CLUBS</option>
          {clubs.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        {/* <sm: the column-header sort lives in the hidden table — surface it */}
        <select
          value={sort.key}
          onChange={(e) => {
            const k = e.target.value as SortKey;
            setSort({ key: k, dir: DEFAULT_DIR[k] });
          }}
          aria-label="Sort players"
          className="rounded border border-hairline bg-surface px-2 py-1.5 font-mono text-[11px] text-ink-mid outline-none focus:border-ink-dim sm:hidden"
        >
          {(Object.keys(SORT_LABEL) as SortKey[]).map((k) => (
            <option key={k} value={k}>
              SORT — {SORT_LABEL[k]}
            </option>
          ))}
        </select>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Search player or club…"
          aria-label="Search players"
          className="min-w-44 flex-1 rounded border border-hairline bg-surface px-3 py-1.5 text-[13px] text-ink placeholder:text-ink-dim focus:border-ink-dim focus:outline-none sm:max-w-72"
        />
        <span className="microlabel tnum ml-auto">
          {index ? `${rows.length} PLAYERS` : "…"}
        </span>
      </div>

      <Card>
        {!index || isLoading ? (
          <div className="space-y-2 p-4">
            {Array.from({ length: 12 }, (_, i) => (
              <Skeleton key={i} className="h-8" />
            ))}
          </div>
        ) : (
          <>
            {/* <sm: compact mobile rows — name + club/pos/price small, xP
                prominent right, sparkbar spanning the row. Tap navigates. */}
            <ul className="sm:hidden">
              {rows.map((p) => {
                const row = p.firstRow;
                const xpGw = p.xpByGw.get(gw) ?? 0;
                return (
                  <li key={p.player_code} className="border-b border-hairline-soft last:border-b-0">
                    <button
                      onClick={() => router.push(`/players/${p.player_code}`)}
                      className="block w-full px-3.5 py-2.5 text-left transition-colors hover:bg-raised"
                    >
                      <div className="flex items-baseline justify-between gap-3">
                        <span className="min-w-0 truncate text-[13.5px] font-medium text-ink">
                          {p.web_name}
                          {row.n_fixtures > 1 ? (
                            <span className="ml-1.5 rounded bg-raised px-1 py-0.5 font-mono text-[9px] text-ink-mid">
                              DGW
                            </span>
                          ) : null}
                          <PenBadge order={p.penOrder} />
                        </span>
                        <span className="shrink-0 font-mono text-[15px] font-semibold tnum text-ink">
                          {xp2(xpGw)}
                          <span className="ml-1 text-[9px] font-normal tracking-[0.08em] text-ink-dim">
                            xP
                          </span>
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-center justify-between gap-3 font-mono text-[10px] text-ink-dim">
                        <span className="tnum">
                          {p.team_short} · {p.position} · £{moneyBare(p.price)} · Σ{" "}
                          {p.xpHorizon.toFixed(1)}
                          {p.ownership !== null ? ` · ${p.ownership.toFixed(1)}%` : ""}
                        </span>
                        <span title={riskTitle(p)}>
                          <RiskMark q0={p.q0ByGw.get(gw) ?? 0} />
                        </span>
                      </div>
                      <div className="mt-1.5">
                        <Sparkbar components={row} max={maxXp} />
                      </div>
                    </button>
                  </li>
                );
              })}
            </ul>
            {/* ≥sm: the full table inside a fading scroll container */}
            <ScrollFade fade="surface" className="hidden sm:block">
              <table className="w-full min-w-[760px] border-collapse text-left">
            <thead>
              <tr className="border-b border-hairline">
                <Th label="PLAYER" k="name" sort={sort} onSort={onSort} />
                <Th label="POS" k="pos" sort={sort} onSort={onSort} />
                <Th label="CLUB" k="club" sort={sort} onSort={onSort} />
                <Th label="£" k="price" sort={sort} onSort={onSort} className="text-right" />
                <Th label={`xP GW${gw}`} k="xp" sort={sort} onSort={onSort} className="text-right" />
                <th className="microlabel px-3 py-2">COMPONENTS</th>
                <Th label={`Σ xP ${index.gws.length}GW`} k="xph" sort={sort} onSort={onSort} className="text-right" />
                <Th label="OWN%" k="own" sort={sort} onSort={onSort} className="text-right" />
                <Th label="RISK q0" k="q0" sort={sort} onSort={onSort} />
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => {
                const row = p.firstRow;
                const xpGw = p.xpByGw.get(gw) ?? 0;
                return (
                  <tr
                    key={p.player_code}
                    onClick={() => router.push(`/players/${p.player_code}`)}
                    className="cursor-pointer border-b border-hairline-soft transition-colors last:border-b-0 hover:bg-raised"
                  >
                    <td className="px-3 py-2 text-[13px] font-medium text-ink">
                      {p.web_name}
                      {row.n_fixtures > 1 ? (
                        <span className="ml-1.5 rounded bg-raised px-1 py-0.5 font-mono text-[9px] text-ink-mid">
                          DGW
                        </span>
                      ) : null}
                      <PenBadge order={p.penOrder} />
                    </td>
                    <td className="px-3 py-2 font-mono text-[11px] text-ink-dim">{p.position}</td>
                    <td className="px-3 py-2 font-mono text-[11px] text-ink-dim">{p.team_short}</td>
                    <td className="px-3 py-2 text-right font-mono text-[12px] tnum text-ink-mid">
                      {moneyBare(p.price)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-[12.5px] font-semibold tnum text-ink">
                      {xp2(xpGw)}
                    </td>
                    <td className="w-36 px-3 py-2">
                      <Sparkbar components={row} max={maxXp} />
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-[12px] tnum text-ink-mid">
                      {p.xpHorizon.toFixed(1)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-[12px] tnum text-ink-dim">
                      {p.ownership !== null ? p.ownership.toFixed(1) : "—"}
                    </td>
                    <td className="px-3 py-2">
                      <span title={riskTitle(p)}>
                        <RiskMark q0={p.q0ByGw.get(gw) ?? 0} />
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
              </table>
            </ScrollFade>
          </>
        )}
      </Card>
    </div>
  );
}
