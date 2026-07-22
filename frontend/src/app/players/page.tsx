"use client";

/**
 * Players: the predictions explorer. Dense sortable/filterable table of next-GW xP
 * (with component sparkbars), horizon xP and availability risk; search, position
 * tabs and club filter. Row click opens the player detail page.
 */

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { moneyBare, xp2 } from "@/lib/format";
import { usePlayerIndex, type IndexedPlayer } from "@/lib/playerIndex";
import { POSITIONS, type Position } from "@/lib/types";
import { Sparkbar, SparkbarLegend } from "@/components/charts";
import { Card, EmptyState, PageTitle, RiskMark, Skeleton } from "@/components/ui";

type SortKey = "name" | "pos" | "club" | "price" | "xp" | "xph" | "q0";

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
};

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
        className={`microlabel inline-flex items-center gap-1 ${active ? "!text-ink" : "hover:text-ink-mid"}`}
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

  const gw = index?.nextGw ?? 0;

  const rows = useMemo(() => {
    if (!index) return [];
    const needle = q.trim().toLowerCase();
    return [...index.byCode.values()]
      .filter((p) => pos === "ALL" || p.position === pos)
      .filter((p) => club === "ALL" || p.team_short === club)
      .filter(
        (p) =>
          needle === "" ||
          p.web_name.toLowerCase().includes(needle) ||
          p.team_short.toLowerCase().includes(needle),
      )
      .sort((a, b) => compare(a, b, sort, gw));
  }, [index, q, pos, club, sort, gw]);

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
      <EmptyState
        title="NO PREDICTIONS FEED"
        detail="The predictions endpoint is not answering. Run `fplai predict` on the backend, or set NEXT_PUBLIC_MOCK=1 for the standalone demo."
      />
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
              className={`rounded px-2.5 py-1.5 font-mono text-[10.5px] tracking-[0.12em] ${
                pos === p ? "bg-raised text-ink" : "text-ink-dim hover:text-ink-mid"
              }`}
            >
              {p}
            </button>
          ))}
        </div>
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

      <Card className="overflow-x-auto deskscroll">
        {!index || isLoading ? (
          <div className="space-y-2 p-4">
            {Array.from({ length: 12 }, (_, i) => (
              <Skeleton key={i} className="h-8" />
            ))}
          </div>
        ) : (
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
                    <td className="px-3 py-2">
                      <RiskMark q0={p.q0ByGw.get(gw) ?? 0} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
