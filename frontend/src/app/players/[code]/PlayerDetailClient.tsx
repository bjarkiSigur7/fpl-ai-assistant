"use client";

/**
 * Player detail: identity header, season history chart (FPL points bars + xG
 * overlay line), and the upcoming-fixture xP breakdown bars.
 */

import Link from "next/link";
import { useParams } from "next/navigation";
import { useMemo } from "react";
import { usePlayerDetail } from "@/lib/api";
import { moneyBare, shortUtc, xp2 } from "@/lib/format";
import { usePlayerIndex } from "@/lib/playerIndex";
import type { FixturePrediction } from "@/lib/types";
import { HistoryChart, Sparkbar, SparkbarLegend } from "@/components/charts";
import { Card, CardHead, EmptyState, RiskMark, Skeleton } from "@/components/ui";

function StatTile({
  label,
  value,
  sub,
  className = "",
}: {
  label: string;
  value: string;
  sub?: string;
  className?: string;
}) {
  return (
    <div className={`rounded border border-hairline-soft bg-raised px-3 py-2.5 ${className}`}>
      <div className="microlabel mb-1">{label}</div>
      <div className="font-mono text-[18px] font-semibold tnum text-ink">{value}</div>
      {sub ? <div className="font-mono text-[10px] text-ink-dim">{sub}</div> : null}
    </div>
  );
}

function FixtureRow({
  fx,
  opponentShort,
  max,
}: {
  fx: FixturePrediction;
  /** null when the feed has no per-fixture opponent (static bundle GW rows). */
  opponentShort: string | null;
  max: number;
}) {
  return (
    <div className="flex items-center gap-3 px-4 py-2.5">
      <div className="w-28 shrink-0">
        <div className="font-mono text-[11.5px] text-ink">
          GW{fx.gw}
          {opponentShort !== null ? ` · ${opponentShort} (${fx.was_home ? "H" : "A"})` : ""}
        </div>
        <div className="font-mono text-[9.5px] text-ink-dim">
          {fx.kickoff_utc ? shortUtc(fx.kickoff_utc) : "TBC"}
        </div>
      </div>
      <div className="flex-1">
        <Sparkbar components={fx} max={max} height={10} />
      </div>
      <div className="w-14 text-right font-mono text-[13px] font-semibold tnum text-ink">
        {xp2(fx.xp)}
      </div>
    </div>
  );
}

export function PlayerDetailClient() {
  const params = useParams<{ code: string }>();
  const code = Number(params.code);
  const { data: player, error, isLoading } = usePlayerDetail(Number.isFinite(code) ? code : null);
  const { index } = usePlayerIndex();

  const maxFixtureXp = useMemo(
    () => (player ? Math.max(...player.upcoming.map((f) => f.xp), 1) : 1),
    [player],
  );

  const totals = useMemo(() => {
    if (!player) return null;
    const pts = player.history.reduce((s, h) => s + h.total_points, 0);
    const minutes = player.history.reduce((s, h) => s + h.minutes, 0);
    const goals = player.history.reduce((s, h) => s + h.goals_scored, 0);
    const assists = player.history.reduce((s, h) => s + h.assists, 0);
    const nextXp = player.upcoming.length
      ? player.upcoming.filter((f) => f.gw === player.upcoming[0].gw).reduce((s, f) => s + f.xp, 0)
      : 0;
    const horizonXp = player.upcoming.reduce((s, f) => s + f.xp, 0);
    const nextQ0 = player.upcoming[0]?.q0 ?? 0;
    return { pts, minutes, goals, assists, nextXp, horizonXp, nextQ0 };
  }, [player]);

  if (error) {
    return (
      <EmptyState
        title="UNKNOWN PLAYER"
        detail="No player with this code in the current predictions window."
        action={
          <Link href="/players" className="microlabel underline decoration-hairline underline-offset-4 hover:text-ink">
            ← BACK TO PLAYERS
          </Link>
        }
      />
    );
  }

  if (isLoading || !player || !totals) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-24 w-full max-w-xl" />
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} className="h-20" />
          ))}
        </div>
        <Skeleton className="h-64" />
        <Skeleton className="h-52" />
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <div>
        <Link href="/players" className="microlabel hit relative inline-block hover:text-ink">
          ← PLAYERS
        </Link>
        <div className="mt-2 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="display text-[34px] sm:text-[44px]">{player.web_name}</h1>
            <div className="mt-1.5 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim">
              {[player.first_name, player.second_name].filter(Boolean).join(" ") ||
                player.web_name}{" "}
              · {player.position} · {player.team_name ?? player.team_short_name ?? player.team_code}{" "}
              · CODE {player.player_code}
            </div>
          </div>
          <RiskMark q0={totals.nextQ0} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        <StatTile label="PRICE" value={`£${moneyBare(player.price)}m`} />
        <StatTile
          label={`NEXT GW xP`}
          value={xp2(totals.nextXp)}
          sub={player.upcoming[0] ? `GW${player.upcoming[0].gw}` : undefined}
        />
        <StatTile label="HORIZON xP" value={totals.horizonXp.toFixed(1)} sub={`${player.upcoming.length} FIXTURES`} />
        <StatTile label="SEASON PTS" value={String(totals.pts)} sub={`${totals.minutes} MIN`} />
        {/* spans the 2-col mobile grid so the odd fifth tile leaves no gap */}
        <StatTile
          label="G + A"
          value={`${totals.goals} + ${totals.assists}`}
          className="col-span-2 sm:col-span-1"
        />
      </div>

      <Card>
        <CardHead
          label={`SEASON HISTORY — ${player.season}-${String((player.season + 1) % 100).padStart(2, "0")}`}
          right="POINTS + xG"
        />
        <div className="px-3 pb-3 pt-2">
          {player.history.length ? (
            <HistoryChart
              data={player.history.map((h) => ({
                gw: h.gw,
                points: h.total_points,
                xg: h.xg,
                minutes: h.minutes,
              }))}
            />
          ) : (
            <p className="px-2 py-6 text-[13px] text-ink-dim">
              No appearances recorded this season yet.
            </p>
          )}
        </div>
      </Card>

      <Card>
        {/* <sm: title row then legend row (the one-line head wraps at 390px) */}
        <header className="flex flex-col items-start gap-1.5 border-b border-hairline-soft px-4 py-2.5 sm:flex-row sm:items-center sm:justify-between sm:gap-3">
          <span className="microlabel">UPCOMING — PER-FIXTURE xP BREAKDOWN</span>
          <span className="microlabel text-ink-dim">
            <SparkbarLegend />
          </span>
        </header>
        {player.upcoming.length ? (
          <div className="divide-y divide-hairline-soft">
            {player.upcoming.map((fx) => (
              <FixtureRow
                key={`${fx.gw}-${fx.fpl_fixture_id}`}
                fx={fx}
                opponentShort={
                  fx.opponent_short_name ??
                  index?.teams.get(fx.opponent_code)?.short_name ??
                  (fx.opponent_code !== 0 ? String(fx.opponent_code) : null)
                }
                max={maxFixtureXp}
              />
            ))}
          </div>
        ) : (
          <p className="px-4 py-6 text-[13px] text-ink-dim">
            No upcoming fixtures in the prediction window — between seasons the desk waits
            for the 2026-27 fixture list.
          </p>
        )}
      </Card>
    </div>
  );
}
