"use client";

/**
 * Dashboard: THE VERDICT (hero), the two pitches (dream team vs my team /
 * initial build), and the top-movers rail with the xP trend chart.
 */

import Link from "next/link";
import { useMemo, useState } from "react";
import { OFFLINE_HINT, useCaptaincy, useRecommendation } from "@/lib/api";
import { xp1 } from "@/lib/format";
import { usePlayerIndex, type PlayerIndex } from "@/lib/playerIndex";
import type { Recommendation } from "@/lib/types";
import { LineChart } from "@/components/charts";
import { Pitch, PitchSkeleton, type PitchPlayer } from "@/components/Pitch";
import { Card, CardHead, EmptyState, RiskMark, Skeleton } from "@/components/ui";
import { VerdictCard, VerdictSkeleton } from "@/components/VerdictCard";
import { useEntryId } from "@/lib/useEntryId";

function toPitchPlayers(codes: number[], index: PlayerIndex, gw: number): PitchPlayer[] {
  return codes.map((code) => {
    const p = index.get(code);
    return {
      code,
      name: p?.web_name ?? `#${code}`,
      position: p?.position ?? "MID",
      price: p?.price ?? null,
      xp: p?.xpByGw.get(gw) ?? null,
    };
  });
}

function MoverRow({
  index,
  code,
  metric,
  metricTitle,
}: {
  index: PlayerIndex;
  code: number;
  metric: string;
  metricTitle: string;
}) {
  const p = index.get(code)!;
  return (
    <Link
      href={`/players/${code}`}
      className="flex items-center gap-2 rounded px-2 py-1.5 transition-colors hover:bg-raised"
    >
      <span className="w-10 font-mono text-[10px] text-ink-dim">{p.team_short}</span>
      <span className="flex-1 truncate text-[12.5px] font-medium text-ink">{p.web_name}</span>
      <RiskMark q0={p.q0ByGw.get(index.nextGw) ?? 0} />
      <span className="w-12 text-right font-mono text-[12px] tnum text-ink" title={metricTitle}>
        {metric}
      </span>
    </Link>
  );
}

function MoversCard({ index }: { index: PlayerIndex }) {
  const gw = index.nextGw;
  const players = useMemo(() => [...index.byCode.values()], [index]);
  const topXp = useMemo(
    () =>
      [...players]
        .filter((p) => (p.xpByGw.get(gw) ?? 0) > 0)
        .sort((a, b) => (b.xpByGw.get(gw) ?? 0) - (a.xpByGw.get(gw) ?? 0))
        .slice(0, 6),
    [players, gw],
  );
  const value = useMemo(
    () =>
      [...players]
        .filter((p) => (p.q0ByGw.get(gw) ?? 1) < 0.4)
        .sort((a, b) => b.xpHorizon / b.price - a.xpHorizon / a.price)
        .slice(0, 6),
    [players, gw],
  );

  return (
    <Card>
      <CardHead label="TOP MOVERS" right={`GW${gw}`} />
      <div className="px-2 py-2">
        <div className="microlabel px-2 pb-1 pt-1">HIGHEST xP — GW{gw}</div>
        {topXp.map((p) => (
          <MoverRow
            key={p.player_code}
            index={index}
            code={p.player_code}
            metric={xp1(p.xpByGw.get(gw) ?? 0)}
            metricTitle="expected points this GW"
          />
        ))}
        <div className="microlabel px-2 pb-1 pt-3">VALUE — xP PER £m (HORIZON)</div>
        {value.map((p) => (
          <MoverRow
            key={p.player_code}
            index={index}
            code={p.player_code}
            metric={(p.xpHorizon / (p.price / 10)).toFixed(2)}
            metricTitle="horizon xP per £1.0m"
          />
        ))}
      </div>
    </Card>
  );
}

/**
 * Distributional captaincy comparison — what the xP point estimate hides.
 * 2,000 joint Monte Carlo rollouts per candidate (shared fixture scorelines,
 * so same-match candidates are correlated); P(BEST) splits ties evenly.
 */
function CaptaincyCard({ index }: { index: PlayerIndex }) {
  const { data } = useCaptaincy();
  if (!data || data.length === 0) return null;
  const rows = data.slice(0, 5);
  const maxBest = Math.max(...rows.map((r) => r.p_best), 0.01);
  return (
    <Card>
      <CardHead label="CAPTAINCY — MC DISTRIBUTIONS" right={`GW${rows[0].gw}`} />
      <div className="px-3 pb-3 pt-1">
        <div className="microlabel grid grid-cols-[1fr_3rem_3rem_3rem] gap-2 px-1 pb-1 text-right">
          <span className="text-left">{rows[0].n_draws.toLocaleString()} ROLLOUTS</span>
          <span title="P(scores 10+ points)">HAUL</span>
          <span title="P(scores 2 or fewer)">BLANK</span>
          <span title="P(best captain of the candidate set; ties split)">BEST</span>
        </div>
        {rows.map((r) => {
          const p = index.get(r.player_code);
          return (
            <div
              key={r.player_code}
              className="grid grid-cols-[1fr_3rem_3rem_3rem] items-center gap-2 rounded px-1 py-1 hover:bg-raised"
              title={
                r.p_beats_top !== null
                  ? `P(outscores the top pick): ${(r.p_beats_top * 100).toFixed(0)}%`
                  : "The model's top captaincy pick by xP"
              }
            >
              <span className="relative min-w-0">
                <span
                  aria-hidden="true"
                  className="absolute inset-y-0 left-0 rounded-sm bg-raised"
                  style={{ width: `${(r.p_best / maxBest) * 100}%` }}
                />
                <span className="relative flex items-baseline gap-1.5 truncate pl-1 text-[12.5px] font-medium text-ink">
                  {p?.web_name ?? `#${r.player_code}`}
                  <span className="font-mono text-[10px] tnum text-ink-dim">
                    {xp1(r.xp)}xP
                  </span>
                </span>
              </span>
              <span className="text-right font-mono text-[11px] tnum text-ink-mid">
                {(r.p_haul * 100).toFixed(0)}%
              </span>
              <span className="text-right font-mono text-[11px] tnum text-ink-dim">
                {(r.p_blank * 100).toFixed(0)}%
              </span>
              <span className="text-right font-mono text-[11.5px] font-semibold tnum text-ink">
                {(r.p_best * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>
    </Card>
  );
}

function TrendCard({ index }: { index: PlayerIndex }) {
  const top4 = useMemo(
    () => [...index.byCode.values()].sort((a, b) => b.xpHorizon - a.xpHorizon).slice(0, 4),
    [index],
  );
  return (
    <Card>
      <CardHead
        label="xP CURVE — TOP ASSETS"
        right={`GW${index.gws[0]}–${index.gws[index.gws.length - 1]}`}
      />
      <div className="px-3 pb-3 pt-2">
        <LineChart
          series={top4.map((p) => ({
            label: p.web_name,
            values: index.gws.map((gw) => p.xpByGw.get(gw) ?? null),
          }))}
          xLabels={index.gws.map((gw) => `GW${gw}`)}
          unit=" xP"
          vw={380}
          vh={230}
        />
      </div>
    </Card>
  );
}

function Pitches({
  rec,
  index,
  entryScoped,
}: {
  rec: Recommendation;
  index: PlayerIndex;
  entryScoped: boolean;
}) {
  const [view, setView] = useState<"dream" | "mine">("dream");
  const gw = rec.gw;
  const dream = rec.dream_team;

  const dreamPitch = dream ? (
    <Pitch
      label="DREAM TEAM — FRESH £100m"
      right={`${dream.formation ?? ""} · £${(dream.total_cost / 10).toFixed(1)}m · ${xp1(dream.expected_points)}xP`}
      lineup={toPitchPlayers(dream.lineup, index, gw)}
      bench={toPitchPlayers(dream.bench_order, index, gw)}
      captain={dream.captain}
      vice={dream.vice}
    />
  ) : (
    <EmptyState
      title="NO BENCHMARK"
      detail="The dream-team solve has not run for this gameweek yet."
    />
  );

  const mineLabel = entryScoped
    ? "MY TEAM — OPTIMAL XI"
    : rec.action === "initial-squad"
      ? "INITIAL SQUAD — HORIZON BUILD"
      : "CAMPAIGN SQUAD — OPTIMAL XI"; // entry-scoped shared verdict (static build)
  const minePitch =
    rec.lineup.length > 0 ? (
      <Pitch
        label={mineLabel}
        right={`${rec.formation ?? ""} · ${xp1(rec.expected_points)}xP`}
        lineup={toPitchPlayers(rec.lineup, index, gw)}
        bench={toPitchPlayers(rec.bench_order, index, gw)}
        captain={rec.captain}
        vice={rec.vice}
      />
    ) : (
      <EmptyState
        title="NO SQUAD YET"
        detail="Configure your FPL entry ID in settings to get a squad-aware verdict, transfers and chip timing."
        action={
          <Link
            href="/settings"
            className="microlabel underline decoration-hairline underline-offset-4 hover:text-ink"
          >
            OPEN SETTINGS →
          </Link>
        }
      />
    );

  return (
    <>
      {/* mobile: toggle between the two pitches */}
      <div className="lg:hidden">
        <div
          className="mb-3 inline-flex rounded border border-hairline p-0.5"
          role="tablist"
          aria-label="Pitch view"
        >
          {(["dream", "mine"] as const).map((v) => (
            <button
              key={v}
              role="tab"
              aria-selected={view === v}
              onClick={() => setView(v)}
              className={`hit relative rounded px-3 py-1.5 font-mono text-[10.5px] tracking-[0.12em] ${
                view === v ? "bg-raised text-ink" : "text-ink-dim"
              }`}
            >
              {v === "dream" ? "DREAM XI" : entryScoped ? "MY XI" : "BUILD"}
            </button>
          ))}
        </div>
        {view === "dream" ? dreamPitch : minePitch}
      </div>
      {/* desktop: side by side */}
      <div className="hidden lg:block">{dreamPitch}</div>
      <div className="hidden lg:block">{minePitch}</div>
    </>
  );
}

export default function DashboardPage() {
  const [entryId] = useEntryId();
  const { data: rec, error, isLoading, entryScoped, mutate } = useRecommendation(entryId);
  const { index } = usePlayerIndex();

  if (error) {
    return (
      <EmptyState
        title="DESK OFFLINE"
        detail={OFFLINE_HINT}
        action={
          <button
            onClick={() => void mutate()}
            className="microlabel underline decoration-hairline underline-offset-4 hover:text-ink"
          >
            RETRY →
          </button>
        }
      />
    );
  }

  if (isLoading || !rec || !index) {
    return (
      <div className="space-y-5">
        <VerdictSkeleton />
        <div className="grid gap-5 lg:grid-cols-3">
          <PitchSkeleton label="DREAM TEAM" />
          <div className="hidden lg:block">
            <PitchSkeleton label="MY TEAM" />
          </div>
          <div className="space-y-5">
            <Skeleton className="h-64" />
            <Skeleton className="h-48" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-5">
      <VerdictCard rec={rec} />
      <div className="grid gap-5 lg:grid-cols-3">
        <Pitches rec={rec} index={index} entryScoped={entryScoped} />
        <div className="space-y-5">
          <CaptaincyCard index={index} />
          <MoversCard index={index} />
          <TrendCard index={index} />
        </div>
      </div>
    </div>
  );
}
