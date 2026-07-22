"use client";

/**
 * Planner: the multi-GW plan timeline (per-GW cards: transfers, captain, chip,
 * expected points), chip EV curves as small multiples, and plan-stability
 * support bars from the noise re-solves.
 */

import { useMemo } from "react";
import { useChipCurves, useRecommendation } from "@/lib/api";
import { moneyBare, xp1 } from "@/lib/format";
import { usePlayerIndex, type PlayerIndex } from "@/lib/playerIndex";
import { chipName, type ChipCurvePoint, type GwPlan } from "@/lib/types";
import { MiniBarChart } from "@/components/charts";
import { Card, CardHead, EmptyState, PageTitle, Skeleton } from "@/components/ui";
import { useEntryId } from "@/lib/useEntryId";

function GwCard({
  plan,
  index,
  first,
}: {
  plan: GwPlan;
  index: PlayerIndex;
  first: boolean;
}) {
  const moves = plan.transfers_in.map((codeIn, i) => ({
    in: codeIn,
    out: plan.transfers_out[i] ?? null,
  }));
  return (
    <div
      className={`w-56 shrink-0 rounded-md border bg-surface ${
        first ? "border-ink-dim/60" : "border-hairline"
      }`}
    >
      <div className="flex items-center justify-between border-b border-hairline-soft px-3 py-2">
        <span className="display text-[15px]">GW{plan.gw}</span>
        {plan.chip ? (
          <span className="rounded bg-pitch px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-[0.08em] text-[#0d0d0d]">
            {chipName(plan.chip)}
          </span>
        ) : first ? (
          <span className="microlabel">NEXT</span>
        ) : null}
      </div>
      <div className="space-y-2 px-3 py-2.5">
        {moves.length === 0 ? (
          <div className="font-mono text-[11px] text-ink-dim">NO TRANSFERS — ROLL</div>
        ) : (
          moves.map((m, i) => (
            <div key={i} className="space-y-0.5 font-mono text-[11px]">
              <div className="text-pitch-bright">▲ IN&nbsp;&nbsp;{index.name(m.in)}</div>
              {m.out !== null ? <div className="text-neg">▼ OUT {index.name(m.out)}</div> : null}
            </div>
          ))
        )}
        {plan.hit_points > 0 ? (
          <div className="font-mono text-[11px] text-neg">HIT −{plan.hit_points}</div>
        ) : null}
        <div className="border-t border-hairline-soft pt-2 font-mono text-[11px] text-ink-mid">
          <span className="mr-1 inline-flex h-[15px] w-[15px] items-center justify-center rounded-full bg-pitch text-[9px] font-bold text-[#0d0d0d]">
            C
          </span>
          {index.name(plan.captain)}
        </div>
        <div className="flex items-baseline justify-between">
          <span className="microlabel">EXPECTED</span>
          <span className="font-mono text-[17px] font-semibold tnum text-ink">
            {xp1(plan.expected_points)}
          </span>
        </div>
        <div className="flex justify-between font-mono text-[9.5px] text-ink-dim">
          <span>BANK £{moneyBare(plan.bank)}m</span>
          <span>{plan.free_transfers === null ? "FT ∞" : `FT ${plan.free_transfers}`}</span>
        </div>
      </div>
    </div>
  );
}

function ChipMultiples({
  curves,
  gws,
}: {
  curves: ChipCurvePoint[];
  gws: number[];
}) {
  const chips = useMemo(() => [...new Set(curves.map((c) => c.chip))], [curves]);
  if (chips.length === 0) {
    return (
      <p className="px-4 py-6 text-[13px] text-ink-dim">
        No chips left to evaluate — the inventory is spent.
      </p>
    );
  }
  return (
    <div className="grid gap-4 p-4 sm:grid-cols-2 xl:grid-cols-3">
      {chips.map((chip) => {
        const rows = curves.filter((c) => c.chip === chip);
        const values = gws.map(
          (gw) => rows.find((r) => r.gw === gw)?.delta_vs_no_chip ?? null,
        );
        const best = values.reduce(
          (acc, v, i) => (v !== null && (acc.v === null || v > acc.v) ? { i, v } : acc),
          { i: -1, v: null as number | null },
        );
        return (
          <div key={chip} className="rounded border border-hairline-soft bg-raised/40 p-3">
            <div className="mb-1 flex items-baseline justify-between">
              <span className="text-[12.5px] font-semibold text-ink">{chipName(chip)}</span>
              <span className="microlabel">{chip.toUpperCase()}</span>
            </div>
            <MiniBarChart
              values={values}
              xLabels={gws.map((gw) => `${gw}`)}
              highlight={best.i >= 0 ? best.i : undefined}
              unit=" xP"
            />
            <div className="mt-1 font-mono text-[10px] text-ink-dim">
              {best.v !== null ? (
                <>
                  BEST GW{gws[best.i]} · {best.v >= 0 ? "+" : "−"}
                  {Math.abs(best.v).toFixed(1)} xP VS NO CHIP
                </>
              ) : (
                "OUTSIDE WINDOW"
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function StabilityBars({
  entries,
  index,
}: {
  entries: { move: string; support_pct: number }[];
  index: PlayerIndex;
}) {
  if (entries.length === 0) {
    return (
      <p className="px-4 py-6 text-[13px] text-ink-dim">
        No noise re-solves for this plan — stability runs with a configured squad.
      </p>
    );
  }
  const sorted = [...entries].sort((a, b) => b.support_pct - a.support_pct);
  return (
    <div className="space-y-2.5 p-4">
      {sorted.map((e) => (
        <div key={e.move} className="flex items-center gap-3">
          <span className="w-40 shrink-0 truncate text-[12px] text-ink-mid">
            {index.humanizeMove(e.move)}
          </span>
          <div className="h-[7px] flex-1 overflow-hidden rounded-full bg-raised">
            <div
              className="h-full rounded-full bg-s1"
              style={{ width: `${Math.min(e.support_pct, 100)}%` }}
            />
          </div>
          <span className="w-11 text-right font-mono text-[11px] tnum text-ink">
            {Math.round(e.support_pct)}%
          </span>
        </div>
      ))}
      <p className="pt-1 font-mono text-[9.5px] uppercase tracking-[0.1em] text-ink-dim">
        % of noise re-solves agreeing with each this-GW move
      </p>
    </div>
  );
}

export default function PlannerPage() {
  const [entryId] = useEntryId();
  const { data: rec, error, isLoading } = useRecommendation(entryId);
  const { data: chipData } = useChipCurves(entryId);
  const { index } = usePlayerIndex();

  if (error) {
    return (
      <EmptyState
        title="NO PLAN FEED"
        detail="The planner needs the recommendation endpoint. Start the backend or run with NEXT_PUBLIC_MOCK=1."
      />
    );
  }

  if (isLoading || !rec || !index) {
    return (
      <div className="space-y-5">
        <Skeleton className="h-10 w-64" />
        <div className="flex gap-3 overflow-hidden">
          {Array.from({ length: 5 }, (_, i) => (
            <Skeleton key={i} className="h-56 w-56 shrink-0" />
          ))}
        </div>
        <Skeleton className="h-64" />
        <Skeleton className="h-48" />
      </div>
    );
  }

  const gws = rec.plan.gws ?? [];
  const horizonGws = gws.map((g) => g.gw);
  const solveMeta = [
    rec.plan.status ? `HiGHS ${rec.plan.status.toUpperCase()}` : null,
    rec.plan.solve_seconds !== undefined ? `${rec.plan.solve_seconds.toFixed(1)}s` : null,
    rec.plan.gap !== undefined ? `GAP ${(rec.plan.gap * 100).toFixed(2)}%` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="space-y-5">
      <PageTitle
        kicker={`${entryId !== null ? `ENTRY ${entryId}` : "INITIAL BUILD"} · OBJECTIVE ${xp1(rec.objective)} xP`}
        title="PLANNER"
        right={<span className="microlabel">{solveMeta}</span>}
      />

      <Card>
        <CardHead
          label="MULTI-GW PLAN"
          right={horizonGws.length ? `GW${horizonGws[0]}–${horizonGws[horizonGws.length - 1]}` : undefined}
        />
        {gws.length ? (
          <div className="deskscroll flex gap-3 overflow-x-auto p-4">
            {gws.map((g, i) => (
              <GwCard key={g.gw} plan={g} index={index} first={i === 0} />
            ))}
          </div>
        ) : (
          <p className="px-4 py-6 text-[13px] text-ink-dim">
            No solved plan attached to this recommendation.
          </p>
        )}
      </Card>

      <Card>
        <CardHead label="CHIP EV CURVES — FORCED-CHIP RE-SOLVES" right="Δ xP VS NO CHIP" />
        {chipData ? (
          <ChipMultiples
            curves={chipData.curves}
            gws={horizonGws.length ? horizonGws : [...new Set(chipData.curves.map((c) => c.gw))].sort((a, b) => a - b)}
          />
        ) : (
          <div className="grid gap-4 p-4 sm:grid-cols-2 xl:grid-cols-3">
            {Array.from({ length: 3 }, (_, i) => (
              <Skeleton key={i} className="h-40" />
            ))}
          </div>
        )}
      </Card>

      <Card>
        <CardHead label="PLAN STABILITY — NOISE RE-SOLVES" right={`n=30`} />
        <StabilityBars entries={rec.stability} index={index} />
      </Card>
    </div>
  );
}
