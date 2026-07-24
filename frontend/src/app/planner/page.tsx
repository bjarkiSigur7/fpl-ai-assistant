"use client";

/**
 * Planner: the multi-GW plan timeline (per-GW cards: transfers, captain, chip,
 * expected points), chip EV curves as small multiples, and plan-stability
 * support bars from the noise re-solves.
 */

import { useMemo, useState } from "react";
import { OFFLINE_HINT, useChipCurves, useRecommendation } from "@/lib/api";
import { moneyBare, xp1 } from "@/lib/format";
import { usePlayerIndex, type PlayerIndex } from "@/lib/playerIndex";
import { chipName, type ChipAdvice, type ChipCurvePoint, type GwPlan } from "@/lib/types";
import { MiniBarChart } from "@/components/charts";
import { ScrollFade } from "@/components/ScrollFade";
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
      className={`w-56 shrink-0 snap-start scroll-ml-4 rounded-md border bg-surface ${
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

/** "hold vs play" verdict pill with the simulation confidence wording. */
function VerdictPill({ advice }: { advice: ChipAdvice }) {
  const conf = advice.confidence
    ? ` · ${(advice.confidence === "medium" ? "med" : advice.confidence).toUpperCase()} CONF`
    : "";
  if (advice.verdict === "play") {
    return (
      <span className="rounded bg-pitch px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-[0.08em] text-[#0d0d0d]">
        PLAY NOW{conf}
      </span>
    );
  }
  return (
    <span className="rounded border border-hairline bg-raised px-1.5 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-[0.08em] text-ink-mid">
      HOLD{conf}
    </span>
  );
}

function ChipMultiples({
  curves,
  gws,
  advice,
}: {
  curves: ChipCurvePoint[];
  gws: number[];
  advice: ChipAdvice[];
}) {
  const chips = useMemo(() => [...new Set(curves.map((c) => c.chip))], [curves]);
  const adviceOf = useMemo(
    () => new Map(advice.map((a) => [a.chip, a])),
    [advice],
  );
  if (chips.length === 0) {
    return (
      <p className="px-4 py-6 text-[13px] text-ink-dim">
        No chips left to evaluate — the inventory is spent.
      </p>
    );
  }
  const hasSim = curves.some(
    (c) => c.sd != null || c.p_best_week != null || c.p_beats_hold != null,
  );
  const assumptions =
    advice.find((a) => a.assumptions && a.assumptions.length)?.assumptions ?? null;
  const nRollouts =
    advice.find((a) => a.n_rollouts != null)?.n_rollouts ??
    curves.find((c) => c.n_rollouts != null)?.n_rollouts ??
    null;
  return (
    <div>
      <div className="grid gap-4 p-4 sm:grid-cols-2 xl:grid-cols-3">
        {chips.map((chip) => {
          const rows = curves.filter((c) => c.chip === chip);
          const rowAt = (gw: number) => rows.find((r) => r.gw === gw);
          const values = gws.map((gw) => rowAt(gw)?.delta_vs_no_chip ?? null);
          const bands = gws.map((gw) => rowAt(gw)?.sd ?? null);
          const pBest = gws.map((gw) => rowAt(gw)?.p_best_week ?? null);
          const best = values.reduce(
            (acc, v, i) => (v !== null && (acc.v === null || v > acc.v) ? { i, v } : acc),
            { i: -1, v: null as number | null },
          );
          const chipAdvice = adviceOf.get(chip);
          const simBacked = chipAdvice?.p_beats_hold != null;
          return (
            <div key={chip} className="rounded border border-hairline-soft bg-raised/40 p-3">
              <div className="mb-1 flex items-baseline justify-between gap-2">
                <span className="truncate text-[12.5px] font-semibold text-ink">
                  {chipName(chip)}{" "}
                  <span className="microlabel">{chip.toUpperCase()}</span>
                </span>
                {chipAdvice ? <VerdictPill advice={chipAdvice} /> : null}
              </div>
              <MiniBarChart
                values={values}
                xLabels={gws.map((gw) => `${gw}`)}
                highlight={best.i >= 0 ? best.i : undefined}
                unit=" xP"
                bands={bands}
                pBest={pBest}
              />
              <div className="mt-1 space-y-0.5 font-mono text-[10px] text-ink-dim">
                <div>
                  {best.v !== null ? (
                    <>
                      BEST GW{gws[best.i]} · {best.v >= 0 ? "+" : "−"}
                      {Math.abs(best.v).toFixed(1)} xP VS NO CHIP
                    </>
                  ) : (
                    "OUTSIDE WINDOW"
                  )}
                </div>
                {simBacked && chipAdvice ? (
                  <div>
                    BEATS HOLD NOW {Math.round((chipAdvice.p_beats_hold ?? 0) * 100)}%
                    {chipAdvice.recommended_gw != null ? (
                      <>
                        {" "}· BEST WINDOW GW{chipAdvice.recommended_gw}
                        {chipAdvice.p_best_week_reco != null
                          ? ` (P ${chipAdvice.p_best_week_reco.toFixed(2)})`
                          : ""}
                      </>
                    ) : null}
                  </div>
                ) : null}
              </div>
            </div>
          );
        })}
      </div>
      {hasSim ? (
        <div className="border-t border-hairline-soft px-4 py-2.5">
          <p className="font-mono text-[9.5px] uppercase tracking-[0.1em] text-ink-dim">
            BARS Δ xP VS NO CHIP · WHISKERS ±1 SD ACROSS ROLLOUTS · MARKER ROW P(BEST WEEK)
            {nRollouts != null ? ` · N=${nRollouts.toLocaleString("en-GB")}` : ""}
          </p>
          {assumptions ? (
            <>
              {/* <sm: collapsible so the sim small print doesn't dominate */}
              <details className="group mt-1 sm:hidden">
                <summary className="hit relative inline-flex cursor-pointer list-none items-center gap-1.5 font-mono text-[9.5px] uppercase tracking-[0.1em] text-ink-dim [&::-webkit-details-marker]:hidden">
                  <span
                    aria-hidden="true"
                    className="inline-block text-[8px] transition-transform group-open:rotate-90 motion-reduce:transition-none"
                  >
                    ▶
                  </span>
                  SIM ASSUMPTIONS
                </summary>
                <p className="mt-1 text-[11px] leading-relaxed text-ink-dim">
                  Assumptions: {assumptions.join(" · ")}.
                </p>
              </details>
              <p className="mt-1 hidden text-[11px] leading-relaxed text-ink-dim sm:block">
                Assumptions: {assumptions.join(" · ")}.
              </p>
            </>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** <sm: only the top entries show until expanded; ≥sm always shows the lot. */
const STABILITY_TOP = 15;

function StabilityBars({
  entries,
  index,
}: {
  entries: { move: string; support_pct: number }[];
  index: PlayerIndex;
}) {
  const [showAll, setShowAll] = useState(false);
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
      {sorted.map((e, i) => (
        <div
          key={e.move}
          className={`items-center gap-3 ${
            i >= STABILITY_TOP && !showAll ? "hidden sm:flex" : "flex"
          }`}
        >
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
      {sorted.length > STABILITY_TOP ? (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="microlabel hit relative underline decoration-hairline underline-offset-4 hover:text-ink sm:hidden"
        >
          {showAll ? `SHOW TOP ${STABILITY_TOP}` : `SHOW ALL (${sorted.length})`}
        </button>
      ) : null}
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
      <EmptyState title="NO PLAN FEED" detail={OFFLINE_HINT} />
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
          <ScrollFade fade="surface" arrow innerClassName="snap-x snap-mandatory">
            <div className="flex gap-3 p-4">
              {gws.map((g, i) => (
                <GwCard key={g.gw} plan={g} index={index} first={i === 0} />
              ))}
            </div>
          </ScrollFade>
        ) : (
          <p className="px-4 py-6 text-[13px] text-ink-dim">
            No solved plan attached to this recommendation.
          </p>
        )}
      </Card>

      <Card>
        <CardHead
          label={
            chipData?.curves.some((c) => c.p_beats_hold != null)
              ? "CHIP PLANNER — MONTE CARLO SEASON SIM"
              : "CHIP EV CURVES — FORCED-CHIP RE-SOLVES"
          }
          right="Δ xP VS NO CHIP"
        />
        {chipData ? (
          <ChipMultiples
            curves={chipData.curves}
            gws={horizonGws.length ? horizonGws : [...new Set(chipData.curves.map((c) => c.gw))].sort((a, b) => a - b)}
            advice={rec.chip_advice}
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
