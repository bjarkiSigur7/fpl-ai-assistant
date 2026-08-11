"use client";

/**
 * AI RATING — rate any 15. Official FPL squad-selection anatomy in desk chrome:
 * player pool panel on the left, the 15-slot pitch + build console on the right,
 * and the EVALUATE moment on top — POST /api/rate-team scores the squad 0-100
 * against the model optimum (dream team) and the cheapest-legal floor.
 */

import { useCallback, useMemo, useState } from "react";
import { OFFLINE_HINT, fetchDreamTeam, rateTeam } from "@/lib/api";
import { seasonLabel } from "@/lib/format";
import { usePlayerIndex, type IndexedPlayer } from "@/lib/playerIndex";
import {
  POSITIONS,
  RateTeamValidationError,
  type Position,
  type RateTeamResponse,
} from "@/lib/types";
import {
  RatingPitch,
  SQUAD_QUOTA,
  type RatingHighlight,
  type RatingSlotPlayer,
} from "@/components/RatingPitch";
import { InlineUnlock } from "@/components/Gate";
import { useGate } from "@/lib/gate";
import { RatingMobileBar } from "@/components/RatingMobileBar";
import { RatingPlayerPanel } from "@/components/RatingPlayerPanel";
import { RatingResult } from "@/components/RatingResult";
import { RatingStatusBar } from "@/components/RatingStatusBar";
import { Card, CardHead, EmptyState, PageTitle, Skeleton } from "@/components/ui";

const BUDGET = 1000; // £100.0m in 0.1m units
const CLUB_LIMIT = 3;

interface SquadShape {
  players: IndexedPlayer[];
  posCount: Record<Position, number>;
  clubCount: Map<number, number>;
  spent: number;
}

function shapeOf(codes: number[], get: (code: number) => IndexedPlayer | undefined): SquadShape {
  const players = codes
    .map((c) => get(c))
    .filter((p): p is IndexedPlayer => p !== undefined);
  const posCount: Record<Position, number> = { GKP: 0, DEF: 0, MID: 0, FWD: 0 };
  const clubCount = new Map<number, number>();
  let spent = 0;
  for (const p of players) {
    posCount[p.position] += 1;
    clubCount.set(p.team_code, (clubCount.get(p.team_code) ?? 0) + 1);
    spent += p.price;
  }
  return { players, posCount, clubCount, spent };
}

export default function RatingPage() {
  const { index, isLoading, error } = usePlayerIndex();
  const [codes, setCodes] = useState<number[]>([]);
  const [result, setResult] = useState<RateTeamResponse | null>(null);
  const [violations, setViolations] = useState<string[] | null>(null);
  const [engineDown, setEngineDown] = useState(false);
  const [evaluating, setEvaluating] = useState(false);
  const [loadingModel, setLoadingModel] = useState(false);
  const [modelHint, setModelHint] = useState<string | null>(null);
  const { locked } = useGate();
  const [showUnlock, setShowUnlock] = useState(false);

  const gw = index?.nextGw ?? 0;
  const selected = useMemo(() => new Set(codes), [codes]);
  const shape = useMemo(
    () => shapeOf(codes, (c) => index?.get(c)),
    [codes, index],
  );
  const budgetLeft = BUDGET - shape.spent;

  /** Any verdict is stale the moment the squad changes. */
  const clearOutcome = useCallback(() => {
    setResult(null);
    setViolations(null);
    setEngineDown(false);
    setModelHint(null);
  }, []);

  const canAdd = useCallback(
    (p: IndexedPlayer): string | null => {
      if (codes.length >= 15) return "SQUAD FULL — 15/15";
      if (shape.posCount[p.position] >= SQUAD_QUOTA[p.position]) {
        return `QUOTA FULL — ${SQUAD_QUOTA[p.position]} ${p.position} MAX`;
      }
      if ((shape.clubCount.get(p.team_code) ?? 0) >= CLUB_LIMIT) {
        return `CLUB LIMIT — ${CLUB_LIMIT} PER CLUB`;
      }
      if (p.price > budgetLeft) return "OVER BUDGET";
      return null;
    },
    [codes.length, shape, budgetLeft],
  );

  const toggle = useCallback(
    (code: number) => {
      clearOutcome();
      setCodes((cur) =>
        cur.includes(code) ? cur.filter((c) => c !== code) : [...cur, code],
      );
    },
    [clearOutcome],
  );

  const remove = useCallback(
    (code: number) => {
      clearOutcome();
      setCodes((cur) => cur.filter((c) => c !== code));
    },
    [clearOutcome],
  );

  // Client-side legality mirror of the backend's rules (the EVALUATE gate).
  const legal =
    codes.length === 15 &&
    shape.players.length === 15 &&
    POSITIONS.every((pos) => shape.posCount[pos] === SQUAD_QUOTA[pos]) &&
    [...shape.clubCount.values()].every((n) => n <= CLUB_LIMIT) &&
    budgetLeft >= 0;

  const hints = useMemo(() => {
    const out: string[] = [];
    for (const pos of POSITIONS) {
      const missing = SQUAD_QUOTA[pos] - shape.posCount[pos];
      if (missing > 0) out.push(`NEED ${missing} ${pos}`);
    }
    if (budgetLeft < 0) out.push(`OVER BUDGET £${(-budgetLeft / 10).toFixed(1)}m`);
    for (const [teamCode, n] of shape.clubCount) {
      if (n > CLUB_LIMIT) {
        const short = index?.teams.get(teamCode)?.short_name ?? teamCode;
        out.push(`CLUB LIMIT — ${n} FROM ${short}`);
      }
    }
    if (out.length === 0 && codes.length < 15) out.push(`PICK ${15 - codes.length} MORE`);
    return out;
  }, [shape, budgetLeft, codes.length, index]);

  const evaluate = useCallback(async () => {
    setEvaluating(true);
    setResult(null);
    setViolations(null);
    setEngineDown(false);
    try {
      const res = await rateTeam({ player_codes: codes, season: null, gw: null });
      setResult(res);
    } catch (err) {
      if (err instanceof RateTeamValidationError) setViolations(err.detail);
      else setEngineDown(true);
    } finally {
      setEvaluating(false);
    }
  }, [codes]);

  const doLoadModel = useCallback(async () => {
    setShowUnlock(false);
    setLoadingModel(true);
    clearOutcome();
    try {
      const dream = await fetchDreamTeam();
      setCodes(dream.squad.slice(0, 15));
    } catch {
      setModelHint("MODEL XV UNAVAILABLE — dream_team.json is not being served.");
    } finally {
      setLoadingModel(false);
    }
  }, [clearOutcome]);

  const loadModelXv = useCallback(async () => {
    if (locked) {
      // Keyholders-only shortcut: surface the inline key form instead.
      setShowUnlock(true);
      return;
    }
    await doLoadModel();
  }, [locked, doLoadModel]);

  const clearSquad = useCallback(() => {
    clearOutcome();
    setCodes([]);
  }, [clearOutcome]);

  const slots = useMemo(() => {
    const out = {} as Record<Position, (RatingSlotPlayer | null)[]>;
    for (const pos of POSITIONS) {
      const filled: RatingSlotPlayer[] = shape.players
        .filter((p) => p.position === pos)
        .map((p) => ({
          code: p.player_code,
          name: p.web_name,
          team_short: p.team_short,
          price: p.price,
          xp: p.xpByGw.get(gw) ?? null,
        }));
      out[pos] = [
        ...filled,
        ...Array.from({ length: Math.max(SQUAD_QUOTA[pos] - filled.length, 0) }, () => null),
      ];
    }
    return out;
  }, [shape.players, gw]);

  const highlight: RatingHighlight | null = result
    ? { bestXi: new Set(result.best_xi_gw1), captain: result.suggested_captain }
    : null;

  if (error) {
    return (
      <EmptyState title="NO PLAYER POOL" detail={OFFLINE_HINT} />
    );
  }

  return (
    <div>
      <PageTitle
        kicker={
          index
            ? `SEASON ${seasonLabel(index.season)} · WINDOW GW${index.gws[0]}–GW${index.gws[index.gws.length - 1]}`
            : "RATE ANY 15"
        }
        title="AI RATING"
        right={<span className="microlabel">MODEL-SCORED 0–100 VS THE OPTIMUM</span>}
      />

      {result ? (
        <div className="mb-5">
          <RatingResult result={result} nameOf={(c) => index?.name(c) ?? `#${c}`} />
        </div>
      ) : null}

      {!index || isLoading ? (
        <div className="grid gap-5 lg:grid-cols-2">
          <Skeleton className="h-[560px]" />
          <div className="space-y-5">
            <Skeleton className="h-36" />
            <Skeleton className="h-[480px]" />
          </div>
        </div>
      ) : (
        <div className="grid items-start gap-5 lg:grid-cols-2">
          <div className="order-2 min-w-0 lg:order-1">
            <RatingPlayerPanel
              index={index}
              gw={gw}
              selected={selected}
              canAdd={canAdd}
              onToggle={toggle}
            />
          </div>
          <div className="order-1 min-w-0 space-y-5 lg:order-2">
            <RatingStatusBar
              nSelected={codes.length}
              budgetLeft={budgetLeft}
              hints={hints}
              legal={legal}
              evaluating={evaluating}
              loadingModel={loadingModel}
              onEvaluate={evaluate}
              onLoadModel={loadModelXv}
              onClear={clearSquad}
              modelLocked={locked}
            />
            {locked && showUnlock ? (
              <InlineUnlock onUnlocked={() => void doLoadModel()} />
            ) : null}
            {modelHint ? (
              <p className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-ink-dim">
                <span aria-hidden="true" className="mr-1.5 text-warn">
                  ▲
                </span>
                {modelHint}
              </p>
            ) : null}
            {violations ? (
              <Card>
                <CardHead
                  label="SQUAD REJECTED — 422"
                  right={`${violations.length} RULE${violations.length === 1 ? "" : "S"}`}
                />
                <ul className="space-y-1.5 px-4 py-3">
                  {violations.map((v) => (
                    <li key={v} className="flex gap-2 text-[12.5px] leading-relaxed text-ink-mid">
                      <span aria-hidden="true" className="mt-[1px] font-mono text-crit">
                        ■
                      </span>
                      {v}
                    </li>
                  ))}
                </ul>
              </Card>
            ) : null}
            {engineDown ? (
              <EmptyState
                title="RATING ENGINE OFFLINE"
                detail={OFFLINE_HINT}
                action={
                  <button
                    onClick={() => void evaluate()}
                    className="microlabel underline decoration-hairline underline-offset-4 hover:text-ink"
                  >
                    RETRY →
                  </button>
                }
              />
            ) : null}
            <RatingPitch
              slots={slots}
              highlight={highlight}
              onRemove={remove}
              right={
                result
                  ? `BEST XI ${result.formation_gw1} · GW${result.from_gw}`
                  : `${codes.length}/15 SELECTED`
              }
            />
          </div>
        </div>
      )}
      {index && !isLoading ? (
        <>
          {/* clearance for the fixed mobile action bar */}
          <div aria-hidden="true" className="h-16 lg:hidden" />
          <RatingMobileBar
            nSelected={codes.length}
            budgetLeft={budgetLeft}
            legal={legal}
            evaluating={evaluating}
            hasResult={result !== null}
            onEvaluate={() => void evaluate()}
          />
        </>
      ) : null}
    </div>
  );
}
