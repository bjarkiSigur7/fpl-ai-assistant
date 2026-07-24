"use client";

/**
 * RATING STATUS BAR — the build console above the squad board: n/15 selected,
 * budget remaining from £100.0m, live legality hints, LOAD MODEL XV / CLEAR
 * conveniences, and the EVALUATE button (display type — the page's one action).
 */

import { Card, StatusBadge } from "./ui";

function MiniButton({
  label,
  onClick,
  disabled,
  busy,
}: {
  label: string;
  onClick: () => void;
  disabled?: boolean;
  busy?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || busy}
      className={`rounded border border-hairline px-2.5 py-1.5 font-mono text-[10px] tracking-[0.12em] transition-colors ${
        disabled || busy
          ? "cursor-not-allowed text-ink-dim opacity-50"
          : "text-ink-mid hover:border-ink-dim hover:bg-raised hover:text-ink"
      } ${busy ? "tickpulse" : ""}`}
    >
      {label}
    </button>
  );
}

export function RatingStatusBar({
  nSelected,
  budgetLeft,
  hints,
  legal,
  evaluating,
  loadingModel,
  onEvaluate,
  onLoadModel,
  onClear,
}: {
  nSelected: number;
  /** 0.1m units remaining of the £100.0m budget (negative when over). */
  budgetLeft: number;
  /** Live legality hints, e.g. ["NEED 2 DEF", "NEED 1 FWD"] or a club-limit note. */
  hints: string[];
  /** True when the squad is exactly 15 and passes every rule client-side. */
  legal: boolean;
  evaluating: boolean;
  loadingModel: boolean;
  onEvaluate: () => void;
  onLoadModel: () => void;
  onClear: () => void;
}) {
  const overBudget = budgetLeft < 0;
  const ready = legal && !evaluating;
  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 px-4 pt-3">
        <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1 font-mono text-[11px] text-ink-dim">
          <span>
            SQUAD{" "}
            <span className="tnum text-[16px] font-semibold text-ink">{nSelected}</span>
            /15
          </span>
          <span>
            BUDGET LEFT{" "}
            <span
              className={`tnum text-[16px] font-semibold ${overBudget ? "text-neg" : "text-ink"}`}
            >
              £{(budgetLeft / 10).toFixed(1)}m
            </span>{" "}
            OF £100.0m
          </span>
        </div>
        <div className="flex gap-2">
          <MiniButton
            label={loadingModel ? "LOADING…" : "LOAD MODEL XV"}
            onClick={onLoadModel}
            busy={loadingModel}
            disabled={evaluating}
          />
          <MiniButton
            label="CLEAR"
            onClick={onClear}
            disabled={nSelected === 0 || evaluating || loadingModel}
          />
        </div>
      </div>
      <div className="px-4 pb-1.5 pt-2">
        {legal ? (
          <StatusBadge kind="good" label="SQUAD LEGAL — READY TO SCORE" />
        ) : (
          <div className="flex flex-wrap gap-x-3 gap-y-1">
            {hints.map((h) => (
              <StatusBadge key={h} kind={h.startsWith("NEED") ? "warn" : "crit"} label={h} />
            ))}
          </div>
        )}
      </div>
      <div className="px-3 pb-3 pt-1.5">
        <button
          onClick={onEvaluate}
          disabled={!ready}
          className={`display w-full rounded-md py-3.5 text-[21px] tracking-[0.02em] transition-colors ${
            ready
              ? "bg-pitch text-[#0d0d0d] hover:bg-pitch-bright"
              : "cursor-not-allowed border border-hairline bg-raised !text-ink-dim"
          } ${evaluating ? "tickpulse" : ""}`}
        >
          {evaluating ? "SCORING…" : "EVALUATE"}
        </button>
      </div>
    </Card>
  );
}
