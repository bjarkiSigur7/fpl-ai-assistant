"use client";

/**
 * RATING STATUS BAR — the build console above the squad board: n/15 selected,
 * budget remaining from £100.0m, live legality hints, SCAN SCREENSHOT (local
 * mode only) / LOAD MODEL XV / CLEAR conveniences, and the EVALUATE button
 * (display type — the page's one action).
 */

import { useRef } from "react";

import { LockGlyph } from "./Gate";
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
      className={`hit relative rounded border border-hairline px-2.5 py-1.5 font-mono text-[10px] tracking-[0.12em] transition-colors ${
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
  modelLocked = false,
  onScanFile,
  scanning = false,
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
  /** Public build, no key yet: the shortcut opens the inline key form. */
  modelLocked?: boolean;
  /** Local mode only — undefined hides the SCAN SCREENSHOT button entirely. */
  onScanFile?: (file: File) => void;
  scanning?: boolean;
}) {
  const overBudget = budgetLeft < 0;
  const ready = legal && !evaluating && !scanning;
  const fileRef = useRef<HTMLInputElement | null>(null);
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
          {onScanFile ? (
            <>
              <input
                ref={fileRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) onScanFile(file);
                  e.target.value = ""; // allow re-picking the same screenshot
                }}
              />
              <button
                onClick={() => fileRef.current?.click()}
                disabled={scanning || evaluating || loadingModel}
                className={`hit relative rounded px-2.5 py-1.5 font-mono text-[10px] font-semibold tracking-[0.12em] transition-colors ${
                  scanning || evaluating || loadingModel
                    ? "cursor-not-allowed bg-pitch/60 text-[#0d0d0d]"
                    : "bg-pitch text-[#0d0d0d] hover:bg-pitch-bright"
                } ${scanning ? "tickpulse" : ""}`}
              >
                {scanning ? "SCANNING…" : "SCAN SCREENSHOT"}
              </button>
            </>
          ) : null}
          {modelLocked ? (
            <button
              onClick={onLoadModel}
              disabled={evaluating || scanning}
              title="Keyholders-only — enter the desk key to load the model's squad"
              className="hit relative inline-flex items-center gap-1.5 rounded border border-hairline px-2.5 py-1.5 font-mono text-[10px] tracking-[0.12em] text-ink-dim transition-colors hover:border-ink-dim hover:bg-raised hover:text-ink-mid"
            >
              <LockGlyph />
              MODEL XV
            </button>
          ) : (
            <MiniButton
              label={loadingModel ? "LOADING…" : "LOAD MODEL XV"}
              onClick={onLoadModel}
              busy={loadingModel}
              disabled={evaluating || scanning}
            />
          )}
          <MiniButton
            label="CLEAR"
            onClick={onClear}
            disabled={nSelected === 0 || evaluating || loadingModel || scanning}
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
