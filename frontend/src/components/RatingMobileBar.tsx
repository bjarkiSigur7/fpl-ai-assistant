"use client";

/**
 * RATING MOBILE BAR — sticky bottom console for the AI RATING page below lg,
 * where the player pool sits under the pitch: live n/15 + budget readouts and
 * the page's one action stay in reach while building from the pool. The action
 * slot walks the flow: SCAN SCREENSHOT (primary) while the squad is short and
 * scanning is available, EVALUATE once 15 are in, SCROLL TO RESULT after a
 * verdict. Backdrop-blurred over the page and safe-area aware; hidden at lg+
 * (two-column layout keeps the build console in view there).
 */

import { useRef } from "react";

export function RatingMobileBar({
  nSelected,
  budgetLeft,
  legal,
  evaluating,
  hasResult,
  onEvaluate,
  onScanFile,
  scanning = false,
}: {
  nSelected: number;
  /** 0.1m units remaining of the £100.0m budget (negative when over). */
  budgetLeft: number;
  legal: boolean;
  evaluating: boolean;
  hasResult: boolean;
  onEvaluate: () => void;
  /** Undefined (scan unavailable) keeps EVALUATE in the action slot throughout. */
  onScanFile?: (file: File) => void;
  scanning?: boolean;
}) {
  const overBudget = budgetLeft < 0;
  const ready = legal && !evaluating && !scanning;
  const fileRef = useRef<HTMLInputElement | null>(null);

  const scrollToResult = () => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    document
      .getElementById("rating-result")
      ?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  };

  return (
    <div
      className="fixed inset-x-0 bottom-0 z-40 border-t border-hairline bg-bg/85 backdrop-blur-md lg:hidden"
      style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
    >
      <div className="mx-auto flex w-full max-w-[1280px] items-center justify-between gap-3 px-4 py-2 sm:px-6">
        <div className="flex flex-col gap-y-0.5 whitespace-nowrap font-mono text-[10px] uppercase tracking-[0.08em] text-ink-dim">
          <span>
            SQUAD{" "}
            <span className="tnum text-[13px] font-semibold text-ink">{nSelected}</span>
            /15
          </span>
          <span>
            BUDGET{" "}
            <span
              className={`tnum text-[13px] font-semibold ${overBudget ? "text-neg" : "text-ink"}`}
            >
              £{(budgetLeft / 10).toFixed(1)}m
            </span>{" "}
            LEFT
          </span>
        </div>
        {hasResult ? (
          <button
            onClick={scrollToResult}
            className="display min-h-11 whitespace-nowrap rounded-md border border-hairline bg-raised px-4 text-[13px] tracking-[0.02em] transition-colors hover:bg-surface"
          >
            ↑ SCROLL TO RESULT
          </button>
        ) : nSelected < 15 && onScanFile ? (
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
              disabled={scanning || evaluating}
              className={`display min-h-11 whitespace-nowrap rounded-md px-6 text-[16px] tracking-[0.02em] transition-colors ${
                scanning || evaluating
                  ? "cursor-not-allowed bg-pitch/60 text-[#0d0d0d]"
                  : "bg-pitch text-[#0d0d0d] hover:bg-pitch-bright"
              } ${scanning ? "tickpulse" : ""}`}
            >
              {scanning ? "SCANNING…" : "SCAN SCREENSHOT"}
            </button>
          </>
        ) : (
          <button
            onClick={onEvaluate}
            disabled={!ready}
            className={`display min-h-11 whitespace-nowrap rounded-md px-6 text-[16px] tracking-[0.02em] transition-colors ${
              ready
                ? "bg-pitch text-[#0d0d0d] hover:bg-pitch-bright"
                : "cursor-not-allowed border border-hairline bg-raised !text-ink-dim"
            } ${evaluating ? "tickpulse" : ""}`}
          >
            {evaluating ? "SCORING…" : "EVALUATE"}
          </button>
        )}
      </div>
    </div>
  );
}
