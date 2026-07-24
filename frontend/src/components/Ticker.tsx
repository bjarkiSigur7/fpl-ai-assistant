"use client";

/**
 * THE TICKER — thin top strip: live deadline countdown, data freshness, model
 * version, season state. Refreshes state every 30s; the countdown ticks every second.
 */

import { MOCK, useDeskState } from "@/lib/api";
import { useNow } from "@/lib/useNow";
import {
  PROVISIONAL_GW1_DEADLINE_UTC,
  countdownString,
  countdownTo,
  hoursSince,
  seasonSlash,
  shortUtc,
} from "@/lib/format";
import { StatusBadge, type StatusKind } from "./ui";

function Sep() {
  return <span className="hidden text-ink-dim/50 sm:inline">·</span>;
}

export function Ticker() {
  const { data: state, error } = useDeskState();
  const now = useNow();

  const live = state?.is_live_2026_27 ?? false;
  const deadlineIso = state?.next_deadline_utc ?? PROVISIONAL_GW1_DEADLINE_UTC;
  const deadlineLabel = state?.next_deadline_utc
    ? `GW${state.next_gw ?? "?"} DEADLINE`
    : "GW1 26/27 (PROV.)";
  const countdown =
    now !== null ? countdownString(countdownTo(deadlineIso, now)) : "—— --:--:--";

  const predUtc = state?.data_freshness?.predictions_utc ?? null;
  let dataKind: StatusKind = "warn";
  let dataLabel = "NO DATA";
  if (error) {
    dataKind = "crit";
    dataLabel = "API DOWN";
  } else if (predUtc && now) {
    const h = hoursSince(predUtc, now);
    dataKind = h <= 36 ? "good" : "warn";
    dataLabel = h <= 36 ? `FRESH ${shortUtc(predUtc)}` : `STALE ${shortUtc(predUtc)}`;
  }

  const modelLabel = state?.model
    ? `MODEL v${state.model.schema} · ${shortUtc(state.model.created_utc).slice(0, 6)}`
    : state?.model_version
      ? `MODEL ${state.model_version.toUpperCase()}`
      : "MODEL —";

  const seasonChip = live
    ? `LIVE ${seasonSlash(2026)}${state?.next_gw ? ` · GW${state.next_gw}` : ""}`
    : "PRE-SEASON — 2026/27 LAUNCHES SOON";

  return (
    <div className="border-b border-hairline bg-surface/80">
      <div className="mx-auto flex w-full max-w-[1280px] flex-wrap items-center gap-x-4 gap-y-1 px-4 py-1.5 font-mono text-[10.5px] uppercase tracking-[0.1em] sm:px-6">
        <span className="inline-flex items-center gap-2 text-ink-dim">
          {deadlineLabel}
          <span className="tnum text-ink" suppressHydrationWarning>
            {countdown}
          </span>
        </span>
        <Sep />
        <StatusBadge kind={dataKind} label={dataLabel} />
        <Sep />
        <span className="hidden text-ink-dim md:inline">{modelLabel}</span>
        <span className="ml-auto inline-flex items-center gap-2">
          {MOCK ? <StatusBadge kind="warn" label="MOCK FEED" /> : null}
          <span
            className={`inline-flex items-center gap-1.5 ${live ? "text-pitch-bright" : "text-ink-dim"}`}
          >
            <span className={live ? "tickpulse" : ""} aria-hidden="true">
              {live ? "●" : "○"}
            </span>
            {seasonChip}
          </span>
        </span>
      </div>
    </div>
  );
}
