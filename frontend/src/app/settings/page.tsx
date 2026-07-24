"use client";

/**
 * Settings. Local mode: entry-id persistence (localStorage), API wiring readout,
 * model manifest, and the data-refresh trigger with live status polling + log
 * tail. Static (public) mode hides those local-only surfaces and shows the data
 * feed's provenance instead: bundle freshness from meta.json and the GitHub
 * Actions batch cadence.
 */

import { useEffect, useRef, useState } from "react";
import { API_URL, MOCK, STATIC, startRefresh, useDeskState, useRefreshStatus } from "@/lib/api";
import { gwWindow, seasonLabel, shortUtc } from "@/lib/format";
import { Card, CardHead, PageTitle, StatusBadge, Skeleton } from "@/components/ui";
import { useEntryId } from "@/lib/useEntryId";

function EntryCard() {
  const [entryId, setEntryId] = useEntryId();
  const [saved, setSaved] = useState(false);

  const submit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const raw = String(new FormData(e.currentTarget).get("entry") ?? "").trim();
    const n = Number(raw);
    if (Number.isInteger(n) && n > 0) {
      setEntryId(n);
      setSaved(true);
      setTimeout(() => setSaved(false), 1800);
    }
  };

  return (
    <Card>
      <CardHead label="FPL ENTRY" right={entryId !== null ? `#${entryId}` : "NOT SET"} />
      <div className="p-4">
        <p className="mb-3 max-w-lg text-[13px] leading-relaxed text-ink-dim">
          Your FPL team id (the number in the URL on the Points page of the official site).
          With it, the desk reconstructs your squad from public endpoints and turns the
          verdict, planner and chip advice squad-aware. Stored locally in this browser.
        </p>
        <form onSubmit={submit} className="flex flex-wrap items-center gap-2">
          <input
            key={entryId ?? "none"}
            name="entry"
            defaultValue={entryId !== null ? String(entryId) : ""}
            inputMode="numeric"
            pattern="[0-9]*"
            placeholder="e.g. 1178461"
            aria-label="FPL entry id"
            className="w-44 rounded border border-hairline bg-surface px-3 py-2 font-mono text-[13px] tnum text-ink placeholder:text-ink-dim focus:border-ink-dim focus:outline-none"
          />
          <button
            type="submit"
            className="hit relative rounded bg-pitch px-4 py-2 font-mono text-[11px] font-semibold uppercase tracking-[0.12em] text-[#0d0d0d] transition-opacity hover:opacity-90"
          >
            SAVE
          </button>
          {entryId !== null ? (
            <button
              type="button"
              onClick={() => setEntryId(null)}
              className="hit relative rounded border border-hairline px-4 py-2 font-mono text-[11px] uppercase tracking-[0.12em] text-ink-dim transition-colors hover:text-ink"
            >
              CLEAR
            </button>
          ) : null}
          {saved ? <StatusBadge kind="good" label="SAVED" /> : null}
        </form>
      </div>
    </Card>
  );
}

function WiringCard() {
  const { data: state } = useDeskState();
  return (
    <Card>
      <CardHead label="WIRING" right={MOCK ? "MOCK MODE" : "LIVE API"} />
      <div className="space-y-2.5 p-4 font-mono text-[12px]">
        <div className="flex flex-wrap justify-between gap-2">
          <span className="text-ink-dim">API URL</span>
          <span className="tnum text-ink-mid">{MOCK ? "(mocked in-browser)" : API_URL}</span>
        </div>
        <div className="flex flex-wrap justify-between gap-2">
          <span className="text-ink-dim">SEASON SERVED</span>
          <span className="text-ink-mid">
            {state ? seasonLabel(state.season) : "…"}
            {state && !state.is_live_2026_27 ? " (2026-27 not launched)" : ""}
          </span>
        </div>
        <div className="flex flex-wrap justify-between gap-2">
          <span className="text-ink-dim">PREDICTIONS</span>
          <span className="tnum text-ink-mid">
            {state?.data_freshness?.predictions_utc
              ? `${shortUtc(state.data_freshness.predictions_utc)} · ${gwWindow(state.data_freshness.predictions_gws)}`
              : "none"}
          </span>
        </div>
        <div className="flex flex-wrap justify-between gap-2">
          <span className="text-ink-dim">MODEL ARTIFACTS</span>
          <span className="tnum text-ink-mid">
            {state?.model
              ? `${Object.keys(state.model.components).join(" / ")} · trained ${shortUtc(state.model.created_utc)}`
              : "none"}
          </span>
        </div>
      </div>
    </Card>
  );
}

function RefreshCard() {
  const [kicked, setKicked] = useState(false);
  const running = useRefreshStatus(false).data?.running ?? false;
  // Poll while a kicked run is still in flight (or right after kicking, before
  // the first status readback lands).
  const { data: status, mutate } = useRefreshStatus(kicked && running);
  const logRef = useRef<HTMLDivElement>(null);
  const logLength = status?.log?.length ?? 0;

  useEffect(() => {
    const el = logRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [logLength]);

  const kick = async () => {
    setKicked(true);
    try {
      await startRefresh();
      await mutate();
    } catch {
      setKicked(false);
    }
  };

  const badge = running ? (
    <StatusBadge kind="warn" label={`RUNNING · ${status?.stage?.toUpperCase() ?? "…"}`} />
  ) : status?.ok === false ? (
    <StatusBadge kind="crit" label="LAST RUN FAILED" />
  ) : status?.finished_utc ? (
    <StatusBadge kind="good" label={`IDLE · LAST ${shortUtc(status.finished_utc)}`} />
  ) : (
    <StatusBadge kind="warn" label="NEVER RUN" />
  );

  return (
    <Card>
      <CardHead label="DATA REFRESH" right={undefined} />
      <div className="p-4">
        <div className="mb-3 flex flex-wrap items-center gap-3">
          <button
            onClick={() => void kick()}
            disabled={running}
            className="rounded bg-pitch px-4 py-2 font-mono text-[11px] font-semibold uppercase tracking-[0.12em] text-[#0d0d0d] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? "REFRESHING…" : "RUN REFRESH"}
          </button>
          {badge}
        </div>
        <p className="mb-3 max-w-lg text-[13px] leading-relaxed text-ink-dim">
          Snapshot → incremental pulls → rebuild parquet tables → predict. The 2026-27
          launch detector runs on every snapshot; when the new game goes live the desk
          flips to live predictions automatically.
        </p>
        <div
          ref={logRef}
          className="deskscroll max-h-56 overflow-y-auto rounded border border-hairline-soft bg-[#0a0a0a] p-3 font-mono text-[11px] leading-relaxed text-ink-mid"
          aria-live="polite"
          aria-label="Refresh log"
        >
          {status?.log?.length ? (
            status.log.map((line, i) => (
              <div key={i} className="whitespace-pre-wrap">
                <span className="mr-2 select-none text-ink-dim">›</span>
                {line}
              </div>
            ))
          ) : (
            <span className="text-ink-dim">— log empty; run a refresh to tail it here —</span>
          )}
        </div>
      </div>
    </Card>
  );
}

/** Static (public) build: bundle freshness + the GitHub Actions batch cadence. */
function DataFeedCard() {
  const { data: state } = useDeskState();
  const f = state?.data_freshness ?? null;
  const gws = f?.predictions_gws ?? [];
  return (
    <Card>
      <CardHead label="DATA FEED" right="STATIC BUNDLE" />
      <div className="p-4">
        <div className="mb-3 space-y-2.5 font-mono text-[12px]">
          <div className="flex flex-wrap justify-between gap-2">
            <span className="text-ink-dim">LAST MODEL RUN</span>
            <span className="tnum text-ink-mid">
              {f?.predictions_utc ? shortUtc(f.predictions_utc) : "…"}
            </span>
          </div>
          <div className="flex flex-wrap justify-between gap-2">
            <span className="text-ink-dim">SEASON SERVED</span>
            <span className="text-ink-mid">{state ? seasonLabel(state.season) : "…"}</span>
          </div>
          <div className="flex flex-wrap justify-between gap-2">
            <span className="text-ink-dim">PREDICTION WINDOW</span>
            <span className="tnum text-ink-mid">{gws.length ? gwWindow(gws) : "…"}</span>
          </div>
          <div className="flex flex-wrap justify-between gap-2">
            <span className="text-ink-dim">NEXT DEADLINE</span>
            <span className="tnum text-ink-mid">
              {state?.next_deadline_utc
                ? `${state.next_gw !== null ? `GW${state.next_gw} · ` : ""}${shortUtc(state.next_deadline_utc)}`
                : "…"}
            </span>
          </div>
          <div className="flex flex-wrap justify-between gap-2">
            <span className="text-ink-dim">MODEL</span>
            <span className="tnum text-ink-mid">{state?.model_version ?? "…"}</span>
          </div>
        </div>
        <p className="max-w-lg text-[13px] leading-relaxed text-ink-dim">
          The full model batch runs daily on GitHub Actions — snapshot, rebuild,
          predict, optimize — and publishes this site with a fresh data bundle. An
          hourly deadline watch triggers an extra run in the final hours before each
          gameweek deadline. Personalized my-team plans and manual refreshes are
          local-only features of the open-source desk.
        </p>
      </div>
    </Card>
  );
}

export default function SettingsPage() {
  const { isLoading } = useDeskState();
  if (STATIC) {
    return (
      <div>
        <PageTitle kicker="DESK CONFIGURATION" title="SETTINGS" />
        <div className="space-y-5">
          {isLoading ? <Skeleton className="h-64" /> : <DataFeedCard />}
        </div>
      </div>
    );
  }
  return (
    <div>
      <PageTitle kicker="DESK CONFIGURATION" title="SETTINGS" />
      <div className="space-y-5">
        <EntryCard />
        {isLoading ? <Skeleton className="h-40" /> : <WiringCard />}
        <RefreshCard />
      </div>
    </div>
  );
}
