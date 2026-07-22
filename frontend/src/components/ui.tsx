/** Shared desk primitives: cards, section headers, skeletons, status marks. */

import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <section
      className={`rounded-md border border-hairline bg-surface ${className}`}
    >
      {children}
    </section>
  );
}

/** Card header row: micro-label left, optional right-side slot. */
export function CardHead({
  label,
  right,
}: {
  label: string;
  right?: ReactNode;
}) {
  return (
    <header className="flex items-center justify-between gap-3 border-b border-hairline-soft px-4 py-2.5">
      <span className="microlabel">{label}</span>
      {right ? <span className="microlabel text-ink-dim">{right}</span> : null}
    </header>
  );
}

export function PageTitle({
  kicker,
  title,
  right,
}: {
  kicker: string;
  title: string;
  right?: ReactNode;
}) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
      <div>
        <div className="microlabel mb-1.5">{kicker}</div>
        <h1 className="display text-[28px] sm:text-[34px]">{title}</h1>
      </div>
      {right}
    </div>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}

/** Designed empty / pre-season / error state — never a raw spinner or stack trace. */
export function EmptyState({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-col items-start gap-3 rounded-md border border-dashed border-hairline bg-surface px-6 py-8">
      <div className="microlabel">NO FEED</div>
      <div className="display text-xl">{title}</div>
      <p className="max-w-md text-[13px] leading-relaxed text-ink-dim">{detail}</p>
      {action}
    </div>
  );
}

export type StatusKind = "good" | "warn" | "crit";

const STATUS_COLOR: Record<StatusKind, string> = {
  good: "text-good",
  warn: "text-warn",
  crit: "text-crit",
};

const STATUS_MARK: Record<StatusKind, string> = {
  good: "●",
  warn: "▲",
  crit: "■",
};

/** Status is always icon + label — never color alone. */
export function StatusBadge({ kind, label }: { kind: StatusKind; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.12em]">
      <span className={STATUS_COLOR[kind]} aria-hidden="true">
        {STATUS_MARK[kind]}
      </span>
      <span className="text-ink-mid">{label}</span>
    </span>
  );
}

/** Availability-risk mark from q0 = P(0 minutes): dot + percentage label. */
export function RiskMark({ q0 }: { q0: number }) {
  const kind: StatusKind = q0 >= 0.35 ? "crit" : q0 >= 0.15 ? "warn" : "good";
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[11px] tnum">
      <span className={STATUS_COLOR[kind]} aria-hidden="true">
        {STATUS_MARK[kind]}
      </span>
      <span className="text-ink-dim">{Math.round(q0 * 100)}%</span>
    </span>
  );
}

/** Signed xP delta, green for positive, red for negative (mono, tabular). */
export function Delta({ value, digits = 1 }: { value: number; digits?: number }) {
  const s = Math.abs(value).toFixed(digits);
  return (
    <span
      className={`font-mono tnum ${value >= 0 ? "text-pitch-bright" : "text-neg"}`}
    >
      {value >= 0 ? "+" : "−"}
      {s}
    </span>
  );
}
