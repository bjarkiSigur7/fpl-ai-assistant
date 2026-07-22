/** Formatting helpers: prices in 0.1m units, xP, deadlines, season labels. */

/** "£5.5m" from 55 (0.1m units). */
export function money(tenths: number): string {
  return `£${(tenths / 10).toFixed(1)}m`;
}

/** "5.5" from 55 — bare price for dense tables. */
export function moneyBare(tenths: number): string {
  return (tenths / 10).toFixed(1);
}

/** xP with 2 decimals for tables ("6.53"). */
export function xp2(v: number): string {
  return v.toFixed(2);
}

/** xP with 1 decimal for badges ("6.5"). */
export function xp1(v: number): string {
  return v.toFixed(1);
}

/** Signed delta with 1 decimal ("+8.8" / "−3.2"), using a true minus sign. */
export function signed1(v: number): string {
  const s = Math.abs(v).toFixed(1);
  return v >= 0 ? `+${s}` : `−${s}`;
}

/** "2016-17" style season label from the start year. */
export function seasonLabel(season: number): string {
  return `${season}-${String((season + 1) % 100).padStart(2, "0")}`;
}

/** "2026/27" style label (display variant). */
export function seasonSlash(season: number): string {
  return `${season}/${String((season + 1) % 100).padStart(2, "0")}`;
}

export interface Countdown {
  days: number;
  hours: number;
  minutes: number;
  seconds: number;
  past: boolean;
}

export function countdownTo(deadlineIso: string, now: Date): Countdown {
  const target = new Date(deadlineIso).getTime();
  let diff = Math.floor((target - now.getTime()) / 1000);
  const past = diff < 0;
  if (past) diff = -diff;
  const days = Math.floor(diff / 86400);
  const hours = Math.floor((diff % 86400) / 3600);
  const minutes = Math.floor((diff % 3600) / 60);
  const seconds = diff % 60;
  return { days, hours, minutes, seconds, past };
}

/** "29D 14:03:22" ticker countdown string. */
export function countdownString(c: Countdown): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  const core = `${pad(c.hours)}:${pad(c.minutes)}:${pad(c.seconds)}`;
  return `${c.past ? "−" : ""}${c.days}D ${core}`;
}

/** "22 JUL 06:30 UTC" from an ISO datetime. */
export function shortUtc(iso: string): string {
  const d = new Date(iso);
  const months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
  ];
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getUTCDate()} ${months[d.getUTCMonth()]} ${pad(d.getUTCHours())}:${pad(
    d.getUTCMinutes(),
  )} UTC`;
}

/** Hours elapsed since an ISO timestamp. */
export function hoursSince(iso: string, now: Date): number {
  return (now.getTime() - new Date(iso).getTime()) / 3_600_000;
}

/** Provisional 2026-27 GW1 deadline shown while the API has no live deadline. */
export const PROVISIONAL_GW1_DEADLINE_UTC = "2026-08-21T17:30:00Z";
