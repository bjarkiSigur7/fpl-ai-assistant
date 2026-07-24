"use client";

/**
 * RATING PITCH — the 15-slot squad board for the AI RATING page, adapted from
 * the dashboard's SVG pitch idiom (line markings, mowing stripes, vignette).
 * All 15 squad slots live on the turf in position rows (2 GKP / 5 DEF / 5 MID /
 * 3 FWD); filled slots are interactive HTML chips (name, price · xP, an × to
 * remove), empty slots are dashed placeholders. After an evaluation the GW1
 * best XI stays lit, the bench-4 dim, and the suggested captain wears the C.
 */

import { useId } from "react";
import type { Position } from "@/lib/types";
import { POSITIONS } from "@/lib/types";
import { moneyBare, xp1 } from "@/lib/format";
import { Card, CardHead } from "./ui";

export interface RatingSlotPlayer {
  code: number;
  name: string;
  team_short: string;
  price: number;
  xp: number | null;
}

export interface RatingHighlight {
  /** Codes of the suggested GW1 best XI — everyone else dims to the bench. */
  bestXi: Set<number>;
  captain: number;
}

export const SQUAD_QUOTA: Record<Position, number> = { GKP: 2, DEF: 5, MID: 5, FWD: 3 };

const W = 360;
const H = 520;
const ROW_Y: Record<Position, number> = { GKP: 10.8, DEF: 36.2, MID: 61.5, FWD: 86.9 };

function PitchSurface() {
  const vignetteId = useId().replace(/:/g, "");
  const line = "rgba(255,255,255,0.09)";
  const stripeH = (H - 16) / 8;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      aria-hidden="true"
      className="absolute inset-0 h-full w-full"
    >
      <defs>
        <radialGradient id={vignetteId} cx="50%" cy="46%" r="75%">
          <stop offset="55%" stopColor="#000000" stopOpacity="0" />
          <stop offset="100%" stopColor="#000000" stopOpacity="0.38" />
        </radialGradient>
      </defs>
      <rect x={0} y={0} width={W} height={H} rx={6} fill="#121410" />
      {/* mowing stripes */}
      {Array.from({ length: 8 }, (_, i) =>
        i % 2 === 0 ? (
          <rect
            key={i}
            x={8}
            y={8 + i * stripeH}
            width={W - 16}
            height={stripeH}
            fill="rgba(255,255,255,0.016)"
          />
        ) : null,
      )}
      {/* outline + halfway + center */}
      <rect x={8} y={8} width={W - 16} height={H - 16} fill="none" stroke={line} />
      <line x1={8} y1={H / 2} x2={W - 8} y2={H / 2} stroke={line} />
      <circle cx={W / 2} cy={H / 2} r={38} fill="none" stroke={line} />
      <circle cx={W / 2} cy={H / 2} r={1.8} fill={line} />
      {/* penalty + goal boxes, both ends */}
      <rect x={92} y={8} width={176} height={56} fill="none" stroke={line} />
      <rect x={134} y={8} width={92} height={22} fill="none" stroke={line} />
      <path d="M 138 64 A 42 42 0 0 0 222 64" fill="none" stroke={line} />
      <rect x={92} y={H - 64} width={176} height={56} fill="none" stroke={line} />
      <rect x={134} y={H - 30} width={92} height={22} fill="none" stroke={line} />
      <path d={`M 138 ${H - 64} A 42 42 0 0 1 222 ${H - 64}`} fill="none" stroke={line} />
      <rect x={0} y={0} width={W} height={H} fill={`url(#${vignetteId})`} />
    </svg>
  );
}

function SlotChip({
  p,
  pos,
  x,
  y,
  widthPct,
  highlight,
  onRemove,
}: {
  p: RatingSlotPlayer | null;
  pos: Position;
  x: number;
  y: number;
  widthPct: number;
  highlight: RatingHighlight | null;
  onRemove: (code: number) => void;
}) {
  const style = { left: `${x}%`, top: `${y}%`, width: `${widthPct}%` };
  if (!p) {
    return (
      <div className="absolute -translate-x-1/2 -translate-y-1/2" style={style}>
        <div className="rounded border border-dashed border-hairline bg-bg/30 px-1 py-2.5 text-center">
          <span className="microlabel">{pos}</span>
        </div>
      </div>
    );
  }
  const benched = highlight !== null && !highlight.bestXi.has(p.code);
  const isCaptain = highlight !== null && highlight.captain === p.code;
  return (
    <div className="absolute -translate-x-1/2 -translate-y-1/2" style={style}>
      <div
        className={`relative rounded border bg-surface px-1 pb-1 pt-1.5 text-center transition-opacity ${
          benched ? "border-hairline opacity-40" : "border-[rgba(255,255,255,0.14)]"
        }`}
      >
        <div className="truncate text-[11px] font-medium leading-tight text-ink">{p.name}</div>
        <div className="truncate font-mono text-[8.5px] tnum leading-tight text-ink-dim">
          {moneyBare(p.price)} ·{" "}
          <span className="text-ink-mid">{p.xp !== null ? `${xp1(p.xp)}xP` : "—"}</span>
        </div>
        <button
          onClick={() => onRemove(p.code)}
          aria-label={`Remove ${p.name}`}
          className="absolute -right-1.5 -top-1.5 flex h-4 w-4 items-center justify-center rounded-full border border-hairline bg-raised font-mono text-[10px] leading-none text-ink-dim transition-colors hover:border-neg hover:text-neg"
        >
          ×
        </button>
        {isCaptain ? (
          <span
            title="Suggested captain"
            className="absolute -left-1.5 -top-1.5 flex h-4 w-4 items-center justify-center rounded-full bg-pitch font-mono text-[9px] font-bold text-[#0d0d0d]"
          >
            C<span className="sr-only"> — suggested captain</span>
          </span>
        ) : null}
        {benched ? (
          <span className="absolute -bottom-2 left-1/2 -translate-x-1/2 rounded border border-hairline-soft bg-raised px-1 font-mono text-[7.5px] uppercase tracking-[0.1em] text-ink-dim">
            BENCH
          </span>
        ) : null}
      </div>
    </div>
  );
}

export function RatingPitch({
  slots,
  highlight,
  onRemove,
  right,
}: {
  /** Per position: quota-length arrays, filled from the front, null = empty slot. */
  slots: Record<Position, (RatingSlotPlayer | null)[]>;
  highlight: RatingHighlight | null;
  onRemove: (code: number) => void;
  right?: string;
}) {
  return (
    <Card>
      <CardHead label="THE SQUAD — 15 SLOTS" right={right} />
      <div className="p-3">
        <div className="relative mx-auto w-full max-w-[440px]" style={{ aspectRatio: "360 / 520" }}>
          <PitchSurface />
          {POSITIONS.map((pos) => {
            const row = slots[pos];
            const n = row.length;
            const widthPct = Math.min(21, 100 / (n + 1) - 1.2);
            return row.map((p, i) => (
              <SlotChip
                key={p ? p.code : `${pos}-${i}`}
                p={p}
                pos={pos}
                x={((i + 1) / (n + 1)) * 100}
                y={ROW_Y[pos]}
                widthPct={widthPct}
                highlight={highlight}
                onRemove={onRemove}
              />
            ));
          })}
        </div>
        {highlight ? (
          <p className="mt-3 border-t border-hairline-soft pt-2.5 text-center font-mono text-[9.5px] uppercase tracking-[0.1em] text-ink-dim">
            GW1 BEST XI LIT · BENCH-4 DIMMED · C = SUGGESTED CAPTAIN
          </p>
        ) : null}
      </div>
    </Card>
  );
}
