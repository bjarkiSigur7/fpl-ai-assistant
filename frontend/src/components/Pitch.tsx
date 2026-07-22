"use client";

/**
 * THE PITCH — squads rendered on a vertical SVG football pitch: subtle line
 * markings, mowing stripes, edge vignette; player chips positioned by formation,
 * captain badged C (the accent moment), price + xP under each name; bench rail below.
 */

import { useId } from "react";
import type { Position } from "@/lib/types";
import { moneyBare, xp1 } from "@/lib/format";
import { Card, CardHead, Skeleton } from "./ui";

export interface PitchPlayer {
  code: number;
  name: string;
  position: Position;
  price: number | null;
  xp: number | null;
}

const ROW_ORDER: Position[] = ["GKP", "DEF", "MID", "FWD"];
const ROW_Y: Record<Position, number> = { GKP: 58, DEF: 162, MID: 286, FWD: 404 };

const W = 360;
const H = 476;

function truncate(name: string, max: number): string {
  return name.length > max ? `${name.slice(0, max - 1)}…` : name;
}

function Chip({
  p,
  x,
  y,
  width,
  isCaptain,
  isVice,
}: {
  p: PitchPlayer;
  x: number;
  y: number;
  width: number;
  isCaptain: boolean;
  isVice: boolean;
}) {
  const fontSize = width >= 68 ? 10 : 9;
  const maxChars = Math.floor((width - 6) / (fontSize * 0.58));
  return (
    <g>
      <rect
        x={x - width / 2}
        y={y - 12}
        width={width}
        height={24}
        rx={4}
        fill="#1a1a19"
        stroke="rgba(255,255,255,0.14)"
        strokeWidth={1}
      />
      <text
        x={x}
        y={y + 3.5}
        textAnchor="middle"
        fontSize={fontSize}
        fontWeight={600}
        fill="#ffffff"
        fontFamily="var(--font-plex-sans)"
      >
        {truncate(p.name, maxChars)}
      </text>
      <text
        x={x}
        y={y + 24}
        textAnchor="middle"
        fontSize={width >= 68 ? 8.5 : 7.6}
        fill="#898781"
        fontFamily="var(--font-plex-mono)"
        style={{ fontVariantNumeric: "tabular-nums" }}
      >
        {p.price !== null ? moneyBare(p.price) : "—"}
        {" · "}
        <tspan fill="#c3c2b7">{p.xp !== null ? `${xp1(p.xp)}xP` : "—"}</tspan>
      </text>
      {isCaptain ? (
        <g>
          <circle cx={x + width / 2 - 2} cy={y - 12} r={7.5} fill="#0ca30c" />
          <text
            x={x + width / 2 - 2}
            y={y - 9}
            textAnchor="middle"
            fontSize={8.5}
            fontWeight={700}
            fill="#0d0d0d"
            fontFamily="var(--font-plex-mono)"
          >
            C
          </text>
        </g>
      ) : null}
      {isVice ? (
        <g>
          <circle
            cx={x + width / 2 - 2}
            cy={y - 12}
            r={7}
            fill="#1a1a19"
            stroke="rgba(255,255,255,0.4)"
            strokeWidth={1}
          />
          <text
            x={x + width / 2 - 2}
            y={y - 9}
            textAnchor="middle"
            fontSize={8}
            fontWeight={600}
            fill="#c3c2b7"
            fontFamily="var(--font-plex-mono)"
          >
            V
          </text>
        </g>
      ) : null}
    </g>
  );
}

function Markings({ vignetteId }: { vignetteId: string }) {
  const line = "rgba(255,255,255,0.09)";
  return (
    <g>
      {/* mowing stripes */}
      {Array.from({ length: 8 }, (_, i) =>
        i % 2 === 0 ? (
          <rect
            key={i}
            x={8}
            y={8 + i * 57.5}
            width={W - 16}
            height={57.5}
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
      <path d="M 138 412 A 42 42 0 0 1 222 412" fill="none" stroke={line} />
      {/* vignette */}
      <rect x={0} y={0} width={W} height={H} fill={`url(#${vignetteId})`} />
    </g>
  );
}

export function Pitch({
  label,
  right,
  lineup,
  bench,
  captain,
  vice,
  footer,
}: {
  label: string;
  right?: string;
  lineup: PitchPlayer[];
  bench: PitchPlayer[];
  captain: number | null;
  vice: number | null;
  footer?: React.ReactNode;
}) {
  const vignetteId = useId().replace(/:/g, "");
  const rows = ROW_ORDER.map((pos) => lineup.filter((p) => p.position === pos));
  return (
    <Card>
      <CardHead label={label} right={right} />
      <div className="p-3">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`${label}: pitch view`}
          className="w-full"
        >
          <defs>
            <radialGradient id={vignetteId} cx="50%" cy="46%" r="75%">
              <stop offset="55%" stopColor="#000000" stopOpacity="0" />
              <stop offset="100%" stopColor="#000000" stopOpacity="0.38" />
            </radialGradient>
          </defs>
          <rect x={0} y={0} width={W} height={H} rx={6} fill="#121410" />
          <Markings vignetteId={vignetteId} />
          {rows.map((row) => {
            const spacing = W / (row.length + 1);
            const chipW = Math.min(78, spacing - 4);
            return row.map((p, i) => (
              <Chip
                key={p.code}
                p={p}
                x={spacing * (i + 1)}
                y={ROW_Y[p.position]}
                width={chipW}
                isCaptain={p.code === captain}
                isVice={p.code === vice}
              />
            ));
          })}
        </svg>
        {/* bench rail */}
        <div className="mt-3 border-t border-hairline-soft pt-3">
          <div className="microlabel mb-2">BENCH</div>
          <div className="grid grid-cols-4 gap-2">
            {bench.map((p, i) => (
              <div
                key={p.code}
                className="rounded border border-hairline-soft bg-raised px-2 py-1.5"
              >
                <div className="truncate text-[11px] font-medium text-ink">
                  <span className="mr-1 font-mono text-[9px] text-ink-dim">{i + 1}</span>
                  {p.name}
                </div>
                <div className="font-mono text-[9.5px] tnum text-ink-dim">
                  {p.position} · {p.xp !== null ? `${xp1(p.xp)}xP` : "—"}
                </div>
              </div>
            ))}
          </div>
        </div>
        {footer}
      </div>
    </Card>
  );
}

export function PitchSkeleton({ label }: { label: string }) {
  return (
    <Card>
      <CardHead label={label} />
      <div className="p-3">
        <Skeleton className="aspect-[360/476] w-full" />
        <div className="mt-3 grid grid-cols-4 gap-2">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-10" />
          ))}
        </div>
      </div>
    </Card>
  );
}
