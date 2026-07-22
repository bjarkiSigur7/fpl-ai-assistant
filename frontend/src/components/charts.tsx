"use client";

/**
 * Hand-rolled SVG charts (no chart lib), following the desk chart rules:
 * one y-axis, thin marks (2px lines, rounded bar ends), recessive hairline grid
 * (#2c2c2a), direct labels on <=4 series, crosshair + tooltip on lines, per-mark
 * tooltips on bars, tabular numerals, text never set in series colors.
 */

import { useRef, useState } from "react";

/** Validated dark-surface categorical series colors — use in this order. */
export const SERIES: readonly string[] = ["#3987e5", "#d95926", "#199e70", "#c98500"];

const GRID = "#2c2c2a";
const INK = "#ffffff";
const INK_MID = "#c3c2b7";
const INK_DIM = "#898781";
const MONO = "var(--font-plex-mono)";

const fmt = (v: number, digits = 1) => v.toFixed(digits);

function niceMax(v: number): number {
  if (v <= 0) return 1;
  const pow = 10 ** Math.floor(Math.log10(v));
  for (const m of [1, 2, 2.5, 5, 10]) {
    if (m * pow >= v) return m * pow;
  }
  return 10 * pow;
}

// ---------------------------------------------------------------------------
// tooltip chrome (HTML overlay)
// ---------------------------------------------------------------------------

interface TipState {
  xFrac: number;
  yFrac: number;
  lines: { swatch?: string; text: string }[];
  title: string;
}

function Tip({ tip }: { tip: TipState }) {
  const left = tip.xFrac > 0.62;
  return (
    <div
      className="pointer-events-none absolute z-10 rounded border border-hairline bg-[#0d0d0d]/95 px-2.5 py-1.5 font-mono text-[10.5px] leading-relaxed shadow-lg"
      style={{
        left: `${tip.xFrac * 100}%`,
        top: `${Math.min(tip.yFrac * 100, 60)}%`,
        transform: left ? "translate(calc(-100% - 10px), 0)" : "translate(10px, 0)",
      }}
    >
      <div className="mb-0.5 text-ink-dim">{tip.title}</div>
      {tip.lines.map((l, i) => (
        <div key={i} className="tnum flex items-center gap-1.5 text-ink-mid">
          {l.swatch ? (
            <span
              aria-hidden="true"
              className="inline-block h-[2px] w-2.5 rounded-full"
              style={{ background: l.swatch }}
            />
          ) : null}
          {l.text}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// LineChart — <=4 series, direct labels, crosshair + tooltip
// ---------------------------------------------------------------------------

export interface LineSeries {
  label: string;
  values: (number | null)[];
}

export function LineChart({
  series,
  xLabels,
  unit = "",
  vw = 560,
  vh = 200,
}: {
  series: LineSeries[];
  xLabels: string[];
  unit?: string;
  /** viewBox width — match roughly to the rendered CSS width so text stays ~9-10px. */
  vw?: number;
  vh?: number;
}) {
  const [tip, setTip] = useState<TipState | null>(null);
  const [hoverI, setHoverI] = useState<number | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const VW = vw;
  const VH = vh;
  const pad = { l: 30, r: 78, t: 10, b: 20 };
  const iw = VW - pad.l - pad.r;
  const ih = VH - pad.t - pad.b;
  const n = xLabels.length;

  const allValues = series.flatMap((s) => s.values.filter((v): v is number => v !== null));
  const yMax = niceMax(Math.max(...allValues, 0.1) * 1.08);
  const x = (i: number) => pad.l + (n <= 1 ? iw / 2 : (i / (n - 1)) * iw);
  const y = (v: number) => pad.t + ih - (v / yMax) * ih;

  const gridLines = [0.25, 0.5, 0.75, 1].map((f) => f * yMax);

  // Direct labels at line ends, pushed apart so they never collide.
  const MIN_GAP = 12;
  const labelYs: number[] = (() => {
    const ends = series.map((s, si) => {
      const lastIdx = s.values.reduce<number>((acc, v, i) => (v !== null ? i : acc), 0);
      const v = s.values[lastIdx];
      return { si, y: v !== null ? y(v) : VH - pad.b };
    });
    const sorted = [...ends].sort((a, b) => a.y - b.y);
    for (let i = 1; i < sorted.length; i++) {
      if (sorted[i].y - sorted[i - 1].y < MIN_GAP) sorted[i].y = sorted[i - 1].y + MIN_GAP;
    }
    const out: number[] = [];
    for (const e of sorted) out[e.si] = e.y;
    return out;
  })();

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const fx = (e.clientX - box.left) / box.width;
    const vx = fx * VW;
    const i = Math.max(0, Math.min(n - 1, Math.round(((vx - pad.l) / iw) * (n - 1))));
    setHoverI(i);
    setTip({
      xFrac: x(i) / VW,
      yFrac: (e.clientY - box.top) / box.height,
      title: xLabels[i],
      lines: series
        .map((s, si) =>
          s.values[i] !== null
            ? { swatch: SERIES[si % SERIES.length], text: `${s.label} ${fmt(s.values[i]!)}${unit}` }
            : null,
        )
        .filter((l): l is { swatch: string; text: string } => l !== null),
    });
  };

  return (
    <div ref={ref} className="relative">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full" role="img" aria-label="line chart">
        {gridLines.map((v) => (
          <g key={v}>
            <line x1={pad.l} y1={y(v)} x2={VW - pad.r} y2={y(v)} stroke={GRID} strokeWidth={1} />
            <text x={pad.l - 6} y={y(v) + 3} textAnchor="end" fontSize={9} fill={INK_DIM} fontFamily={MONO}>
              {fmt(v, yMax >= 10 ? 0 : 1)}
            </text>
          </g>
        ))}
        <line x1={pad.l} y1={y(0)} x2={VW - pad.r} y2={y(0)} stroke={GRID} strokeWidth={1} />
        {hoverI !== null ? (
          <line
            x1={x(hoverI)}
            y1={pad.t}
            x2={x(hoverI)}
            y2={pad.t + ih}
            stroke="rgba(255,255,255,0.25)"
            strokeWidth={1}
          />
        ) : null}
        {series.map((s, si) => {
          const pts = s.values
            .map((v, i) => (v !== null ? `${x(i)},${y(v)}` : null))
            .filter((p): p is string => p !== null);
          const lastIdx = s.values.reduce<number>((acc, v, i) => (v !== null ? i : acc), 0);
          const lastVal = s.values[lastIdx];
          return (
            <g key={s.label}>
              <polyline
                points={pts.join(" ")}
                fill="none"
                stroke={SERIES[si % SERIES.length]}
                strokeWidth={2}
                strokeLinejoin="round"
                strokeLinecap="round"
              />
              {hoverI !== null && s.values[hoverI] !== null ? (
                <circle cx={x(hoverI)} cy={y(s.values[hoverI]!)} r={3} fill={SERIES[si % SERIES.length]} />
              ) : null}
              {lastVal !== null ? (
                <g>
                  <circle
                    cx={x(lastIdx) + 7}
                    cy={labelYs[si] - 1}
                    r={2.5}
                    fill={SERIES[si % SERIES.length]}
                  />
                  <text
                    x={x(lastIdx) + 13}
                    y={labelYs[si] + 2}
                    fontSize={9.5}
                    fill={INK_MID}
                    fontFamily={MONO}
                  >
                    {s.label}
                  </text>
                </g>
              ) : null}
            </g>
          );
        })}
        {xLabels.map((l, i) =>
          n <= 10 || i % Math.ceil(n / 10) === 0 ? (
            <text key={i} x={x(i)} y={VH - 5} textAnchor="middle" fontSize={9} fill={INK_DIM} fontFamily={MONO}>
              {l}
            </text>
          ) : null,
        )}
        <rect
          x={pad.l}
          y={pad.t}
          width={iw}
          height={ih}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => {
            setTip(null);
            setHoverI(null);
          }}
        />
      </svg>
      {tip ? <Tip tip={tip} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// MiniBarChart — chip EV small-multiples: one series, per-mark tooltip
// ---------------------------------------------------------------------------

export function MiniBarChart({
  values,
  xLabels,
  color = SERIES[0],
  highlight,
  unit = "",
}: {
  values: (number | null)[];
  xLabels: string[];
  color?: string;
  highlight?: number;
  unit?: string;
}) {
  const [tip, setTip] = useState<TipState | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const VW = 240;
  const VH = 120;
  const pad = { l: 26, r: 8, t: 16, b: 18 };
  const iw = VW - pad.l - pad.r;
  const ih = VH - pad.t - pad.b;
  const n = values.length;

  const present = values.filter((v): v is number => v !== null);
  const maxAbs = niceMax(Math.max(...present.map((v) => Math.abs(v)), 0.1) * 1.05);
  const hasNeg = present.some((v) => v < 0);
  const yZero = hasNeg ? pad.t + ih / 2 : pad.t + ih;
  const scale = (hasNeg ? ih / 2 : ih) / maxAbs;
  const x = (i: number) => pad.l + ((i + 0.5) / n) * iw;

  return (
    <div ref={ref} className="relative">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full" role="img" aria-label="bar chart">
        <line x1={pad.l} y1={yZero} x2={VW - pad.r} y2={yZero} stroke={GRID} strokeWidth={1} />
        <text x={pad.l - 5} y={pad.t + 3} textAnchor="end" fontSize={8.5} fill={INK_DIM} fontFamily={MONO}>
          {fmt(maxAbs, 0)}
        </text>
        <text x={pad.l - 5} y={yZero + 3} textAnchor="end" fontSize={8.5} fill={INK_DIM} fontFamily={MONO}>
          0
        </text>
        {values.map((v, i) => {
          if (v === null) return null;
          const h = Math.max(Math.abs(v) * scale, 2);
          const yTop = v >= 0 ? yZero - h : yZero;
          const hl = i === highlight;
          return (
            <g key={i}>
              <rect
                x={x(i) - 4}
                y={yTop}
                width={8}
                height={h}
                rx={3}
                fill={color}
                opacity={hl ? 1 : 0.5}
                onPointerEnter={(e) => {
                  const box = ref.current?.getBoundingClientRect();
                  if (!box) return;
                  setTip({
                    xFrac: x(i) / VW,
                    yFrac: (e.clientY - box.top) / box.height,
                    title: xLabels[i],
                    lines: [{ swatch: color, text: `${v >= 0 ? "+" : "−"}${fmt(Math.abs(v))}${unit}` }],
                  });
                }}
                onPointerLeave={() => setTip(null)}
              />
              {hl ? (
                <text
                  x={x(i)}
                  y={yTop - 4}
                  textAnchor="middle"
                  fontSize={9.5}
                  fontWeight={600}
                  fill={INK}
                  fontFamily={MONO}
                >
                  {v >= 0 ? "+" : "−"}
                  {fmt(Math.abs(v))}
                </text>
              ) : null}
            </g>
          );
        })}
        {xLabels.map((l, i) => (
          <text key={i} x={x(i)} y={VH - 5} textAnchor="middle" fontSize={8.5} fill={INK_DIM} fontFamily={MONO}>
            {l}
          </text>
        ))}
      </svg>
      {tip ? <Tip tip={tip} /> : null}
    </div>
  );
}

// ---------------------------------------------------------------------------
// HistoryChart — bars (FPL points) + line (xG) on one axis, crosshair tooltip
// ---------------------------------------------------------------------------

export interface HistoryPoint {
  gw: number;
  points: number;
  xg: number | null;
  minutes: number;
}

export function HistoryChart({ data }: { data: HistoryPoint[] }) {
  const [tip, setTip] = useState<TipState | null>(null);
  const [hoverI, setHoverI] = useState<number | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  const VW = 1000;
  const VH = 220;
  const pad = { l: 34, r: 50, t: 12, b: 22 };
  const iw = VW - pad.l - pad.r;
  const ih = VH - pad.t - pad.b;
  const n = data.length;

  const yMax = niceMax(Math.max(...data.map((d) => d.points), 4) * 1.1);
  const x = (i: number) => pad.l + ((i + 0.5) / n) * iw;
  const y = (v: number) => pad.t + ih - (v / yMax) * ih;
  const barW = Math.min(6, (iw / n) * 0.45);

  const xgPts = data
    .map((d, i) => (d.xg !== null ? `${x(i)},${y(d.xg)}` : null))
    .filter((p): p is string => p !== null);
  const lastXg = data.reduce<{ i: number; v: number } | null>(
    (acc, d, i) => (d.xg !== null ? { i, v: d.xg } : acc),
    null,
  );

  const onMove = (e: React.PointerEvent<SVGRectElement>) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    const fx = (e.clientX - box.left) / box.width;
    const vx = fx * VW;
    const i = Math.max(0, Math.min(n - 1, Math.floor(((vx - pad.l) / iw) * n)));
    const d = data[i];
    setHoverI(i);
    setTip({
      xFrac: x(i) / VW,
      yFrac: (e.clientY - box.top) / box.height,
      title: `GW${d.gw}`,
      lines: [
        { swatch: SERIES[0], text: `PTS ${d.points}` },
        ...(d.xg !== null ? [{ swatch: SERIES[1], text: `xG ${d.xg.toFixed(2)}` }] : []),
        { text: `MIN ${d.minutes}` },
      ],
    });
  };

  return (
    <div ref={ref} className="relative">
      <svg viewBox={`0 0 ${VW} ${VH}`} className="w-full" role="img" aria-label="season history">
        {[0.25, 0.5, 0.75, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} y1={y(f * yMax)} x2={VW - pad.r} y2={y(f * yMax)} stroke={GRID} strokeWidth={1} />
            <text x={pad.l - 6} y={y(f * yMax) + 3} textAnchor="end" fontSize={9} fill={INK_DIM} fontFamily={MONO}>
              {fmt(f * yMax, 0)}
            </text>
          </g>
        ))}
        <line x1={pad.l} y1={y(0)} x2={VW - pad.r} y2={y(0)} stroke={GRID} strokeWidth={1} />
        {hoverI !== null ? (
          <line x1={x(hoverI)} y1={pad.t} x2={x(hoverI)} y2={pad.t + ih} stroke="rgba(255,255,255,0.25)" strokeWidth={1} />
        ) : null}
        {data.map((d, i) => (
          <rect
            key={d.gw}
            x={x(i) - barW / 2}
            y={y(d.points)}
            width={barW}
            height={Math.max(y(0) - y(d.points), d.points > 0 ? 2 : 0)}
            rx={2}
            fill={SERIES[0]}
            opacity={hoverI === null || hoverI === i ? 0.85 : 0.4}
          />
        ))}
        <polyline
          points={xgPts.join(" ")}
          fill="none"
          stroke={SERIES[1]}
          strokeWidth={2}
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {lastXg ? (
          <text x={x(lastXg.i) + 6} y={y(lastXg.v) + 3} fontSize={9.5} fill={INK_MID} fontFamily={MONO}>
            xG
          </text>
        ) : null}
        {data.map((d, i) =>
          i % Math.ceil(n / 11) === 0 ? (
            <text key={d.gw} x={x(i)} y={VH - 5} textAnchor="middle" fontSize={9} fill={INK_DIM} fontFamily={MONO}>
              {d.gw}
            </text>
          ) : null,
        )}
        <rect
          x={pad.l}
          y={pad.t}
          width={iw}
          height={ih}
          fill="transparent"
          onPointerMove={onMove}
          onPointerLeave={() => {
            setTip(null);
            setHoverI(null);
          }}
        />
      </svg>
      {tip ? <Tip tip={tip} /> : null}
      <div className="mt-1 flex gap-4 px-1">
        <span className="inline-flex items-center gap-1.5 font-mono text-[10px] text-ink-dim">
          <span aria-hidden="true" className="inline-block h-[7px] w-[7px] rounded-[2px]" style={{ background: SERIES[0] }} />
          FPL POINTS
        </span>
        <span className="inline-flex items-center gap-1.5 font-mono text-[10px] text-ink-dim">
          <span aria-hidden="true" className="inline-block h-[2px] w-3 rounded-full" style={{ background: SERIES[1] }} />
          xG PER GW
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// component sparkbars — xP decomposition as a stacked strip
// ---------------------------------------------------------------------------

export interface XpComponents {
  xp_appearance: number;
  xp_goals: number;
  xp_assists: number;
  xp_cs: number;
  xp_saves: number;
  xp_defcon: number;
  xp_bonus: number;
  xp_other: number;
}

const SEGMENTS: { key: keyof XpComponents; label: string; color: string }[] = [
  { key: "xp_appearance", label: "APP", color: "#4c4c47" },
  { key: "xp_goals", label: "GOL", color: SERIES[0] },
  { key: "xp_assists", label: "AST", color: SERIES[2] },
  { key: "xp_cs", label: "CS", color: SERIES[1] },
  { key: "xp_bonus", label: "BON", color: SERIES[3] },
  { key: "xp_saves", label: "SAV", color: "#6f6f68" },
  { key: "xp_defcon", label: "DC", color: "#8b8b83" },
  { key: "xp_other", label: "OTH", color: "#3a3a37" },
];

/** Stacked xP-component strip; `max` fixes the scale across rows. */
export function Sparkbar({
  components,
  max,
  height = 6,
}: {
  components: XpComponents;
  max: number;
  height?: number;
}) {
  const title = SEGMENTS.filter((s) => components[s.key] > 0.005)
    .map((s) => `${s.label} ${components[s.key].toFixed(2)}`)
    .join(" · ");
  return (
    <div
      className="flex w-full overflow-hidden rounded-full bg-raised"
      style={{ height }}
      title={title}
    >
      {SEGMENTS.map((s) => {
        const v = components[s.key];
        if (v <= 0) return null;
        return (
          <div
            key={s.key}
            style={{ width: `${(v / max) * 100}%`, background: s.color }}
          />
        );
      })}
    </div>
  );
}

export function SparkbarLegend() {
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1">
      {SEGMENTS.slice(0, 5).map((s) => (
        <span key={s.key} className="inline-flex items-center gap-1 font-mono text-[9.5px] text-ink-dim">
          <span aria-hidden="true" className="inline-block h-[6px] w-[6px] rounded-[1.5px]" style={{ background: s.color }} />
          {s.label}
        </span>
      ))}
      <span className="inline-flex items-center gap-1 font-mono text-[9.5px] text-ink-dim">
        <span aria-hidden="true" className="inline-block h-[6px] w-[6px] rounded-[1.5px] bg-[#6f6f68]" />
        OTH
      </span>
    </div>
  );
}
