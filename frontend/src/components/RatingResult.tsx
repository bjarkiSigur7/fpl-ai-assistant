"use client";

/**
 * RATING RESULT — the evaluate moment. A huge count-up percentage in display
 * type (reduced-motion aware), the band-colored verdict word (mark + word,
 * never color alone), the one-axis benchmark bar (model XV vs your XV vs the
 * cheapest-legal floor, thin marks, direct labels), the suggested captain and
 * the weakest links. Renders under the page title as an aria-live status block.
 */

import { useEffect, useRef, useState } from "react";
import type { RateTeamResponse, RateVerdict } from "@/lib/types";
import { xp1 } from "@/lib/format";
import { Card } from "./ui";

const VERDICT_STYLE: Record<RateVerdict, { color: string; mark: string; blurb: string }> = {
  ELITE: {
    color: "text-pitch-bright",
    mark: "●",
    blurb: "Within touching distance of the model optimum.",
  },
  STRONG: {
    color: "text-pitch",
    mark: "●",
    blurb: "A serious squad — only marginal xP left on the table.",
  },
  SOLID: {
    color: "text-ink",
    mark: "◆",
    blurb: "Competitive, with clear upgrade paths.",
  },
  ROUGH: {
    color: "text-warn",
    mark: "▲",
    blurb: "Structural weaknesses are leaking expected points.",
  },
  FODDER: {
    color: "text-crit",
    mark: "■",
    blurb: "Closer to the floor than the frontier — rebuild.",
  },
};

/** 0 -> target ease-out count-up; jumps straight to target under reduced motion. */
function useCountUp(target: number, durationMs = 1100): number {
  const [value, setValue] = useState(0);
  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    const t0 = performance.now();
    const tick = (t: number) => {
      if (reduced) {
        setValue(target);
        return;
      }
      const f = Math.min((t - t0) / durationMs, 1);
      setValue(target * (1 - (1 - f) ** 3));
      if (f < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, durationMs]);
  return value;
}

/** One shared axis, three thin marks: the live series vs two reference grays. */
function BenchmarkBar({ result }: { result: RateTeamResponse }) {
  const rows = [
    { label: "MODEL XV", value: result.optimal_xp_horizon, color: "#6f6f68", live: false },
    { label: "YOUR XV", value: result.team_xp_horizon, color: "#3987e5", live: true },
    { label: "FLOOR", value: result.floor_xp_horizon, color: "#4c4c47", live: false },
  ];
  const axisMax = Math.max(...rows.map((r) => r.value)) * 1.04;
  return (
    <div>
      <div className="microlabel mb-2.5">
        BENCHMARK — WINDOW xP, BEST XI + CAPTAIN PER GW
      </div>
      <div className="space-y-2.5">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center gap-3">
            <span
              className={`microlabel w-[72px] shrink-0 ${r.live ? "!text-ink" : ""}`}
            >
              {r.label}
            </span>
            <div className="h-[7px] flex-1 overflow-hidden rounded-full bg-raised">
              <div
                className="h-full rounded-full"
                style={{
                  width: `${Math.min((r.value / axisMax) * 100, 100)}%`,
                  background: r.color,
                }}
              />
            </div>
            <span
              className={`w-14 shrink-0 text-right font-mono text-[11.5px] tnum ${
                r.live ? "font-semibold text-ink" : "text-ink-mid"
              }`}
            >
              {xp1(r.value)}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex items-center gap-3">
        <span className="w-[72px] shrink-0" aria-hidden="true" />
        <div className="flex flex-1 justify-between border-t border-hairline-soft pt-1 font-mono text-[9px] text-ink-dim">
          <span>0</span>
          <span className="tnum">{xp1(axisMax)} xP</span>
        </div>
        <span className="w-14 shrink-0" aria-hidden="true" />
      </div>
    </div>
  );
}

export function RatingResult({
  result,
  nameOf,
}: {
  result: RateTeamResponse;
  nameOf: (code: number) => string;
}) {
  const style = VERDICT_STYLE[result.verdict];
  const score = useCountUp(result.score);
  const lastGw = result.from_gw + result.horizon - 1;
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    ref.current?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  }, []);

  return (
    <div ref={ref} role="status" className="scroll-mt-4">
      <span className="sr-only">
        Squad rated {result.score.toFixed(1)} out of 100 — {result.verdict}.
      </span>
      <Card className="relative overflow-hidden">
        {/* faint oversize watermark, pure chrome */}
        <div
          aria-hidden="true"
          className="display pointer-events-none absolute -right-4 -top-7 select-none text-[130px] text-ink/[0.04]"
        >
          /100
        </div>
        <div className="relative px-5 py-6 sm:px-7 sm:py-8">
          <div className="microlabel reveal mb-2" style={{ animationDelay: "0ms" }}>
            THE RATING — SEASON {result.season}-
            {String((result.season + 1) % 100).padStart(2, "0")} · GW{result.from_gw}–{lastGw} (
            {result.horizon} GW WINDOW)
          </div>

          <div className="flex flex-wrap items-end gap-x-10 gap-y-4">
            <div>
              <div
                aria-hidden="true"
                className="reveal flex items-baseline"
                style={{ animationDelay: "60ms" }}
              >
                <span className="display tnum text-[min(22vw,118px)]">{score.toFixed(1)}</span>
                <span className="display ml-2 text-[24px] !text-ink-dim">/100</span>
              </div>
              <div
                className={`display reveal mt-1 text-[30px] sm:text-[36px] ${style.color}`}
                style={{ animationDelay: "220ms" }}
              >
                <span className="mr-2 text-[0.6em] align-middle" aria-hidden="true">
                  {style.mark}
                </span>
                {result.verdict}.
              </div>
              <p
                className="reveal mt-2 max-w-xl text-[13.5px] leading-relaxed text-ink-mid"
                style={{ animationDelay: "300ms" }}
              >
                {style.blurb}
              </p>
            </div>

            <div
              className="reveal min-w-[260px] flex-1 space-y-5 sm:min-w-[320px]"
              style={{ animationDelay: "380ms" }}
            >
              <BenchmarkBar result={result} />
              <div className="flex flex-wrap items-baseline gap-x-6 gap-y-2 border-t border-hairline-soft pt-4 font-mono text-[11px] text-ink-dim">
                <span>
                  GW{result.from_gw} xP{" "}
                  <span className="tnum text-[15px] font-semibold text-ink">
                    {xp1(result.team_xp_gw1)}
                  </span>
                </span>
                <span>
                  XI <span className="text-[13px] font-semibold text-ink">{result.formation_gw1}</span>
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span
                    aria-hidden="true"
                    className="flex h-[15px] w-[15px] items-center justify-center rounded-full bg-pitch text-[9px] font-bold text-[#0d0d0d]"
                  >
                    C
                  </span>
                  CAPTAIN{" "}
                  <span className="text-[13px] font-semibold text-ink">
                    {nameOf(result.suggested_captain)}
                  </span>
                </span>
              </div>
            </div>
          </div>

          {result.weakest.length > 0 ? (
            <div
              className="reveal mt-6 border-t border-hairline-soft pt-4"
              style={{ animationDelay: "480ms" }}
            >
              <div className="microlabel mb-2">WEAKEST LINKS — WINDOW xP</div>
              <ul className="flex flex-wrap gap-x-8 gap-y-1.5">
                {result.weakest.map((w) => (
                  <li key={w.player_code} className="flex items-baseline gap-2 text-[13px]">
                    <span aria-hidden="true" className="font-mono text-[11px] text-neg">
                      ▼
                    </span>
                    <span className="font-medium text-ink">{w.web_name}</span>
                    <span className="font-mono text-[11.5px] tnum text-ink-dim">
                      {xp1(w.xp_horizon)} xP
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </Card>
    </div>
  );
}
