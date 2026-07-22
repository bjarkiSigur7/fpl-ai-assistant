"use client";

/**
 * THE VERDICT CARD — the hero of the home page. One huge word in display type
 * (pitch-green: the single accent, spent here), a one-line reason, the xP line,
 * and a stagger-reveal of supporting rationale bullets.
 */

import type { Recommendation } from "@/lib/types";
import { chipName } from "@/lib/types";
import { signed1, xp1 } from "@/lib/format";
import { Card, Skeleton } from "./ui";

function verdictWord(action: string): string {
  if (action === "hold") return "HOLD.";
  if (action === "transfer") return "TRANSFER.";
  if (action === "initial-squad") return "ASSEMBLE.";
  if (action.startsWith("chip:")) {
    return `${chipName(action.slice(5)).toUpperCase()}.`;
  }
  return action.toUpperCase();
}

function subline(rec: Recommendation): string {
  if (rec.action === "initial-squad") {
    return "Pre-season build: the optimal fresh-£100m squad from the multi-GW solve.";
  }
  return rec.rationale[0] ?? "The optimal plan for this gameweek.";
}

export function VerdictCard({ rec }: { rec: Recommendation }) {
  const transferDelta = rec.transfers.reduce((s, t) => s + t.xp_delta, 0);
  const bullets =
    rec.action === "initial-squad" ? rec.rationale.slice(1) : rec.rationale.slice(1);

  return (
    <Card className="relative overflow-hidden">
      {/* faint oversize GW watermark, pure chrome */}
      <div
        aria-hidden="true"
        className="display pointer-events-none absolute -right-4 -top-7 select-none text-[130px] text-ink/[0.04]"
      >
        GW{rec.gw}
      </div>
      <div className="relative px-5 py-6 sm:px-7 sm:py-8">
        <div className="microlabel reveal mb-3" style={{ animationDelay: "0ms" }}>
          THE VERDICT — GW{rec.gw} · SEASON {rec.season}-
          {String((rec.season + 1) % 100).padStart(2, "0")}
        </div>
        <h2
          className="display reveal text-pitch-bright text-[min(13vw,92px)]"
          style={{ animationDelay: "60ms" }}
        >
          {verdictWord(rec.action)}
        </h2>
        <p
          className="reveal mt-3 max-w-2xl text-[14px] leading-relaxed text-ink-mid"
          style={{ animationDelay: "160ms" }}
        >
          {subline(rec)}
        </p>

        <div
          className="reveal mt-5 flex flex-wrap items-baseline gap-x-6 gap-y-2 font-mono text-[12px]"
          style={{ animationDelay: "240ms" }}
        >
          <span className="text-ink-dim">
            EXPECTED{" "}
            <span className="tnum text-[17px] font-semibold text-ink">
              {xp1(rec.expected_points)}
            </span>{" "}
            xP THIS GW
          </span>
          {rec.transfers.length > 0 ? (
            <span className="text-ink-dim">
              MOVE VALUE{" "}
              <span
                className={`tnum text-[17px] font-semibold ${transferDelta >= 0 ? "text-pitch-bright" : "text-neg"}`}
              >
                {signed1(transferDelta)}
              </span>{" "}
              xP / HORIZON
            </span>
          ) : null}
          {rec.hits > 0 ? (
            <span className="text-ink-dim">
              HITS <span className="tnum text-[17px] font-semibold text-neg">−{rec.hits}</span>
            </span>
          ) : null}
          <span className="text-ink-dim">
            PLAN OBJ{" "}
            <span className="tnum text-[17px] font-semibold text-ink">{xp1(rec.objective)}</span>
          </span>
        </div>

        {bullets.length > 0 ? (
          <ul className="mt-6 space-y-2 border-t border-hairline-soft pt-5">
            {bullets.map((b, i) => (
              <li
                key={i}
                className="reveal flex gap-3 text-[13px] leading-relaxed text-ink-mid"
                style={{ animationDelay: `${320 + i * 90}ms` }}
              >
                <span aria-hidden="true" className="mt-[2px] font-mono text-ink-dim">
                  {String(i + 1).padStart(2, "0")}
                </span>
                <span>{b}</span>
              </li>
            ))}
          </ul>
        ) : null}
      </div>
    </Card>
  );
}

export function VerdictSkeleton() {
  return (
    <Card className="px-5 py-6 sm:px-7 sm:py-8">
      <Skeleton className="mb-4 h-3 w-44" />
      <Skeleton className="h-20 w-72 max-w-full" />
      <Skeleton className="mt-4 h-4 w-96 max-w-full" />
      <div className="mt-6 space-y-2.5">
        <Skeleton className="h-3.5 w-80 max-w-full" />
        <Skeleton className="h-3.5 w-72 max-w-full" />
        <Skeleton className="h-3.5 w-64 max-w-full" />
      </div>
    </Card>
  );
}
