"use client";

/**
 * SCROLLFADE — horizontal scroll container with a hidden scrollbar and a
 * right-edge fade (optionally a subtle "→") that appears only while more
 * content lies off-screen to the right. Used for the nav rail, the players
 * table and the planner's multi-GW cards.
 */

import { useEffect, useRef, useState, type ReactNode } from "react";

export function ScrollFade({
  children,
  className = "",
  innerClassName = "",
  fade = "surface",
  arrow = false,
}: {
  children: ReactNode;
  className?: string;
  /** Extra classes for the scrollable element itself (e.g. scroll-snap). */
  innerClassName?: string;
  /** The chrome color the fade dissolves into — the scroll area's backdrop. */
  fade?: "surface" | "bg";
  /** Render a subtle "→" inside the fade as an extra affordance. */
  arrow?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [more, setMore] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const update = () =>
      setMore(el.scrollLeft + el.clientWidth < el.scrollWidth - 1);
    update();
    el.addEventListener("scroll", update, { passive: true });
    const ro = new ResizeObserver(update);
    ro.observe(el);
    for (const child of el.children) ro.observe(child);
    return () => {
      el.removeEventListener("scroll", update);
      ro.disconnect();
    };
  }, []);

  const color = fade === "surface" ? "var(--color-surface)" : "var(--color-bg)";

  return (
    <div className={`relative min-w-0 ${className}`}>
      <div ref={ref} className={`noscrollbar overflow-x-auto ${innerClassName}`}>
        {children}
      </div>
      <div
        aria-hidden="true"
        className={`pointer-events-none absolute inset-y-0 right-0 flex w-12 items-center justify-end transition-opacity duration-200 ${
          more ? "opacity-100" : "opacity-0"
        }`}
        style={{ background: `linear-gradient(to left, ${color} 15%, transparent)` }}
      >
        {arrow ? (
          <span className="pr-1.5 font-mono text-[12px] text-ink-dim">→</span>
        ) : null}
      </div>
    </div>
  );
}
