"use client";

/** Desk chrome: wordmark + primary navigation. */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ScrollFade } from "./ScrollFade";

const LINKS = [
  { href: "/", label: "DASHBOARD" },
  { href: "/players", label: "PLAYERS" },
  { href: "/fixtures", label: "FIXTURES" },
  { href: "/planner", label: "PLANNER" },
  { href: "/rating", label: "AI RATING" },
  { href: "/settings", label: "SETTINGS" },
];

export function Nav() {
  const pathname = usePathname();
  const isActive = (href: string) =>
    href === "/" ? pathname === "/" : pathname.startsWith(href);

  return (
    <div className="border-b border-hairline">
      <div className="mx-auto flex w-full max-w-[1280px] flex-wrap items-center justify-between gap-x-4 px-4 sm:px-6">
        <Link href="/" className="flex shrink-0 items-baseline gap-2.5 py-3.5">
          <span className="display text-[19px] leading-none">PITCHSIDE</span>
          <span className="microlabel hidden sm:inline">FPL QUANT DESK</span>
        </Link>
        <nav className="-mx-1.5 min-w-0 max-w-full sm:mx-0" aria-label="Primary">
          <ScrollFade fade="bg">
            <div className="flex items-center gap-0 sm:gap-1">
              {LINKS.map(({ href, label }) => {
                const active = isActive(href);
                return (
                  <Link
                    key={href}
                    href={href}
                    aria-current={active ? "page" : undefined}
                    className={`relative whitespace-nowrap px-1.5 py-4 font-mono text-[10px] tracking-[0.06em] transition-colors sm:px-3.5 sm:text-[11px] sm:tracking-[0.14em] ${
                      active ? "text-ink" : "text-ink-dim hover:text-ink-mid"
                    }`}
                  >
                    {label}
                    {active ? (
                      <span
                        aria-hidden="true"
                        className="absolute inset-x-1.5 bottom-0 h-[2px] bg-ink sm:inset-x-3.5"
                      />
                    ) : null}
                  </Link>
                );
              })}
            </div>
          </ScrollFade>
        </nav>
      </div>
    </div>
  );
}
