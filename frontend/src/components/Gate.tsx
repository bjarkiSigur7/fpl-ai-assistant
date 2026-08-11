"use client";

/**
 * Gate UI: the full-page key screen (every page except AI Rating), the inline
 * key form (rating page's locked LOAD MODEL XV), and the shared lock glyph.
 * Styled as part of the desk — the gate is a first-class surface, not an
 * apology.
 */

import { usePathname, useRouter } from "next/navigation";
import Link from "next/link";
import { useCallback, useEffect, useId, useState, type FormEvent, type ReactNode } from "react";
import { useGate } from "@/lib/gate";

export function LockGlyph({ className = "" }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 12 14"
      aria-hidden="true"
      className={`inline-block h-[11px] w-[10px] ${className}`}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
    >
      <rect x="1.5" y="6" width="9" height="6.5" rx="1.2" fill="currentColor" stroke="none" />
      <path d="M3.5 6V4.2a2.5 2.5 0 0 1 5 0V6" />
    </svg>
  );
}

/** Shared key form: input + unlock button + wrong-key state. */
function KeyForm({
  autoFocus = false,
  compact = false,
  onUnlocked,
}: {
  autoFocus?: boolean;
  compact?: boolean;
  onUnlocked?: () => void;
}) {
  const { unlock } = useGate();
  const inputId = useId();
  const [key, setKey] = useState("");
  const [wrong, setWrong] = useState(false);
  const [checking, setChecking] = useState(false);

  const submit = useCallback(
    async (e: FormEvent) => {
      e.preventDefault();
      if (!key.trim() || checking) return;
      setChecking(true);
      const ok = await unlock(key);
      setChecking(false);
      if (ok) {
        onUnlocked?.();
      } else {
        setWrong(true);
        setKey("");
      }
    },
    [key, checking, unlock, onUnlocked],
  );

  return (
    <form onSubmit={(e) => void submit(e)} className={compact ? "" : "mt-5"}>
      <label htmlFor={inputId} className="microlabel block pb-1.5">
        DESK KEY
      </label>
      <div className="flex gap-2">
        <input
          id={inputId}
          type="password"
          value={key}
          onChange={(e) => {
            setKey(e.target.value);
            setWrong(false);
          }}
          autoFocus={autoFocus}
          autoComplete="off"
          placeholder="••••••"
          className={`w-full min-w-0 rounded border bg-surface px-3 py-2 font-mono text-[14px] tracking-[0.3em] text-ink placeholder:text-ink-dim focus:outline-none ${
            wrong ? "border-crit" : "border-hairline focus:border-ink-dim"
          } ${compact ? "max-w-44 py-1.5 text-[12px]" : ""}`}
        />
        <button
          type="submit"
          disabled={!key.trim() || checking}
          className={`hit relative shrink-0 rounded bg-pitch font-mono font-semibold uppercase tracking-[0.12em] text-[#0d0d0d] transition-opacity disabled:opacity-40 ${
            compact ? "px-3 py-1.5 text-[10px]" : "px-4 py-2 text-[11px]"
          }`}
        >
          {checking ? "…" : "UNLOCK"}
        </button>
      </div>
      <p
        role="status"
        className={`mt-1.5 font-mono text-[10px] uppercase tracking-[0.1em] ${
          wrong ? "text-crit" : "text-transparent"
        }`}
      >
        WRONG KEY — TRY AGAIN
      </p>
    </form>
  );
}

/** Full-page takeover for gated routes. */
export function GateScreen() {
  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <div className="w-full max-w-sm">
        <div className="rounded-md border border-hairline bg-surface p-6">
          <div className="flex items-center justify-between">
            <span className="microlabel">PITCHSIDE / PRIVATE DESK</span>
            <span className="text-ink-dim">
              <LockGlyph className="h-[13px] w-[12px]" />
            </span>
          </div>
          <h1 className="display mt-3 text-[26px] leading-tight">
            THE DESK IS
            <br />
            KEYHOLDERS-ONLY
          </h1>
          <p className="mt-2 text-[12.5px] leading-relaxed text-ink-dim">
            Verdicts, plans and projections are gated for the season. Enter the
            desk key to open everything.
          </p>
          <KeyForm autoFocus />
        </div>
        <Link
          href="/rating"
          className="mt-3 block rounded-md border border-hairline-soft bg-raised/40 px-4 py-3 transition-colors hover:border-hairline hover:bg-raised"
        >
          <span className="microlabel">OPEN WITHOUT A KEY →</span>
          <span className="mt-0.5 block text-[12.5px] text-ink-mid">
            AI RATING — score any 15-man squad against the model, 0-100.
          </span>
        </Link>
      </div>
    </div>
  );
}

/** Inline unlock panel (rating page: locked LOAD MODEL XV). */
export function InlineUnlock({ onUnlocked }: { onUnlocked?: () => void }) {
  return (
    <div className="rounded border border-hairline-soft bg-raised/40 px-3 py-2.5">
      <p className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.1em] text-ink-dim">
        <LockGlyph />
        MODEL XV IS KEYHOLDERS-ONLY — THE PICKER AND SCORING STAY OPEN
      </p>
      <div className="mt-2">
        <KeyForm compact onUnlocked={onUnlocked} />
      </div>
    </div>
  );
}

/**
 * Layout wrapper: gates every route except /rating and /unlock on the public
 * build. Keyless visitors landing on "/" are bounced to the open AI Rating
 * tool instead of a key wall — the full-page key screen lives at /unlock (the
 * nav's ENTER KEY) and still appears in place on gated deep links.
 * Pre-hydration renders the gate (never the desk) — prerendered HTML included.
 */
export function GateShell({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();
  const { ready, locked } = useGate();
  const open =
    pathname === "/rating" ||
    pathname.startsWith("/rating/") ||
    pathname === "/unlock" ||
    pathname.startsWith("/unlock/");
  const bounceToRating = locked && pathname === "/";
  useEffect(() => {
    if (ready && bounceToRating) router.replace("/rating");
  }, [ready, bounceToRating, router]);
  if (bounceToRating) return null;
  if (!locked || open) return <>{children}</>;
  return <GateScreen />;
}
