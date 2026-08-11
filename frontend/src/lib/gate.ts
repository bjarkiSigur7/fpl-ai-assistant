"use client";

/**
 * Client-side access gate for the public (static) build.
 *
 * Everything except the AI Rating tool sits behind a key; on the rating page
 * only the LOAD MODEL XV shortcut is gated. The key is never shipped in the
 * bundle — entry is verified against a SHA-256 digest via Web Crypto and the
 * digest itself is what unlock state stores, so casual source-reading reveals
 * nothing.
 *
 * HONEST LIMITATION (by architecture): this is a curtain, not a vault. A
 * static site has no server to enforce auth — the underlying JSON bundle
 * stays fetchable by URL and the gate lives in client JS. It keeps the desk
 * out of casual view (mini-league rivals with a link), nothing more.
 */

import { useCallback, useSyncExternalStore } from "react";

/** Gate only the deployed static site; local dev stays open. */
export const GATE_ENABLED: boolean = process.env.NEXT_PUBLIC_STATIC === "1";

const STORE_KEY = "pitchside.gate.v1";

/** SHA-256 hex digest of the access key. */
const KEY_DIGEST = "049ce156305c721ce413cf3178238427e77a4669adc3ae9b50598a0a93216726";

type Snapshot = { ready: boolean; locked: boolean };

const LOCKED_SSR: Snapshot = { ready: false, locked: true };

let snapshot: Snapshot = LOCKED_SSR;
const listeners = new Set<() => void>();

function emit(next: Snapshot): void {
  snapshot = next;
  for (const fn of listeners) fn();
}

function readStorage(): boolean {
  try {
    return window.localStorage.getItem(STORE_KEY) === KEY_DIGEST;
  } catch {
    return false;
  }
}

function subscribe(fn: () => void): () => void {
  listeners.add(fn);
  if (!snapshot.ready && typeof window !== "undefined") {
    // First client subscriber: resolve the persisted unlock state.
    emit({ ready: true, locked: GATE_ENABLED ? !readStorage() : false });
  }
  return () => listeners.delete(fn);
}

async function digestOf(text: string): Promise<string> {
  const data = new TextEncoder().encode(text);
  const hash = await crypto.subtle.digest("SHA-256", data);
  return [...new Uint8Array(hash)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

/** Try a key; persists and unlocks everywhere on success. */
export async function tryUnlock(key: string): Promise<boolean> {
  const ok = (await digestOf(key.trim())) === KEY_DIGEST;
  if (ok) {
    try {
      window.localStorage.setItem(STORE_KEY, KEY_DIGEST);
    } catch {
      // Private-mode storage failure still unlocks this session.
    }
    emit({ ready: true, locked: false });
  }
  return ok;
}

/** Re-lock (used by the settings surface; handy for shared machines). */
export function relock(): void {
  try {
    window.localStorage.removeItem(STORE_KEY);
  } catch {
    // ignore
  }
  emit({ ready: true, locked: GATE_ENABLED });
}

/**
 * Gate state. Pre-hydration (`ready === false`) reads as locked so prerendered
 * HTML ships the gate, never the desk.
 */
export function useGate(): Snapshot & { unlock: (key: string) => Promise<boolean> } {
  const snap = useSyncExternalStore(
    subscribe,
    () => snapshot,
    () => LOCKED_SSR,
  );
  const unlock = useCallback((key: string) => tryUnlock(key), []);
  return { ...snap, locked: GATE_ENABLED && snap.locked, unlock };
}
