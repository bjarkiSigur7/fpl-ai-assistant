"use client";

/**
 * FPL entry-id persistence: localStorage-backed with a custom event so every
 * mounted component reacts to changes immediately (settings page -> dashboard).
 */

import { useCallback, useSyncExternalStore } from "react";

const KEY = "fplai.entry_id";
const EVENT = "fplai:entry-id";

function read(): number | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(KEY);
  if (raw === null) return null;
  const n = Number(raw);
  return Number.isInteger(n) && n > 0 ? n : null;
}

let cached: number | null = null;
let cachedRaw: string | null | undefined;

function subscribe(callback: () => void): () => void {
  const handler = () => callback();
  window.addEventListener(EVENT, handler);
  window.addEventListener("storage", handler);
  return () => {
    window.removeEventListener(EVENT, handler);
    window.removeEventListener("storage", handler);
  };
}

function getSnapshot(): number | null {
  const raw = typeof window === "undefined" ? null : window.localStorage.getItem(KEY);
  if (raw !== cachedRaw) {
    cachedRaw = raw;
    cached = read();
  }
  return cached;
}

export function useEntryId(): [number | null, (id: number | null) => void] {
  const entryId = useSyncExternalStore(subscribe, getSnapshot, () => null);
  const setEntryId = useCallback((id: number | null) => {
    if (id === null) window.localStorage.removeItem(KEY);
    else window.localStorage.setItem(KEY, String(id));
    window.dispatchEvent(new Event(EVENT));
  }, []);
  return [entryId, setEntryId];
}
