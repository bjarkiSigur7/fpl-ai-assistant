"use client";

/** 1 Hz clock as an external store — hydration-safe (null on the server pass). */

import { useSyncExternalStore } from "react";

function subscribe(callback: () => void): () => void {
  const id = setInterval(callback, 1000);
  return () => clearInterval(id);
}

const getSnapshot = () => Math.floor(Date.now() / 1000);
const getServerSnapshot = () => 0;

/** Current time ticking every second; null until mounted on the client. */
export function useNow(): Date | null {
  const seconds = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  return seconds === 0 ? null : new Date(seconds * 1000);
}
