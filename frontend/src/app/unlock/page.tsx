"use client";

/**
 * Dedicated key screen (the locked nav's ENTER KEY). Keyless visitors are
 * bounced from "/" to the open AI Rating tool, so the full-page gate needs its
 * own address; unlocking here lands on the dashboard.
 */

import { useRouter } from "next/navigation";
import { useEffect } from "react";
import { GateScreen } from "@/components/Gate";
import { useGate } from "@/lib/gate";

export default function UnlockPage() {
  const router = useRouter();
  const { ready, locked } = useGate();
  useEffect(() => {
    if (ready && !locked) router.replace("/");
  }, [ready, locked, router]);
  if (!locked) return null;
  return <GateScreen />;
}
