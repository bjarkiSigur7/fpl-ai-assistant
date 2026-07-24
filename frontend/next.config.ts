import type { NextConfig } from "next";

/**
 * Dual-mode build:
 *
 * - Local (default): a normal Next.js app against the FastAPI backend at
 *   NEXT_PUBLIC_API_URL. Nothing here changes local dev.
 * - Static public build (NEXT_PUBLIC_STATIC=1): `output: "export"` for GitHub
 *   Pages. The GitHub Actions workflow copies the `fplai publish-static` bundle
 *   into public/data/ before `next build`, then deploys ./out.
 *
 * NEXT_PUBLIC_BASE_PATH is env-driven so the eventual custom-domain cutover is
 * config-only: Actions passes "/fpl-ai-assistant" for the project-pages URL
 * (bjarkisigur7.github.io/fpl-ai-assistant) and drops it once a domain exists.
 */
const isStatic = process.env.NEXT_PUBLIC_STATIC === "1";
const basePath = process.env.NEXT_PUBLIC_BASE_PATH ?? "";

const nextConfig: NextConfig = {
  ...(isStatic
    ? {
        output: "export" as const,
        // Pages has no image optimizer; the desk uses no next/image today, but
        // keep the export self-contained if one ever lands.
        images: { unoptimized: true },
      }
    : {}),
  ...(basePath !== "" ? { basePath } : {}),
};

export default nextConfig;
