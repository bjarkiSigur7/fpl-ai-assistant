import { PlayerDetailClient } from "./PlayerDetailClient";

/**
 * Player detail route. The view itself is a client component (SWR-fed); this
 * thin server wrapper exists so static (public) builds can export the route.
 *
 * NEXT_PUBLIC_STATIC=1 builds prerender one page per player in the bundle: the
 * GitHub Actions workflow copies `fplai publish-static` output into public/data/
 * BEFORE `next build`, and generateStaticParams reads players.json from there at
 * build time. Local builds return no params and render the route on demand.
 *
 * Dataless first boot: deploy-pages deliberately allows a push-to-main deploy
 * BEFORE the first successful model-run (no bundle exists yet anywhere) — in
 * that case no public/data/ bundle is present at all and we export the site
 * shell (output:"export" refuses an empty param list, so a single placeholder
 * page is emitted; its client view renders the normal empty state). A PARTIAL
 * bundle (meta.json present but players.json missing) is a corrupt copy and
 * still fails the build loudly.
 */
export async function generateStaticParams(): Promise<{ code: string }[]> {
  if (process.env.NEXT_PUBLIC_STATIC !== "1") return [];
  const { existsSync, readFileSync } = await import("node:fs");
  const { join } = await import("node:path");
  const dataDir = join(process.cwd(), "public", "data");
  const playersPath = join(dataDir, "players.json");
  if (!existsSync(playersPath)) {
    if (existsSync(join(dataDir, "meta.json"))) {
      throw new Error(
        `static build: ${playersPath} is missing but meta.json is present — ` +
          "corrupt/partial bundle copy; refusing to export without player pages",
      );
    }
    console.warn(
      "static build: no data bundle in public/data/ — exporting the site shell " +
        "with a single placeholder player page (dataless first-boot deploy). " +
        "Copy the `fplai publish-static` bundle into frontend/public/data/ " +
        "before `next build` for a full export.",
    );
    return [{ code: "0" }];
  }
  const players = JSON.parse(readFileSync(playersPath, "utf8")) as {
    player_code: number;
  }[];
  return players.map((p) => ({ code: String(p.player_code) }));
}

export default function PlayerDetailPage() {
  return <PlayerDetailClient />;
}
