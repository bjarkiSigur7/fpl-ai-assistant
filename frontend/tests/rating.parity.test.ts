/**
 * Parity suite: the pure-TS rating engine (src/lib/rating.ts) vs the Python
 * engine (backend fplai.optimizer.rating.rate_team).
 *
 * fixtures/rating-parity.json holds 25 seeded random legal squads scored by the
 * REAL Python engine on real 2026-27 data, plus the exact bundle inputs
 * (players.json / xp.json / rating.json shapes) both engines consume. The
 * acceptance bar: TS score within 0.05 of Python on all 25 squads — plus exact
 * agreement on verdict, best XI, formation and captain.
 *
 * Runs on plain `node --test` (Node 22+ type stripping), no dependencies:
 *   npm run test:unit
 */

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import {
  RatingValidationError,
  rateTeam,
  validateSquad,
  verdictOf,
  type RatingInputs,
  type RatingPoolPlayer,
  type RatingScore,
} from "../src/lib/rating.ts";

interface Fixture {
  season: number;
  players: RatingPoolPlayer[];
  xp: { gws: number[]; players: Record<string, number[]> };
  rating: {
    floor_metric: number;
    optimal_metric: number;
    window_gws: number[];
    captain_bonus: boolean;
  };
  cases: { codes: number[]; expected: RatingScore }[];
}

const here = dirname(fileURLToPath(import.meta.url));
const fixture: Fixture = JSON.parse(
  readFileSync(join(here, "fixtures", "rating-parity.json"), "utf8"),
);

const inputs: RatingInputs = {
  season: fixture.season,
  pool: fixture.players,
  xp: fixture.xp,
  anchors: fixture.rating,
};

const SCORE_TOLERANCE = 0.05;

test("parity: TS engine matches Python on all 25 seeded squads", () => {
  assert.equal(fixture.cases.length, 25, "fixture must carry exactly 25 squads");
  for (const [i, { codes, expected }] of fixture.cases.entries()) {
    const got = rateTeam(codes, inputs);
    const label = `case ${i} [${codes.slice(0, 3).join(",")}…]`;

    assert.ok(
      Math.abs(got.score - expected.score) <= SCORE_TOLERANCE,
      `${label}: score ${got.score} vs Python ${expected.score}`,
    );
    assert.equal(got.verdict, expected.verdict, `${label}: verdict`);
    assert.equal(got.from_gw, expected.from_gw, `${label}: from_gw`);
    assert.equal(got.horizon, expected.horizon, `${label}: horizon`);
    assert.deepEqual(got.best_xi_gw1, expected.best_xi_gw1, `${label}: best XI`);
    assert.equal(got.formation_gw1, expected.formation_gw1, `${label}: formation`);
    assert.equal(got.suggested_captain, expected.suggested_captain, `${label}: captain`);
    assert.ok(
      Math.abs(got.team_xp_gw1 - expected.team_xp_gw1) <= 0.01,
      `${label}: team_xp_gw1 ${got.team_xp_gw1} vs ${expected.team_xp_gw1}`,
    );
    assert.ok(
      Math.abs(got.team_xp_horizon - expected.team_xp_horizon) <= 0.05,
      `${label}: team_xp_horizon ${got.team_xp_horizon} vs ${expected.team_xp_horizon}`,
    );
    // Anchors are the bundle's 3dp constants; Python carries full precision.
    assert.ok(
      Math.abs(got.optimal_xp_horizon - expected.optimal_xp_horizon) <= 0.01,
      `${label}: optimal anchor`,
    );
    assert.ok(
      Math.abs(got.floor_xp_horizon - expected.floor_xp_horizon) <= 0.01,
      `${label}: floor anchor`,
    );
    assert.deepEqual(
      got.weakest.map((w) => w.player_code),
      expected.weakest.map((w) => w.player_code),
      `${label}: weakest links`,
    );
    assert.deepEqual(
      got.player_ratings.map((p) => [p.player_code, p.in_best_xi_gw1]),
      expected.player_ratings.map((p) => [p.player_code, p.in_best_xi_gw1]),
      `${label}: per-player XI membership`,
    );
  }
});

test("validateSquad mirrors the Python rule strings", () => {
  const pool = fixture.players;
  const legal = fixture.cases[0].codes;
  assert.deepEqual(validateSquad(legal, pool), []);

  // Too few players + an unknown code.
  const short = [...legal.slice(0, 13), 999999999];
  const problems = validateSquad(short, pool);
  assert.ok(problems.includes("need exactly 15 players, got 14"), problems.join("; "));
  assert.ok(problems.includes("unknown player_code 999999999"), problems.join("; "));

  // Duplicate: replace the last player with a repeat of the first.
  const dup = [...legal.slice(0, 14), legal[0]];
  const dupProblems = validateSquad(dup, pool);
  assert.ok(
    dupProblems.includes(`duplicate player_code ${legal[0]}`),
    dupProblems.join("; "),
  );

  // rateTeam throws with every violation listed.
  assert.throws(
    () => rateTeam(short, inputs),
    (err: unknown) =>
      err instanceof RatingValidationError && err.problems.length === problems.length,
  );
});

test("verdict bands: >=95 ELITE, >=85 STRONG, >=70 SOLID, >=50 ROUGH, else FODDER", () => {
  assert.equal(verdictOf(100), "ELITE");
  assert.equal(verdictOf(95), "ELITE");
  assert.equal(verdictOf(94.999), "STRONG");
  assert.equal(verdictOf(85), "STRONG");
  assert.equal(verdictOf(84.999), "SOLID");
  assert.equal(verdictOf(70), "SOLID");
  assert.equal(verdictOf(69.999), "ROUGH");
  assert.equal(verdictOf(50), "ROUGH");
  assert.equal(verdictOf(49.999), "FODDER");
  assert.equal(verdictOf(0), "FODDER");
});
