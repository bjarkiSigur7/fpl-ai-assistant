/**
 * Unit tests for the client-side scan matcher (src/lib/scanMatch.ts) — the TS
 * port of backend fplai/scan.py. Cases mirror backend/tests/test_scan.py so the
 * two matchers keep agreeing on the behaviors that matter.
 *
 * Runs on plain `node --test` (Node 22+ type stripping), no dependencies:
 *   npm run test:unit
 */

import { test } from "node:test";
import assert from "node:assert/strict";

import {
  matchSquad,
  normalizeName,
  type ScanPoolPlayer,
  type SeenCard,
} from "../src/lib/scanMatch.ts";

const POOL: ScanPoolPlayer[] = [
  { player_code: 1001, web_name: "M.Salah", position: "MID", price: 127, team_short: "LIV" },
  { player_code: 1002, web_name: "Van Dijk", position: "DEF", price: 60, team_short: "LIV" },
  { player_code: 1003, web_name: "Ødegaard", position: "MID", price: 82, team_short: "ARS" },
  { player_code: 1004, web_name: "Haaland", position: "FWD", price: 151, team_short: "MCI" },
  // Same web_name at two clubs/prices — club + price must disambiguate.
  { player_code: 1005, web_name: "Ward", position: "GKP", price: 40, team_short: "LIV" },
  { player_code: 1006, web_name: "Ward", position: "DEF", price: 43, team_short: "MCI" },
];

const card = (over: Partial<SeenCard> & { name: string }): SeenCard => ({
  club: null,
  price: null,
  position: null,
  ...over,
});

test("normalizeName strips diacritics and punctuation", () => {
  assert.equal(normalizeName("Ødegaard"), "odegaard");
  assert.equal(normalizeName("M.Salah"), "m salah");
  assert.equal(normalizeName("  Van  Dijk "), "van dijk");
  assert.equal(normalizeName("Włodarczyk"), "wlodarczyk");
  assert.equal(normalizeName("Sørloth"), "sorloth");
});

test("exact names and OCR wobble resolve", () => {
  const got = matchSquad(
    [
      card({ name: "M.Salah", position: "MID" }),
      card({ name: "Odegaard", position: "MID" }), // OCR dropped the Ø
      card({ name: "Van Dijk", position: "DEF" }),
    ],
    POOL,
  );
  assert.deepEqual(
    got.map((m) => m.player_code),
    [1001, 1003, 1002],
  );
  assert.equal(got[0].score, 1);
});

test("surname-only containment matches the dotted web_name", () => {
  const got = matchSquad([card({ name: "Salah" })], POOL);
  assert.equal(got[0].player_code, 1001);
});

test("price and club disambiguate duplicate web_names", () => {
  const got = matchSquad(
    [
      card({ name: "Ward", position: "DEF", price: 4.3, club: "MCI" }),
      card({ name: "Ward", position: "GKP", price: 4.0, club: "LIV" }),
    ],
    POOL,
  );
  assert.equal(got[0].player_code, 1006);
  assert.equal(got[1].player_code, 1005);
});

test("duplicate reads claim one code", () => {
  const got = matchSquad([card({ name: "Haaland" }), card({ name: "Haaland" })], POOL);
  const codes = got.map((m) => m.player_code);
  assert.equal(codes.filter((c) => c === 1004).length, 1);
  assert.equal(codes.filter((c) => c === null).length, 1);
});

test("garbage stays unmatched", () => {
  const got = matchSquad([card({ name: "Zzyzx Qwerty" })], POOL);
  assert.equal(got[0].player_code, null);
  assert.equal(got[0].score, 0);
});

test("position mismatch blocks a weak fuzzy match", () => {
  const got = matchSquad([card({ name: "Haland", position: "GKP" })], POOL);
  assert.equal(got[0].player_code, null);
});
