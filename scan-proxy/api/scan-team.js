/**
 * SCAN PROXY — the one server-side piece the public static desk needs.
 *
 * The GitHub Pages build has no backend, so this tiny Vercel function holds the
 * Gemini key (env GEMINI_API_KEY) and does exactly one thing: forward a squad
 * screenshot to gemini-3.7-flash and answer the recognized player cards as JSON.
 * Matching cards onto the roster happens client-side in the desk (lib/scanMatch.ts)
 * against the player bundle the page already loaded — the proxy stays stateless
 * and data-free.
 *
 * Contract (mirrors backend/src/fplai/data/gemini.py):
 *   POST {image_base64, mime_type} -> 200 {cards: [{name, club?, price?, position?}], model}
 *   503 key not configured · 502 Gemini failed · 422 bad payload · 405 non-POST
 *
 * CORS: locked to the desk's origins; override with env ALLOWED_ORIGINS
 * (comma-separated) when the site moves to a custom domain.
 */

const BASE_URL = "https://generativelanguage.googleapis.com/v1beta";
const MODEL = "gemini-3.7-flash";

const DEFAULT_ORIGINS = [
  "https://bjarkisigur7.github.io",
  "http://localhost:3000",
  "http://127.0.0.1:3000",
];

const VALID_POSITIONS = new Set(["GKP", "DEF", "MID", "FWD"]);
const POSITION_ALIASES = {
  GK: "GKP",
  GOALKEEPER: "GKP",
  DEFENDER: "DEF",
  MIDFIELDER: "MID",
  FORWARD: "FWD",
  STRIKER: "FWD",
};

const PROMPT = `This is a screenshot of a Fantasy Premier League (FPL) squad — 15 player cards laid
out as goalkeeper/defender/midfielder/forward rows plus a bench. For EVERY player
card visible, extract:

- name: the player name exactly as printed on the card (e.g. "M.Salah", "Van Dijk").
- position: which row/section the card sits in — one of GKP, DEF, MID, FWD.
  The bench lists a position label per player; use it.
- price: the price in millions if printed on the card (e.g. 12.7 for "£12.7").
  Omit if the card shows points or a fixture instead of a price.
- club: the player's OWN club as a short name/abbreviation, only if you can tell
  from the shirt or badge. IMPORTANT: the small text under the name is usually the
  upcoming FIXTURE (the opponent, e.g. "MCI (H)") — never report that as the club.
  Omit the field when unsure.

Return every card you can see, in reading order. Do not invent players; if a card
is cut off or unreadable, skip it.`;

const RESPONSE_SCHEMA = {
  type: "object",
  properties: {
    players: {
      type: "array",
      items: {
        type: "object",
        properties: {
          name: { type: "string" },
          position: { type: "string" },
          price: { type: "number" },
          club: { type: "string" },
        },
        required: ["name"],
      },
    },
  },
  required: ["players"],
};

function allowedOrigins() {
  const env = (process.env.ALLOWED_ORIGINS || "").trim();
  return env ? env.split(",").map((s) => s.trim()).filter(Boolean) : DEFAULT_ORIGINS;
}

function applyCors(req, res) {
  const origin = req.headers.origin;
  if (origin && allowedOrigins().includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Max-Age", "86400");
  }
}

async function callGemini(key, imageBase64, mimeType) {
  const headers = { "x-goog-api-key": key, "Content-Type": "application/json" };
  let resp = await fetch(`${BASE_URL}/interactions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: MODEL,
      store: false,
      input: [
        { type: "text", text: PROMPT },
        { type: "image", data: imageBase64, mime_type: mimeType },
      ],
      response_format: { type: "text", mime_type: "application/json", schema: RESPONSE_SCHEMA },
      generation_config: { thinking_level: "low" },
    }),
  });
  if (resp.status === 404) {
    // Interactions surface unavailable — legacy generateContent fallback.
    resp = await fetch(`${BASE_URL}/models/${MODEL}:generateContent`, {
      method: "POST",
      headers,
      body: JSON.stringify({
        contents: [
          {
            parts: [
              { text: PROMPT },
              { inline_data: { mime_type: mimeType, data: imageBase64 } },
            ],
          },
        ],
        generationConfig: {
          responseMimeType: "application/json",
          responseSchema: RESPONSE_SCHEMA,
        },
      }),
    });
  }
  if (!resp.ok) {
    const detail = (await resp.text().catch(() => "")).slice(0, 300);
    throw new Error(`Gemini answered HTTP ${resp.status}: ${detail}`);
  }
  return resp.json();
}

/** Pull the model's text out of an Interactions or generateContent response. */
function extractText(data) {
  if (typeof data.output_text === "string" && data.output_text.trim()) return data.output_text;
  for (const step of [...(data.steps || [])].reverse()) {
    for (const part of [...(step.content || [])].reverse()) {
      if (part && typeof part.text === "string" && part.text.trim()) return part.text;
    }
  }
  for (const cand of data.candidates || []) {
    const parts = (cand.content && cand.content.parts) || [];
    const joined = parts.map((p) => (p && p.text) || "").join("");
    if (joined.trim()) return joined;
  }
  throw new Error("Gemini response contained no text output");
}

function parseCards(text) {
  let cleaned = text.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned.split("\n").slice(1).join("\n");
    cleaned = cleaned.slice(0, cleaned.lastIndexOf("```"));
  }
  let payload;
  try {
    payload = JSON.parse(cleaned);
  } catch {
    throw new Error(`Gemini answered non-JSON: ${cleaned.slice(0, 200)}`);
  }
  const rows = payload && Array.isArray(payload.players) ? payload.players : null;
  if (!rows) throw new Error("Gemini JSON is missing the players array");
  const cards = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") continue;
    const name = String(row.name || "").trim();
    if (!name) continue;
    const price = Number(row.price);
    let position = typeof row.position === "string" ? row.position.trim().toUpperCase() : null;
    position = POSITION_ALIASES[position] || position;
    cards.push({
      name,
      club: row.club ? String(row.club).trim() || null : null,
      // Card prices print in £m (3.8–15.5); anything else is misread noise.
      price: Number.isFinite(price) && price >= 3.0 && price <= 20.0 ? price : null,
      position: VALID_POSITIONS.has(position) ? position : null,
    });
  }
  if (cards.length === 0) throw new Error("Gemini recognized no player cards in the image");
  return cards;
}

export default async function handler(req, res) {
  applyCors(req, res);
  if (req.method === "OPTIONS") return res.status(204).end();
  if (req.method !== "POST") return res.status(405).json({ detail: "POST only" });

  const key = process.env.GEMINI_API_KEY || "";
  if (!key) {
    return res
      .status(503)
      .json({ detail: "scan proxy is not configured yet — Gemini key missing" });
  }

  const body = req.body || {};
  const imageBase64 = typeof body.image_base64 === "string" ? body.image_base64 : "";
  const mimeType = typeof body.mime_type === "string" ? body.mime_type : "image/jpeg";
  if (!imageBase64 || !/^[A-Za-z0-9+/=]+$/.test(imageBase64)) {
    return res.status(422).json({ detail: "image_base64 is missing or not valid base64" });
  }
  if (!/^image\//.test(mimeType)) {
    return res.status(422).json({ detail: "mime_type must be an image type" });
  }

  try {
    const data = await callGemini(key, imageBase64, mimeType);
    const cards = parseCards(extractText(data));
    return res.status(200).json({ cards, model: MODEL });
  } catch (err) {
    return res.status(502).json({ detail: String(err && err.message ? err.message : err) });
  }
}
