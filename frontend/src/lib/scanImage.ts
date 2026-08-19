"use client";

/**
 * Screenshot -> scan-team payload: downscale big phone screenshots on a canvas
 * and re-encode as JPEG so the POST stays small and Gemini answers fast. Falls
 * back to the untouched file when the browser can't decode it (rare formats) —
 * the backend caps decoded size at 10MB either way.
 */

import type { ScanTeamRequest } from "./types";

/** Longest edge after downscale — plenty for Gemini to read card text. */
const MAX_SIDE = 2000;
const JPEG_QUALITY = 0.9;

function dataUrlToRequest(dataUrl: string): ScanTeamRequest {
  const comma = dataUrl.indexOf(",");
  const header = dataUrl.slice(0, comma); // e.g. "data:image/jpeg;base64"
  const mime = header.slice(5, header.indexOf(";"));
  return { image_base64: dataUrl.slice(comma + 1), mime_type: mime || "image/jpeg" };
}

function readRaw(file: File): Promise<ScanTeamRequest> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error ?? new Error("could not read file"));
    reader.onload = () => resolve(dataUrlToRequest(String(reader.result)));
    reader.readAsDataURL(file);
  });
}

export async function encodeScanImage(file: File): Promise<ScanTeamRequest> {
  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file);
  } catch {
    return readRaw(file);
  }
  try {
    const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(1, Math.round(bitmap.width * scale));
    canvas.height = Math.max(1, Math.round(bitmap.height * scale));
    const ctx = canvas.getContext("2d");
    if (!ctx) return readRaw(file);
    // White ground: transparent PNG regions would otherwise turn black in JPEG.
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
    return dataUrlToRequest(canvas.toDataURL("image/jpeg", JPEG_QUALITY));
  } finally {
    bitmap.close();
  }
}
