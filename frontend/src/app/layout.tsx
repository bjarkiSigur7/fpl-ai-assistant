import type { Metadata } from "next";
import { Archivo_Black, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";
import { GateShell } from "@/components/Gate";
import { Nav } from "@/components/Nav";
import { Ticker } from "@/components/Ticker";

const archivo = Archivo_Black({
  variable: "--font-archivo",
  weight: "400",
  subsets: ["latin"],
});

const plexSans = IBM_Plex_Sans({
  variable: "--font-plex-sans",
  weight: ["400", "500", "600"],
  subsets: ["latin"],
});

const plexMono = IBM_Plex_Mono({
  variable: "--font-plex-mono",
  weight: ["400", "500", "600"],
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PITCHSIDE — the quant desk for football",
  description:
    "FPL AI assistant: expected-points models, MILP squad optimization and chip planning.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${archivo.variable} ${plexSans.variable} ${plexMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        <Ticker />
        <Nav />
        <main className="mx-auto w-full max-w-[1280px] flex-1 px-4 pb-20 pt-6 sm:px-6">
          <GateShell>{children}</GateShell>
        </main>
        <footer className="border-t border-hairline-soft">
          <div className="mx-auto w-full max-w-[1280px] px-4 py-4 sm:px-6">
            <div className="flex items-center justify-between">
              <span className="microlabel">PITCHSIDE / FPL QUANT DESK</span>
              <span className="microlabel">MODELS, NOT VIBES</span>
            </div>
            <p className="mt-2 text-[11px] leading-relaxed text-ink-dim">
              PITCHSIDE is an unofficial, open-source tool and is not affiliated with
              the Premier League or FPL.{" "}
              <a
                href="https://github.com/bjarkiSigur7/fpl-ai-assistant"
                target="_blank"
                rel="noopener noreferrer"
                className="underline decoration-hairline underline-offset-2 hover:text-ink-mid"
              >
                Source on GitHub
              </a>{" "}
              · MIT License.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
