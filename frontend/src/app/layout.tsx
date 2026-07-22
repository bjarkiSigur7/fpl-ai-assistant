import type { Metadata } from "next";
import { Archivo_Black, IBM_Plex_Mono, IBM_Plex_Sans } from "next/font/google";
import "./globals.css";
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
          {children}
        </main>
        <footer className="border-t border-hairline-soft">
          <div className="mx-auto flex w-full max-w-[1280px] items-center justify-between px-4 py-4 sm:px-6">
            <span className="microlabel">PITCHSIDE / FPL QUANT DESK</span>
            <span className="microlabel">MODELS, NOT VIBES</span>
          </div>
        </footer>
      </body>
    </html>
  );
}
