import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const ui = Geist({
  subsets: ["latin"],
  variable: "--font-ui",
  display: "swap",
});

const mono = Geist_Mono({
  subsets: ["latin"],
  variable: "--font-mono-ui",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Ground Control · Reconstruction Studio",
  description:
    "Turn a single drone pass into a georeferenced, measurable 3D workspace. Reconstruct, inspect, measure and plan on your own machine.",
};

export const viewport: Viewport = { themeColor: "#0b0c0f", colorScheme: "dark" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${ui.variable} ${mono.variable}`}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
