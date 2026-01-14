import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import ErrorBoundary from "@/components/ErrorBoundary";
import { ToastProvider } from "@/components/ToastProvider";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "NextGen Scout | Find Your Next Signing",
  description: "AI-powered player scouting and similarity analysis. Discover statistically similar football players using advanced neural networks.",
  keywords: ["football", "soccer", "player analysis", "scouting", "replacements", "AI", "analytics"],
};

import { Analytics } from "@vercel/analytics/next";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} ${jetbrainsMono.variable} antialiased`}>
        <ToastProvider>
          <ErrorBoundary>
            {/* Navigation */}
            <nav className="sticky top-0 z-50 border-b-2 border-black bg-white/95 backdrop-blur-md">
              <div className="max-w-[1280px] mx-auto px-4 sm:px-6 md:px-12 py-4">
                <a href="/" className="flex items-center gap-3 group w-fit">
                  <div className="logo-brand font-mono font-bold text-xl border-2 border-black bg-white text-black w-10 h-10 flex items-center justify-center rounded-lg shadow-[2px_2px_0px_0px_#000000]">
                    NGS
                  </div>
                  <div className="flex flex-col">
                    <span className="text-black font-bold text-lg leading-tight uppercase tracking-tight">
                      NextGen Scout
                    </span>
                    <span className="text-xs text-gray-500 font-mono hidden sm:block">
                      AI-Powered Scouting
                    </span>
                  </div>
                </a>
              </div>
            </nav>

            {/* Main Content */}
            <main>{children}</main>
            <Analytics />

            {/* Footer */}
            <footer className="bg-white text-black mt-16 border-t-2 border-black">
              <div className="max-w-[1280px] mx-auto px-6 md:px-12 py-12">
                <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                  <div className="flex items-center gap-3">
                    <div className="logo-brand font-mono font-bold text-xl border-2 border-black bg-white text-black w-10 h-10 flex items-center justify-center rounded-lg shadow-[2px_2px_0px_0px_#000000]">
                      NGS
                    </div>
                    <span className="font-bold text-lg uppercase">NextGen Scout</span>
                  </div>

                  <p className="text-gray-500 text-sm text-center md:text-right font-mono">
                    Data sourced from FBref · Powered by Siamese Neural Networks
                  </p>
                </div>

                <div className="mt-6 text-right">
                  <p className="text-gray-400 text-xs font-mono bg-gray-50 border border-gray-200 rounded-lg px-4 py-2 inline-block">
                    📊 Stats are an average of the 2024/25 and 2025/26 seasons
                  </p>
                </div>

                <div className="mt-8 pt-8 border-t border-gray-200 text-center text-gray-400 text-sm font-mono">
                  © {new Date().getFullYear()} NextGen Scout. All rights reserved.
                </div>
              </div>
            </footer>
          </ErrorBoundary>
        </ToastProvider>
      </body>
    </html>
  );
}
