"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import SearchBar from "@/components/SearchBar";
import { Player } from "@/lib/supabase";

export default function Home() {
  const router = useRouter();
  const [isSearching, setIsSearching] = useState(false);

  const handlePlayerSelect = (player: Player) => {
    setIsSearching(true);
    router.push(`/player/${player.id}`);
  };

  return (
    <div className="min-h-[80vh] flex flex-col items-center justify-center px-4 py-8 sm:py-12 animate-fadeIn">
      <div className="max-w-2xl w-full text-center">
        {/* Hero */}
        <h1 className="hero-title text-3xl sm:text-4xl md:text-5xl mb-4">
          Find Your Next Signing
        </h1>
        <div className="divider-accent mx-auto mb-6" />
        <p className="text-gray-500 text-base sm:text-lg mb-8 sm:mb-12 leading-relaxed max-w-xl mx-auto px-4">
          Discover statistically similar players using AI-powered analysis.
          Compare performance metrics and find the perfect replacement.
        </p>

        {/* Search */}
        <div className="flex justify-center mb-8 px-2 relative z-20">
          <SearchBar
            onSelect={handlePlayerSelect}
            placeholder="Search for a player by name..."
          />
        </div>

        {isSearching && (
          <div className="flex justify-center">
            <div className="spinner" />
          </div>
        )}

        {/* Features */}
        <div className="grid sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6 mt-12 sm:mt-16 text-left px-2">
          <div className="feature-card group">
            <div className="step-badge">
              <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <h3 className="text-primary text-base sm:text-lg font-semibold mb-2">Search</h3>
            <p className="text-gray-500 text-xs sm:text-sm leading-relaxed">
              Find any player from Europe's top 5 leagues
            </p>
          </div>

          <div className="feature-card group">
            <div className="step-badge">
              <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-primary text-base sm:text-lg font-semibold mb-2">Compare</h3>
            <p className="text-gray-500 text-xs sm:text-sm leading-relaxed">
              View AI-powered similarity scores and stats
            </p>
          </div>

          <div className="feature-card group sm:col-span-2 md:col-span-1">
            <div className="step-badge">
              <svg className="w-5 h-5 sm:w-6 sm:h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4M7.835 4.697a3.42 3.42 0 001.946-.806 3.42 3.42 0 014.438 0 3.42 3.42 0 001.946.806 3.42 3.42 0 013.138 3.138 3.42 3.42 0 00.806 1.946 3.42 3.42 0 010 4.438 3.42 3.42 0 00-.806 1.946 3.42 3.42 0 01-3.138 3.138 3.42 3.42 0 00-1.946.806 3.42 3.42 0 01-4.438 0 3.42 3.42 0 00-1.946-.806 3.42 3.42 0 01-3.138-3.138 3.42 3.42 0 00-.806-1.946 3.42 3.42 0 010-4.438 3.42 3.42 0 00.806-1.946 3.42 3.42 0 013.138-3.138z" />
              </svg>
            </div>
            <h3 className="text-primary text-base sm:text-lg font-semibold mb-2">Discover</h3>
            <p className="text-gray-500 text-xs sm:text-sm leading-relaxed">
              Filter by position, nationality, and age
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="flex flex-wrap justify-center gap-6 sm:gap-12 mt-12 sm:mt-16 pt-6 sm:pt-8 border-t border-gray-200">
          <div className="text-center group">
            <div className="stat-value text-2xl sm:text-3xl">5,000+</div>
            <div className="stat-label mt-1">Players</div>
          </div>
          <div className="text-center group">
            <div className="stat-value text-2xl sm:text-3xl">100+</div>
            <div className="stat-label mt-1">Metrics</div>
          </div>
          <div className="text-center group">
            <div className="stat-value text-2xl sm:text-3xl">5</div>
            <div className="stat-label mt-1">Leagues</div>
          </div>
        </div>
      </div>
    </div>
  );
}
