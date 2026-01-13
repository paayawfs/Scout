"use client";

import { useState, useEffect, useCallback } from "react";
import { supabase, Player } from "@/lib/supabase";

interface SearchBarProps {
    onSelect: (player: Player) => void;
    placeholder?: string;
}

export default function SearchBar({ onSelect, placeholder = "Search player... (supports regex e.g. ^Sal)" }: SearchBarProps) {
    const [query, setQuery] = useState("");
    const [results, setResults] = useState<Player[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [showDropdown, setShowDropdown] = useState(false);

    const search = useCallback(async (searchQuery: string) => {
        if (searchQuery.length < 2) {
            setResults([]);
            return;
        }

        setIsLoading(true);
        try {
            // Check if query looks like a regex pattern
            const isRegex = /[\\^$.*+?()[\]{}|]/.test(searchQuery);

            let query = supabase.from("players").select("*");

            if (isRegex) {
                // Use PostgreSQL regex match (case-insensitive with ~*)
                // Escape any problematic characters for safety
                query = query.filter('name', '~*', searchQuery);
            } else {
                // Standard case-insensitive search
                query = query.ilike("name", `%${searchQuery}%`);
            }

            const { data, error } = await query.limit(15);

            if (error) throw error;
            setResults(data || []);
        } catch (err) {
            console.error("Search error:", err);
            setResults([]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        const debounce = setTimeout(() => {
            search(query);
        }, 300);

        return () => clearTimeout(debounce);
    }, [query, search]);

    const handleSelect = (player: Player) => {
        setQuery("");
        setResults([]);
        setShowDropdown(false);
        onSelect(player);
    };

    return (
        <div className="relative w-full max-w-xl">
            <div className="relative">
                {/* Search Icon */}
                <svg
                    className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-accent"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                    />
                </svg>

                <input
                    type="text"
                    value={query}
                    onChange={(e) => {
                        setQuery(e.target.value);
                        setShowDropdown(true);
                    }}
                    onFocus={() => setShowDropdown(true)}
                    placeholder={placeholder}
                    className="search-hero w-full !pl-12 !pr-12"
                />

                {isLoading && (
                    <div className="absolute right-4 top-1/2 -translate-y-1/2">
                        <div className="spinner" />
                    </div>
                )}
            </div>

            {showDropdown && results.length > 0 && (
                <div className="absolute z-50 w-full mt-2 bg-white border-2 border-black rounded-xl shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] max-h-80 overflow-y-auto">
                    {results.map((player) => (
                        <button
                            key={player.id}
                            onClick={() => handleSelect(player)}
                            className="w-full px-4 py-3 text-left hover:bg-gray-50 border-b border-gray-100 last:border-b-0 transition-colors first:rounded-t-xl last:rounded-b-xl"
                        >
                            <div className="flex justify-between items-center">
                                <div>
                                    <p className="text-primary font-medium">{player.name}</p>
                                    <p className="text-sm text-gray-500">
                                        {player.squad} · {player.position}
                                    </p>
                                </div>
                                {player.nation && (
                                    <span className="text-xs text-gray-400 uppercase tracking-wide bg-gray-100 px-2 py-1 rounded">
                                        {player.nation}
                                    </span>
                                )}
                            </div>
                        </button>
                    ))}
                </div>
            )}

            {showDropdown && query.length >= 2 && results.length === 0 && !isLoading && (
                <div className="absolute z-50 w-full mt-2 bg-white border-2 border-black rounded-xl shadow-[4px_4px_0px_0px_rgba(0,0,0,1)] p-4 text-center text-gray-500 font-bold">
                    No players found
                </div>
            )}
        </div>
    );
}
