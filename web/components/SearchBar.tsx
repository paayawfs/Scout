"use client";

import { useState, useEffect, useCallback } from "react";
import { supabase, Player } from "@/lib/supabase";

interface SearchBarProps {
    onSelect: (player: Player) => void;
    placeholder?: string;
}

// Normalize string by removing accents (é → e, ñ → n, etc.)
const normalizeAccents = (str: string): string => {
    return str.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
};

export default function SearchBar({ onSelect, placeholder = "Search player... (regex: ^Sal)" }: SearchBarProps) {
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
            // Normalize query for accent-insensitive search
            const normalizedQuery = normalizeAccents(searchQuery.toLowerCase());

            // Check if query looks like a regex pattern
            const isRegex = /[\\^$.*+?()[\]{}|]/.test(searchQuery);

            let dbQuery = supabase.from("players").select("*");

            if (isRegex) {
                // Use PostgreSQL regex match (case-insensitive with ~*)
                dbQuery = dbQuery.filter('name', '~*', searchQuery);
            } else {
                // Fetch broader results then filter client-side for accents
                // Use first 2 chars for server query, filter rest client-side
                const simpleQuery = normalizedQuery.slice(0, 3);
                dbQuery = dbQuery.ilike("name", `%${simpleQuery}%`);
            }

            const { data, error } = await dbQuery.limit(50);

            if (error) throw error;

            // Filter results client-side for accent-insensitive matching
            let filtered = data || [];
            if (!isRegex && filtered.length > 0) {
                filtered = filtered.filter(player =>
                    normalizeAccents(player.name.toLowerCase()).includes(normalizedQuery)
                ).slice(0, 15);
            }

            setResults(filtered);
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
