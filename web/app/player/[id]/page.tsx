"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { supabase, Player, PlayerStat, PlayerSeasonStat } from "@/lib/supabase";
import PlayerCard from "@/components/PlayerCard";
import FilterBar from "@/components/FilterBar";
import PlayerInsightsPanel from "@/components/PlayerInsights";
import { ALL_NATIONS } from "@/lib/constants";

interface SimilarPlayer extends Player {
    similarity: number;
}

export default function PlayerPage() {
    const params = useParams();
    const playerId = Number(params.id);

    const [player, setPlayer] = useState<Player | null>(null);
    const [stats, setStats] = useState<PlayerStat[]>([]);
    const [seasonStats, setSeasonStats] = useState<PlayerSeasonStat[]>([]);
    const [similarPlayers, setSimilarPlayers] = useState<SimilarPlayer[]>([]);
    const [filteredPlayers, setFilteredPlayers] = useState<SimilarPlayer[]>([]);
    const [loading, setLoading] = useState(true);
    const [allPositions, setAllPositions] = useState<string[]>([]);
    const [allNations, setAllNations] = useState<string[]>([]);

    useEffect(() => {
        async function fetchData() {
            setLoading(true);
            try {
                // Fetch player
                const { data: playerData } = await supabase
                    .from("players")
                    .select("*")
                    .eq("id", playerId)
                    .single();

                if (playerData) {
                    setPlayer(playerData);
                }

                // Fetch stats (with cache buster to ensure fresh data)
                const { data: statsData } = await supabase
                    .from("player_stats")
                    .select("*")
                    .eq("player_id", playerId)
                    .order("stat_name");

                if (statsData) {
                    const filteredStats = statsData.filter(s => s.stat_name !== 'Goals per Shot');
                    setStats(filteredStats);
                }

                // Fetch per-season stats
                const { data: seasonData } = await supabase
                    .from("player_season_stats")
                    .select("*")
                    .eq("player_id", playerId)
                    .order("season");

                if (seasonData) {
                    setSeasonStats(seasonData);
                }

                setAllNations(ALL_NATIONS);

                // Fetch similar players
                const { data: similarData } = await supabase
                    .from("player_similarity")
                    .select(`
            similarity,
            rank,
            similar_player:players!player_similarity_similar_player_id_fkey (
              id, name, squad, position, age, nation, league
            )
          `)
                    .eq("player_id", playerId)
                    .order("similarity", { ascending: false })
                    .limit(50);

                if (similarData) {
                    const similar = similarData.map((item: any) => ({
                        ...item.similar_player,
                        similarity: item.similarity,
                    }));
                    setSimilarPlayers(similar);
                    setFilteredPlayers(similar);

                    // Extract unique positions from similar players
                    const positions = [...new Set(similar.map((p: any) => p.position).filter(Boolean))];
                    setAllPositions(positions as string[]);
                }
            } catch (err) {
                console.error("Error fetching data:", err);
            } finally {
                setLoading(false);
            }
        }

        if (playerId) {
            fetchData();
        }
    }, [playerId]);

    const handleFilterChange = (filters: {
        position: string | null;
        nation: string | null;
        maxAge: number | null;
    }) => {
        let filtered = [...similarPlayers];

        if (filters.position) {
            filtered = filtered.filter((p) =>
                p.position?.includes(filters.position!)
            );
        }
        if (filters.nation) {
            filtered = filtered.filter((p) => p.nation === filters.nation);
        }
        if (filters.maxAge) {
            filtered = filtered.filter((p) => p.age && p.age <= filters.maxAge!);
        }

        setFilteredPlayers(filtered);
    };

    if (loading) {
        return (
            <div className="min-h-[60vh] flex items-center justify-center">
                <div className="spinner" />
            </div>
        );
    }

    if (!player) {
        return (
            <div className="max-w-6xl mx-auto px-4 py-16 text-center">
                <h1 className="text-2xl text-primary mb-4">Player not found</h1>
                <Link href="/" className="btn-secondary">
                    Back to Search
                </Link>
            </div>
        );
    }

    return (
        <div className="max-w-[1280px] mx-auto px-4 sm:px-6 md:px-12 py-6 sm:py-8 animate-fadeIn">
            {/* Breadcrumb */}
            <div className="text-sm text-gray-500 mb-6 sm:mb-8 flex items-center gap-2 overflow-x-auto">
                <Link href="/" className="hover:text-accent transition-colors whitespace-nowrap">
                    Home
                </Link>
                <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <span className="text-primary font-medium truncate">{player.name}</span>
            </div>

            {/* Player Header */}
            <div className="mb-8 sm:mb-12">
                <h1 className="text-2xl sm:text-3xl md:text-4xl text-primary font-bold mb-2">{player.name}</h1>
                <div className="divider-accent mb-4" />
                <div className="flex flex-wrap gap-2 sm:gap-4 text-gray-600 text-sm sm:text-base">
                    <span className="font-medium">{player.squad}</span>
                    <span className="text-gray-300 hidden sm:inline">•</span>
                    <span className="bg-gray-100 px-2 py-0.5 rounded">{player.position}</span>
                    {player.age && (
                        <span className="bg-gray-100 px-2 py-0.5 rounded">Age {player.age}</span>
                    )}
                    {player.nation && (
                        <span className="bg-accent/10 text-accent px-2 py-0.5 rounded uppercase tracking-wide text-xs sm:text-sm">
                            {player.nation}
                        </span>
                    )}
                </div>
            </div>

            {/* Per-Season Stats Table */}
            {seasonStats.length > 0 && (() => {
                const seasons = [...new Set(seasonStats.map(s => s.season))].sort();
                const statNames = [...new Set(seasonStats.map(s => s.stat_name))];

                // Group stats by category for better organization
                const categories: Record<string, string[]> = {
                    "Attacking": ["Goals", "Assists", "Non-Penalty xG", "xAG"],
                    "Passing": ["Key Passes", "Progressive Passes", "Final Third Passes", "Penalty Area Passes", "Passes Completed", "Passes Attempted"],
                    "Possession": ["Progressive Carries", "Progressive Receives", "Successful Dribbles", "Touches"],
                    "Defensive": ["Tackles", "Tackles Won", "Interceptions", "Blocks", "Clearances"],
                    "Discipline": ["Fouls", "Fouls Drawn", "Yellow Cards", "Red Cards"],
                };

                const categorizedStats = Object.entries(categories)
                    .map(([cat, names]) => ({
                        category: cat,
                        stats: names.filter(n => statNames.includes(n)),
                    }))
                    .filter(c => c.stats.length > 0);

                // Build lookup: season+stat_name -> value
                const lookup: Record<string, number> = {};
                const ninetiesLookup: Record<string, number> = {};
                seasonStats.forEach(s => {
                    lookup[`${s.season}::${s.stat_name}`] = s.value;
                    ninetiesLookup[s.season] = s.nineties;
                });

                const formatSeason = (s: string) => {
                    const [start, end] = s.split("-");
                    return `${start.slice(2)}/${end.slice(2)}`;
                };

                return (
                    <div className="mb-8 sm:mb-12">
                        <h2 className="text-lg sm:text-xl text-primary font-semibold mb-4">Per-Season Statistics</h2>
                        <p className="text-xs text-gray-400 mb-4 font-mono">All values are per 90 minutes</p>
                        <div className="overflow-x-auto border-2 border-black rounded-xl shadow-[4px_4px_0px_0px_#000000]">
                            <table className="w-full text-sm">
                                <thead>
                                    <tr className="bg-gray-50 border-b-2 border-black">
                                        <th className="text-left py-3 px-4 font-bold text-primary">Stat</th>
                                        {seasons.map(s => (
                                            <th key={s} className="text-center py-3 px-4 font-bold text-primary whitespace-nowrap">
                                                {formatSeason(s)}
                                                <div className="text-[10px] font-normal text-gray-400 mt-0.5">
                                                    {ninetiesLookup[s] ? `${ninetiesLookup[s]} 90s` : ""}
                                                </div>
                                            </th>
                                        ))}
                                    </tr>
                                </thead>
                                <tbody>
                                    {categorizedStats.map(({ category, stats: catStats }) => (
                                        <>
                                            <tr key={category}>
                                                <td colSpan={seasons.length + 1} className="bg-gray-100 py-2 px-4 font-bold text-xs uppercase tracking-wider text-gray-500 border-t border-gray-200">
                                                    {category}
                                                </td>
                                            </tr>
                                            {catStats.map(statName => (
                                                <tr key={statName} className="border-t border-gray-100 hover:bg-gray-50 transition-colors">
                                                    <td className="py-2.5 px-4 text-gray-700 font-medium">{statName}</td>
                                                    {seasons.map(s => {
                                                        const val = lookup[`${s}::${statName}`];
                                                        return (
                                                            <td key={s} className="text-center py-2.5 px-4 font-mono tabular-nums">
                                                                {val !== undefined ? val.toFixed(2) : "—"}
                                                            </td>
                                                        );
                                                    })}
                                                </tr>
                                            ))}
                                        </>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                );
            })()}

            {/* Player Insights */}
            {stats.length > 0 && player && (
                <PlayerInsightsPanel
                    playerId={playerId}
                    playerName={player.name}
                    position={player.position || 'Unknown'}
                    squad={player.squad || 'Unknown'}
                    age={player.age || 0}
                    stats={stats}
                />
            )}

            {/* Similar Players */}
            <div>
                <div className="flex flex-wrap items-center justify-between gap-4 mb-4 sm:mb-6">
                    <h2 className="text-lg sm:text-xl text-primary font-semibold">Similar Players</h2>
                    <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                        {filteredPlayers.length} of {similarPlayers.length}
                    </span>
                </div>

                {/* Filters */}
                <div className="card-elevated mb-6 sm:mb-8 relative z-20">
                    <FilterBar
                        positions={allPositions}
                        nations={allNations}
                        onFilterChange={handleFilterChange}
                    />
                </div>

                {/* Results */}
                <div className="grid sm:grid-cols-2 gap-4 sm:gap-6">
                    {filteredPlayers.map((similar) => (
                        <div key={similar.id} className="h-full">
                            <PlayerCard
                                player={similar}
                                similarity={similar.similarity}
                                compareLink={`/compare/${playerId}/${similar.id}`}
                            />
                        </div>
                    ))}
                </div>

                {filteredPlayers.length === 0 && (
                    <div className="card text-center py-8 sm:py-12">
                        <svg className="w-10 h-10 sm:w-12 sm:h-12 mx-auto text-gray-300 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                        </svg>
                        <p className="text-gray-500">
                            No players match the selected filters
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
