"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { supabase, Player, PlayerStat } from "@/lib/supabase";
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

                // Fetch stats
                const { data: statsData } = await supabase
                    .from("player_stats")
                    .select("*")
                    .eq("player_id", playerId);

                if (statsData) {
                    // Filter out "Goals per Shot" as requested
                    const filteredStats = statsData.filter(s => s.stat_name !== 'Goals per Shot');
                    setStats(filteredStats);
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

            {/* Stats Overview - Show more stats */}
            {stats.length > 0 && (
                <div className="mb-8 sm:mb-12">
                    <h2 className="text-lg sm:text-xl text-primary font-semibold mb-4">Key Statistics</h2>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-3 sm:gap-4">
                        {stats.slice(0, 15).map((stat) => (
                            <div key={stat.stat_key} className="data-card text-center py-3 sm:py-4">
                                <div className="text-xl sm:text-2xl text-primary font-bold mb-1">
                                    {stat.value.toFixed(2)}
                                </div>
                                <div className="stat-label text-xs line-clamp-2">
                                    {stat.stat_name}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

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
