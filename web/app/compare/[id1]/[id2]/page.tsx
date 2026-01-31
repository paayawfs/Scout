"use client";

import { useEffect, useState, useMemo } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { supabase, Player, PlayerStat, PlayerSeasonStat } from "@/lib/supabase";
import RadarComparison from "@/components/RadarChart";

interface ComparisonData {
    player1: Player;
    player2: Player;
    stats1: PlayerStat[];
    stats2: PlayerStat[];
    seasonStats1: PlayerSeasonStat[];
    seasonStats2: PlayerSeasonStat[];
    similarity: number;
}

export default function ComparePage() {
    const params = useParams();
    const id1 = Number(params.id1);
    const id2 = Number(params.id2);

    const [data, setData] = useState<ComparisonData | null>(null);
    const [loading, setLoading] = useState(true);
    const [selectedSeason, setSelectedSeason] = useState<string>("average");

    useEffect(() => {
        async function fetchComparison() {
            setLoading(true);
            try {
                // Fetch both players, stats (aggregated + per-season), and similarity
                const [player1Res, player2Res, stats1Res, stats2Res, season1Res, season2Res] = await Promise.all([
                    supabase.from("players").select("*").eq("id", id1).single(),
                    supabase.from("players").select("*").eq("id", id2).single(),
                    supabase.from("player_stats").select("*").eq("player_id", id1),
                    supabase.from("player_stats").select("*").eq("player_id", id2),
                    supabase.from("player_season_stats").select("*").eq("player_id", id1),
                    supabase.from("player_season_stats").select("*").eq("player_id", id2),
                ]);

                const simRes = await supabase
                    .from("player_similarity")
                    .select("similarity")
                    .eq("player_id", id1)
                    .eq("similar_player_id", id2)
                    .single();

                if (player1Res.data && player2Res.data) {
                    setData({
                        player1: player1Res.data,
                        player2: player2Res.data,
                        stats1: stats1Res.data || [],
                        stats2: stats2Res.data || [],
                        seasonStats1: season1Res.data || [],
                        seasonStats2: season2Res.data || [],
                        similarity: simRes.data?.similarity || 0,
                    });
                }
            } catch (err) {
                console.error("Error fetching comparison:", err);
            } finally {
                setLoading(false);
            }
        }

        if (id1 && id2) {
            fetchComparison();
        }
    }, [id1, id2]);

    // Compute available seasons from both players' data
    const availableSeasons = useMemo(() => {
        if (!data) return [];
        const allSeasons = new Set([
            ...data.seasonStats1.map(s => s.season),
            ...data.seasonStats2.map(s => s.season),
        ]);
        return [...allSeasons].sort();
    }, [data]);

    // Build display stats based on selected season
    const displayStats = useMemo(() => {
        if (!data) return { names: [] as string[], values1: [] as number[], values2: [] as number[] };

        if (selectedSeason === "average") {
            const names = data.stats1.map(s => s.stat_name);
            const values1 = data.stats1.map(s => s.value);
            const values2 = names.map(name => {
                const stat = data.stats2.find(s => s.stat_name === name);
                return stat?.value || 0;
            });
            return { names, values1, values2 };
        }

        // Per-season: build from seasonStats
        const s1Map = new Map<string, number>();
        const s2Map = new Map<string, number>();
        data.seasonStats1.filter(s => s.season === selectedSeason).forEach(s => s1Map.set(s.stat_name, s.value));
        data.seasonStats2.filter(s => s.season === selectedSeason).forEach(s => s2Map.set(s.stat_name, s.value));

        // Use stat names from aggregated stats as the canonical list
        const names = data.stats1.map(s => s.stat_name);
        const values1 = names.map(name => s1Map.get(name) ?? 0);
        const values2 = names.map(name => s2Map.get(name) ?? 0);
        return { names, values1, values2 };
    }, [data, selectedSeason]);

    const formatSeason = (s: string) => {
        const [start, end] = s.split("-");
        return `${start.slice(2)}/${end.slice(2)}`;
    };

    if (loading) {
        return (
            <div className="min-h-[60vh] flex items-center justify-center">
                <div className="spinner" />
            </div>
        );
    }

    if (!data) {
        return (
            <div className="max-w-6xl mx-auto px-4 py-16 text-center">
                <h1 className="text-2xl text-primary mb-4">Comparison not found</h1>
                <Link href="/" className="btn-secondary">
                    Back to Search
                </Link>
            </div>
        );
    }

    const { player1, player2, similarity } = data;

    return (
        <div className="max-w-[1280px] mx-auto px-6 md:px-12 py-8 animate-fadeIn">
            {/* Breadcrumb */}
            <div className="text-sm text-gray-500 mb-8 flex items-center gap-2">
                <Link href="/" className="hover:text-accent transition-colors">
                    Home
                </Link>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <Link href={`/player/${id1}`} className="hover:text-accent transition-colors">
                    {player1.name}
                </Link>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
                <span className="text-primary font-medium">Compare</span>
            </div>

            {/* Header */}
            <div className="text-center mb-12">
                <h1 className="text-3xl text-primary font-bold mb-4">Player Comparison</h1>
                <div className="divider-accent mx-auto mb-6" />
                <div className={`similarity-badge ${similarity >= 0.85 ? "high" : ""} text-lg px-6 py-2`}>
                    {(similarity * 100).toFixed(1)}% Similar
                </div>
            </div>

            {/* Players Side by Side */}
            <div className="grid md:grid-cols-2 gap-8 mb-12">
                {/* Player 1 */}
                <div className="card-elevated text-center">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ background: '#FF6B35' }}>
                        <span className="text-white text-2xl font-bold">
                            {player1.name.charAt(0)}
                        </span>
                    </div>
                    <h2 className="text-2xl text-primary font-semibold mb-2">{player1.name}</h2>
                    <p className="text-gray-600 mb-1">{player1.squad}</p>
                    <div className="flex justify-center gap-3 text-sm text-gray-500">
                        <span className="bg-gray-100 px-2 py-0.5 rounded">{player1.position}</span>
                        {player1.age && <span>Age {player1.age}</span>}
                    </div>
                    {player1.nation && (
                        <p className="text-sm text-gray-400 mt-2 uppercase tracking-wide">{player1.nation}</p>
                    )}
                </div>

                {/* Player 2 */}
                <div className="card-elevated text-center">
                    <div className="w-16 h-16 mx-auto mb-4 rounded-full flex items-center justify-center" style={{ background: '#E63946' }}>
                        <span className="text-white text-2xl font-bold">
                            {player2.name.charAt(0)}
                        </span>
                    </div>
                    <h2 className="text-2xl text-primary font-semibold mb-2">{player2.name}</h2>
                    <p className="text-gray-600 mb-1">{player2.squad}</p>
                    <div className="flex justify-center gap-3 text-sm text-gray-500">
                        <span className="bg-gray-100 px-2 py-0.5 rounded">{player2.position}</span>
                        {player2.age && <span>Age {player2.age}</span>}
                    </div>
                    {player2.nation && (
                        <p className="text-sm text-gray-400 mt-2 uppercase tracking-wide">{player2.nation}</p>
                    )}
                </div>
            </div>

            {/* Season Switcher */}
            {availableSeasons.length > 0 && (
                <div className="flex justify-center mb-8">
                    <div className="inline-flex border-2 border-black rounded-lg overflow-hidden shadow-[2px_2px_0px_0px_#000000]">
                        <button
                            onClick={() => setSelectedSeason("average")}
                            className={`px-4 py-2 text-sm font-bold transition-colors ${
                                selectedSeason === "average"
                                    ? "bg-black text-white"
                                    : "bg-white text-gray-700 hover:bg-gray-100"
                            }`}
                        >
                            Average
                        </button>
                        {availableSeasons.map(s => (
                            <button
                                key={s}
                                onClick={() => setSelectedSeason(s)}
                                className={`px-4 py-2 text-sm font-bold border-l-2 border-black transition-colors ${
                                    selectedSeason === s
                                        ? "bg-black text-white"
                                        : "bg-white text-gray-700 hover:bg-gray-100"
                                }`}
                            >
                                {formatSeason(s)}
                            </button>
                        ))}
                    </div>
                </div>
            )}

            {/* Radar Chart */}
            {displayStats.names.length > 0 && (
                <div className="card-elevated mb-12">
                    <h2 className="text-xl text-primary font-semibold mb-6 text-center">
                        Statistical Comparison
                        {selectedSeason !== "average" && (
                            <span className="text-sm font-normal text-gray-400 ml-2">
                                ({formatSeason(selectedSeason)})
                            </span>
                        )}
                    </h2>
                    <RadarComparison
                        player1={{ name: player1.name, values: displayStats.values1, position: player1.position || undefined }}
                        player2={{ name: player2.name, values: displayStats.values2, position: player2.position || undefined }}
                        labels={displayStats.names}
                    />
                </div>
            )}

            {/* Stats Table */}
            <div className="card-elevated">
                <h2 className="text-xl text-primary font-semibold mb-6">
                    Detailed Stats
                    {selectedSeason !== "average" && (
                        <span className="text-sm font-normal text-gray-400 ml-2">
                            ({formatSeason(selectedSeason)})
                        </span>
                    )}
                </h2>
                <div className="overflow-x-auto">
                    <table className="w-full">
                        <thead>
                            <tr className="border-b border-gray-200">
                                <th className="text-left py-3 text-gray-500 text-sm font-medium uppercase tracking-wide">
                                    Statistic
                                </th>
                                <th className="text-center py-3 text-primary font-semibold">
                                    {player1.name}
                                </th>
                                <th className="text-center py-3 text-accent font-semibold">
                                    {player2.name}
                                </th>
                            </tr>
                        </thead>
                        <tbody>
                            {displayStats.names.map((statName, i) => {
                                const val1 = displayStats.values1[i];
                                const val2 = displayStats.values2[i];
                                const diff = val1 - val2;

                                return (
                                    <tr key={statName} className="border-b border-gray-100 hover:bg-gray-50 transition-colors">
                                        <td className="py-3 text-gray-700">{statName}</td>
                                        <td className="py-3 text-center text-primary font-medium">
                                            {val1.toFixed(2)}
                                            {diff > 0 && (
                                                <span className="ml-2 text-xs text-green-500 bg-green-50 px-1.5 py-0.5 rounded">+</span>
                                            )}
                                        </td>
                                        <td className="py-3 text-center text-gray-700">
                                            {val2.toFixed(2)}
                                            {diff < 0 && (
                                                <span className="ml-2 text-xs text-green-500 bg-green-50 px-1.5 py-0.5 rounded">+</span>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Actions */}
            <div className="flex flex-wrap justify-center gap-4 mt-12">
                <Link href={`/player/${id1}`} className="btn-secondary">
                    <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                    </svg>
                    Back to {player1.name}
                </Link>
                <Link href={`/player/${id2}`} className="btn-primary">
                    View {player2.name}
                    <svg className="w-4 h-4 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                </Link>
            </div>
        </div>
    );
}
