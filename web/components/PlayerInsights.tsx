"use client";

import { useState } from "react";
import { PlayerStat } from "@/lib/supabase";

interface GeminiAnalysis {
    summary: string;
    strengths: { stat: string; insight: string }[];
    improvements: { stat: string; insight: string }[];
    playingStyle: string;
    comparison: string;
}

interface PlayerInsightsProps {
    playerId: number;
    playerName: string;
    position: string;
    squad: string;
    age: number;
    stats: PlayerStat[];
}

function calculatePercentile(value: number, min: number, max: number): number {
    if (max === min) return 50;
    return Math.round(Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100)));
}

export default function PlayerInsightsPanel({
    playerId,
    playerName,
    position,
    squad,
    age,
    stats
}: PlayerInsightsProps) {
    const [analysis, setAnalysis] = useState<GeminiAnalysis | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [isLimitError, setIsLimitError] = useState(false);
    const [expanded, setExpanded] = useState(false);
    const [isCached, setIsCached] = useState(false);

    const handleGenerateAnalysis = async () => {
        setLoading(true);
        setError(null);
        setIsLimitError(false);
        setExpanded(true);

        try {
            const statsWithPercentile = stats.map(s => ({
                name: s.stat_name,
                value: s.value,
                percentile: calculatePercentile(s.value, s.min_range, s.max_range),
            }));

            const response = await fetch('/api/insights', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    player: {
                        id: playerId,
                        name: playerName,
                        position,
                        squad,
                        age,
                        stats: statsWithPercentile,
                    },
                }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                if (errorData.error === 'AI_LIMIT_REACHED') {
                    setIsLimitError(true);
                    throw new Error(errorData.friendlyMessage);
                }
                throw new Error(errorData.details || 'Failed to generate analysis');
            }

            const data = await response.json();
            setAnalysis(data.analysis);
            setIsCached(data.cached || false);
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Unknown error';
            setError(message);
            // Only log actual system errors, not limit warnings
            if (!message.includes('break')) {
                console.error(err);
            }
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="mb-8 sm:mb-12">
            {/* Header */}
            <div className="bg-gradient-to-r from-purple-50 to-white border-2 border-black rounded-xl p-6 shadow-[4px_4px_0px_0px_#000000]">
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-xl text-primary font-bold flex items-center gap-2">
                        AI Player Analysis
                    </h2>
                    {!analysis && !loading && (
                        <button
                            onClick={handleGenerateAnalysis}
                            className="btn-primary text-sm flex items-center gap-2"
                        >
                            <span>✨</span> Generate Analysis
                        </button>
                    )}
                </div>

                {/* Loading State */}
                {loading && (
                    <div className="text-center py-8">
                        <div className="spinner mx-auto mb-4" />
                        <p className="text-gray-600 font-medium">Analyzing {playerName}&apos;s statistics...</p>
                        <p className="text-sm text-gray-400 mt-1">This may take a few seconds</p>
                    </div>
                )}

                {/* Error State */}
                {error && (
                    <div className={`border-2 rounded-lg p-4 ${isLimitError ? 'bg-amber-50 border-amber-300 text-amber-800' : 'bg-red-50 border-red-300 text-red-700'}`}>
                        <div className="flex items-start gap-3">
                            <span className="text-xl">{isLimitError ? '⏳' : '⚠️'}</span>
                            <div>
                                <p className="font-medium">{isLimitError ? 'Analysis Limit Reached' : 'Analysis Failed'}</p>
                                <p className="text-sm mt-1">{error}</p>
                            </div>
                        </div>
                        <button
                            onClick={handleGenerateAnalysis}
                            className="mt-3 text-sm underline hover:no-underline ml-8"
                        >
                            Try again
                        </button>
                    </div>
                )}

                {/* Analysis Results */}
                {analysis && !loading && (
                    <div className="animate-fadeIn">
                        {/* Summary */}
                        <div className="mb-6">
                            <p className="text-gray-700 text-lg leading-relaxed">{analysis.summary}</p>
                        </div>

                        {/* Playing Style */}
                        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-6">
                            <h3 className="font-bold text-primary mb-2 flex items-center gap-2">
                                <span>⚽</span> Playing Style
                            </h3>
                            <p className="text-gray-700">{analysis.playingStyle}</p>
                        </div>

                        {/* Strengths */}
                        {analysis.strengths.length > 0 && (
                            <div className="mb-6">
                                <h3 className="font-bold text-primary mb-3 flex items-center gap-2">
                                    <span>💪</span> Key Strengths
                                </h3>
                                <div className="space-y-3">
                                    {analysis.strengths.map((strength, i) => (
                                        <div key={i} className="bg-green-50 border-l-4 border-green-500 pl-4 py-3 pr-4 rounded-r-lg">
                                            <div className="font-bold text-green-800 mb-1">{strength.stat}</div>
                                            <p className="text-gray-700 text-sm">{strength.insight}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Areas for Context / Improvement */}
                        {analysis.improvements.length > 0 && (
                            <div className="mb-6">
                                <h3 className="font-bold text-primary mb-3 flex items-center gap-2">
                                    <span>📊</span> Contextual Notes
                                </h3>
                                <div className="space-y-3">
                                    {analysis.improvements.map((item, i) => (
                                        <div key={i} className="bg-amber-50 border-l-4 border-amber-500 pl-4 py-3 pr-4 rounded-r-lg">
                                            <div className="font-bold text-amber-800 mb-1">{item.stat}</div>
                                            <p className="text-gray-700 text-sm">{item.insight}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Player Comparison */}
                        {analysis.comparison && (
                            <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                                <h3 className="font-bold text-purple-800 mb-2 flex items-center gap-2">
                                    <span>🔄</span> Similar Profile
                                </h3>
                                <p className="text-gray-700">{analysis.comparison}</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Initial State - No analysis yet */}
                {!analysis && !loading && !error && (
                    <p className="text-gray-500 text-center py-4">
                        Click &quot;Generate Analysis&quot; to get AI-powered insights about this player&apos;s statistics,
                        including what each stat means, how it impacts the game, and their playing style.
                    </p>
                )}
            </div>
        </div>
    );
}
