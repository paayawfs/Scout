import { Player } from "@/lib/supabase";
import Link from "next/link";

interface PlayerCardProps {
    player: Player;
    similarity?: number;
    showLink?: boolean;
}

export default function PlayerCard({ player, similarity, showLink = true, compareLink }: PlayerCardProps & { compareLink?: string }) {
    return (
        <div className="card-elevated group relative flex flex-col h-full">
            {/* Similarity Badge - Top right */}
            {similarity !== undefined && (
                <div className="similarity-badge absolute top-4 right-4 z-10">
                    {(similarity * 100).toFixed(1)}% Match
                </div>
            )}

            <div className="pr-20 mb-auto">
                {showLink ? (
                    <Link href={`/player/${player.id}`} className="hover:underline decoration-2 underline-offset-4 decoration-black">
                        <h3 className="text-lg sm:text-xl text-primary font-bold line-clamp-1 uppercase">{player.name}</h3>
                    </Link>
                ) : (
                    <h3 className="text-lg sm:text-xl text-primary font-bold line-clamp-1 uppercase">{player.name}</h3>
                )}

                <div className="divider-accent mt-2 mb-3" />
                <p className="text-gray-600 line-clamp-1 font-mono text-sm">{player.squad}</p>

                {/* Player Details */}
                <div className="flex flex-wrap gap-2 mt-3 text-sm text-gray-500 font-mono">
                    <span className="bg-gray-100 px-2 py-0.5 rounded text-xs sm:text-sm border border-gray-200">{player.position}</span>
                    {player.age && (
                        <span className="bg-gray-100 px-2 py-0.5 rounded text-xs sm:text-sm border border-gray-200">Age {player.age}</span>
                    )}
                    {player.nation && (
                        <span className="bg-blue-50 text-accent px-2 py-0.5 rounded text-xs sm:text-sm uppercase tracking-wide border border-blue-200 font-bold">
                            {player.nation}
                        </span>
                    )}
                </div>
            </div>

            {/* Footer Actions */}
            <div className="mt-4 pt-4 border-t-2 border-gray-100 flex items-center justify-between gap-4">
                {showLink && (
                    <Link href={`/player/${player.id}`} className="text-sm text-gray-500 font-bold hover:text-accent transition-colors flex items-center gap-1 uppercase">
                        Profile
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                        </svg>
                    </Link>
                )}

                {compareLink && (
                    <Link
                        href={compareLink}
                        className="btn-primary text-xs !py-1.5 !px-3 shadow-[2px_2px_0px_0px_rgba(0,0,0,1)] hover:shadow-[3px_3px_0px_0px_rgba(0,0,0,1)] hover:translate-x-[-1px] hover:translate-y-[-1px] transition-all whitespace-nowrap ml-auto"
                    >
                        Compare
                    </Link>
                )}
            </div>
        </div>
    );
}
