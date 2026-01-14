import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

export const supabase = createClient(supabaseUrl, supabaseAnonKey);

// Types for our data
export interface Player {
    id: number;
    name: string;
    squad: string;
    position: string | null;
    age: number | null;
    nation: string | null;
    league: string | null;
}

export interface PlayerStat {
    player_id: number;
    stat_name: string;
    stat_key: string;
    value: number;
    min_range: number;
    max_range: number;
}

export interface PlayerSimilarity {
    player_id: number;
    similar_player_id: number;
    similarity: number;
    rank: number;
}

export interface PlayerInsight {
    id?: number;
    player_id: number;
    summary: string;
    strengths: { stat: string; insight: string }[];
    improvements: { stat: string; insight: string }[];
    playing_style: string;
    comparison: string;
    created_at?: string;
}
