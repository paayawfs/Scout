import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { createClient } from '@supabase/supabase-js';

// Force dynamic rendering - no caching
export const dynamic = 'force-dynamic';
export const revalidate = 0;

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

// Create a fresh Supabase client for this API route (no caching)
const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL!;
const supabaseKey = process.env.SUPABASE_SERVICE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!;

function getSupabase() {
  return createClient(supabaseUrl, supabaseKey, {
    global: {
      fetch: (url, options = {}) => {
        return fetch(url, {
          ...options,
          cache: 'no-store',
        });
      },
    },
  });
}

interface PlayerStats {
  id: number;
  name: string;
  position: string;
  squad: string;
  age: number;
  stats: { name: string; value: number; percentile: number }[];
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { player } = body as { player: PlayerStats };

    if (!player || !player.id) {
      return NextResponse.json(
        { error: 'Player ID is required', receivedPlayer: player },
        { status: 400 }
      );
    }

    // Create fresh Supabase client for each request
    const supabase = getSupabase();

    console.log('Step 1: Checking cache for player', player.id);

    // Check if we have cached insights for this player
    const { data: cached, error: cacheError } = await supabase
      .from('player_insights')
      .select('*')
      .eq('player_id', player.id)
      .single();

    console.log('Step 2: Cache result:', { cached: !!cached, cacheError: cacheError?.message });

    if (cached && !cacheError) {
      // Return cached insights
      console.log('Step 3: Returning cached insights');
      return NextResponse.json({
        analysis: {
          summary: cached.summary,
          strengths: cached.strengths,
          improvements: cached.improvements,
          playingStyle: cached.playing_style,
        },
        cached: true,
      });
    }

    // No cache - generate with Gemini
    if (!process.env.GEMINI_API_KEY) {
      return NextResponse.json(
        { error: 'Gemini API key not configured', details: 'GEMINI_API_KEY environment variable is not set' },
        { status: 500 }
      );
    }

    // FETCH FRESH STATS FROM DATABASE - don't trust frontend data
    console.log('Step 3: Fetching fresh stats from database for player', player.id);
    const { data: freshStats, error: statsError } = await supabase
      .from('player_stats')
      .select('stat_name, value, percentile')
      .eq('player_id', player.id);

    if (statsError || !freshStats || freshStats.length === 0) {
      console.error('Failed to fetch fresh stats:', statsError);
      return NextResponse.json(
        { error: 'Failed to fetch player stats', details: statsError?.message },
        { status: 500 }
      );
    }

    console.log('Step 4: Fresh stats from DB:', freshStats.slice(0, 3), '...');

    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    // Build stats summary with FRESH percentiles from database
    const statsSummary = freshStats
      .filter(s => s.stat_name !== 'Goals per Shot' && s.stat_name !== 'G/Sh')
      .map(s => `- ${s.stat_name}: ${s.value.toFixed(2)} per 90 (${s.percentile}th percentile)`)
      .join('\n');

    const prompt = `You are an expert football analyst writing for a scouting report. Analyze this player's statistics.

PLAYER: ${player.name}
POSITION: ${player.position}
CLUB: ${player.squad}
AGE: ${player.age}

STATISTICS (per 90 minutes, with percentile ranking across all players in top European leagues):
${statsSummary}

The percentile shows where this player ranks compared to all outfield players. 90+ is elite, 70-89 is above average, 30-69 is average, below 30 is below average.

Respond with JSON only:
{
  "summary": "2-3 sentence overview of player profile",
  "strengths": [
    {
      "stat": "Stat Name",
      "insight": "Explain what this stat and percentile tells us. Mention the actual per-90 value and percentile. Discuss tactical implications."
    }
  ],
  "improvements": [
    {
      "stat": "Stat Name",
      "insight": "What this lower percentile suggests. Could be tactical choice, role-based, or area to develop."
    }
  ],
  "playingStyle": "One paragraph on how this player likely plays based on the numbers."
}

Include 3 strengths (highest percentiles) and 2-4 contextual notes (lower percentiles). Write in plain text, no markdown formatting.`;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    // Extract JSON from response
    const jsonMatch = text.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error('Failed to parse AI response');
    }

    const analysis = JSON.parse(jsonMatch[0]);

    // Store in database for future requests
    const { error: insertError } = await supabase
      .from('player_insights')
      .insert({
        player_id: player.id,
        summary: analysis.summary,
        strengths: analysis.strengths,
        improvements: analysis.improvements,
        playing_style: analysis.playingStyle,
      });

    if (insertError) {
      console.error('Failed to cache insights:', insertError);
    }

    return NextResponse.json({ analysis, cached: false });
  } catch (error: any) {
    console.error('Insights API error:', error);

    // Handle Quota/Rate Limit Errors specifically
    const msg = error.message?.toLowerCase() || '';
    if (msg.includes('429') || msg.includes('quota') || msg.includes('exhausted') || msg.includes('limit')) {
      return NextResponse.json(
        {
          error: 'AI_LIMIT_REACHED',
          details: 'Daily AI analysis limit reached. Please try again later or verify subscription.',
          friendlyMessage: "Our AI scout is taking a quick break! We've hit our analysis limit for the moment. Please check back later."
        },
        { status: 429 }
      );
    }

    const errorMessage = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json(
      { error: 'Failed to generate analysis', details: errorMessage },
      { status: 500 }
    );
  }
}
