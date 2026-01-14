import { NextRequest, NextResponse } from 'next/server';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { supabase } from '@/lib/supabase';

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '');

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
    console.log('Received body:', JSON.stringify(body, null, 2));

    const { player } = body as { player: PlayerStats };

    if (!player || !player.id) {
      return NextResponse.json(
        { error: 'Player ID is required', receivedPlayer: player },
        { status: 400 }
      );
    }

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
          comparison: cached.comparison,
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

    const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

    // Build stats summary for the prompt
    const statsSummary = player.stats
      .map(s => `- ${s.name}: ${s.value.toFixed(2)} per 90 (${s.percentile}th percentile)`)
      .join('\n');

    const prompt = `You are an expert football analyst. Analyze this player's statistics and provide insights.

PLAYER: ${player.name}
POSITION: ${player.position}
CLUB: ${player.squad}
AGE: ${player.age}

STATISTICS (per 90 minutes, with league percentile):
${statsSummary}

Provide a detailed analysis in JSON format with these sections:
{
  "summary": "2-3 sentence overview of player profile and role",
  "strengths": [
    {
      "stat": "Stat Name",
      "insight": "What this high number means for gameplay. Explain the stat, how it impacts the team, and what playing style leads to this strength. Be specific about tactical implications."
    }
  ],
  "improvements": [
    {
      "stat": "Stat Name", 
      "insight": "What this lower number indicates. Explain whether this is due to playing style, role, or an area to develop. Not every low stat is a weakness - context matters."
    }
  ],
  "playingStyle": "One paragraph describing how this player likely plays based on their statistical profile. What role do they fulfill? How do they contribute to the team's attack/defense?",
  "comparison": "What type of player are they similar to? Mention 1-2 similar high-profile players if applicable."
}

Focus on the top 3 strengths (stats above 70th percentile) and mention 2 to 4 areas for context (lower percentiles). Be analytical but accessible - explain stats for fans who may not understand advanced metrics.

IMPORTANT: Do NOT use markdown formatting (like **bold** or *italics*) within the JSON response fields. Provide plain text only.`;

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
        comparison: analysis.comparison,
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
