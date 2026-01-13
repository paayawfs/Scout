# Player Replacement Finder - Web App

A Next.js web application for finding statistically similar football players using AI-powered analysis.

## Tech Stack

- **Frontend**: Next.js 14 (App Router)
- **Database**: Supabase (PostgreSQL)
- **Charts**: Recharts
- **Styling**: Tailwind CSS
- **Deployment**: Vercel

## Setup

### 1. Install Dependencies

```bash
cd web
npm install
```

### 2. Configure Supabase

1. Create a new project at [supabase.com](https://supabase.com)
2. Run the SQL from `SUPABASE_SCHEMA.md` in the SQL Editor
3. Copy your project URL and anon key
4. Create `.env.local`:

```env
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your-anon-key
```

### 3. Import Data

1. Run the Jupyter notebook `05b_siamese_multiseason.ipynb`
2. Uncomment and run `export_for_supabase()` at the end
3. Import the JSON files to Supabase:
   - `data/export_players.json` → `players` table
   - `data/export_stats.json` → `player_stats` table
   - `data/export_similarities.json` → `player_similarity` table

### 4. Run Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Deploy to Vercel

1. Push your code to GitHub
2. Connect to Vercel
3. Add environment variables in Vercel dashboard
4. Deploy!

## Features

- **Search**: Find any player from Europe's top leagues
- **Filters**: Filter by position, nationality, and maximum age
- **Radar Charts**: Visual comparison of key statistics
- **Similarity Scores**: AI-powered similarity percentages
- **Detailed Stats**: Side-by-side statistical comparison

## Color Scheme

- **Navy**: #1E3A5F (primary)
- **Gold**: #D4AF37 (accent)
- **White**: #FFFFFF (background)

## Project Structure

```
web/
├── app/
│   ├── page.tsx                    # Home page
│   ├── layout.tsx                  # Root layout
│   ├── globals.css                 # Global styles
│   ├── player/[id]/page.tsx        # Player detail page
│   └── compare/[id1]/[id2]/page.tsx # Comparison page
├── components/
│   ├── SearchBar.tsx               # Player search
│   ├── PlayerCard.tsx              # Player info card
│   ├── FilterBar.tsx               # Filters (position, nation, age)
│   └── RadarChart.tsx              # Radar comparison chart
└── lib/
    └── supabase.ts                 # Supabase client & types
```
