# Player Replacement Finder - Supabase Schema

Run these SQL commands in your Supabase SQL Editor to set up the database.

## Tables

```sql
-- Players table
CREATE TABLE players (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  squad TEXT,
  position TEXT,
  age INTEGER,
  nation TEXT,
  league TEXT
);

-- Create index for search
CREATE INDEX idx_players_name ON players USING gin(to_tsvector('english', name));
CREATE INDEX idx_players_nation ON players(nation);
CREATE INDEX idx_players_position ON players(position);

-- Player statistics
CREATE TABLE player_stats (
  id SERIAL PRIMARY KEY,
  player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
  stat_name TEXT NOT NULL,
  stat_key TEXT NOT NULL,
  value FLOAT NOT NULL DEFAULT 0,
  min_range FLOAT DEFAULT 0,
  max_range FLOAT DEFAULT 1
);

CREATE INDEX idx_player_stats_player ON player_stats(player_id);

-- Precomputed similarities (top 50 per player)
CREATE TABLE player_similarity (
  id SERIAL PRIMARY KEY,
  player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
  similar_player_id INTEGER REFERENCES players(id) ON DELETE CASCADE,
  similarity FLOAT NOT NULL,
  rank INTEGER NOT NULL,
  UNIQUE(player_id, similar_player_id)
);

CREATE INDEX idx_similarity_player ON player_similarity(player_id);
CREATE INDEX idx_similarity_rank ON player_similarity(player_id, rank);
```

## Row Level Security (RLS)

```sql
-- Enable RLS
ALTER TABLE players ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_similarity ENABLE ROW LEVEL SECURITY;

-- Allow public read access
CREATE POLICY "Public read access" ON players FOR SELECT USING (true);
CREATE POLICY "Public read access" ON player_stats FOR SELECT USING (true);
CREATE POLICY "Public read access" ON player_similarity FOR SELECT USING (true);
```

## Import Data

After running the export_for_supabase() function in the Jupyter notebook, you'll have:
- `export_players.json`
- `export_stats.json`
- `export_similarities.json`

Import these using Supabase's CSV/JSON import feature or via the API.
