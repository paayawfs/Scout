"""
Export player data to JSON for Supabase upload.
Run this after processing the data but before training the model.
This script takes the raw multiseason data and exports it for the web app.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "multiseason_data"

print("=" * 60)
print("PLAYER REPLACEMENT FINDER - DATA EXPORT")
print("=" * 60)

# ============================================================
# 1. LOAD AND MERGE DATA
# ============================================================
print("\n[1/6] Loading raw data...")

# Load all tables
# Load all tables
# Use header=1 for most files to get clean metric names (e.g. Gls instead of Per 90 Minutes)
standard = pd.read_csv(RAW_DIR / "merged_standard.csv", header=1).drop_duplicates()
passing = pd.read_csv(RAW_DIR / "merged_passing.csv", header=0).drop_duplicates() # Passing works with header 0
defense = pd.read_csv(RAW_DIR / "merged_defense.csv", header=1).drop_duplicates()
possession = pd.read_csv(RAW_DIR / "merged_possession.csv", header=1).drop_duplicates()
shooting = pd.read_csv(RAW_DIR / "merged_shooting.csv", header=1).drop_duplicates()
misc = pd.read_csv(RAW_DIR / "merged_misc.csv", header=1).drop_duplicates()

# Rename Unnamed columns in header=1 files to match join keys
renames = {
    'Unnamed: 0': 'league',
    'Unnamed: 1': 'season',
    'Unnamed: 2': 'team',
    'Unnamed: 3': 'player',
    'Unnamed: 4': 'nation',
    'Unnamed: 5': 'pos',
    'Unnamed: 6': 'age',
    'Unnamed: 7': 'born'
}

for d in [standard, defense, possession, shooting, misc]:
    d.rename(columns=renames, inplace=True)

print(f"  Standard: {len(standard)} rows")

# Merge tables
KEY_COLS = ['player', 'team', 'season', 'league']
df = standard.copy()

for name, table in [('passing', passing), ('defense', defense), 
                    ('possession', possession), ('shooting', shooting), ('misc', misc)]:
    existing = set(df.columns)
    new_cols = [c for c in table.columns if c not in existing or c in KEY_COLS]
    df = df.merge(table[new_cols], on=KEY_COLS, how='left', suffixes=('', f'_{name}'))

print(f"  Merged: {df.shape}")

# ============================================================
# 2. CLEAN DATA
# ============================================================
print("\n[2/6] Cleaning data...")

# Rename columns
df = df.rename(columns={
    'player': 'Player', 'team': 'Squad', 'pos': 'Pos',
    'age': 'Age', 'nation': 'Nation', 'league': 'League', 'season': 'Season'
})

# Clean age
def clean_age(age_str):
    if pd.isna(age_str):
        return None
    try:
        if isinstance(age_str, str) and '-' in age_str:
            return int(age_str.split('-')[0])
        return int(float(age_str))
    except:
        return None

df['Age'] = df['Age'].apply(clean_age)

# Filter by playing time (min 2 x 90s)
nineties_col = next((c for c in df.columns if '90s' in str(c).lower()), None)
if nineties_col:
    df = df[df[nineties_col] >= 2.0]
    print(f"  After 90s filter: {len(df)} rows")

# Remove goalkeepers
df = df[~df['Pos'].str.contains('GK', na=False)]
print(f"  After removing GKs: {len(df)} rows")

# ============================================================
# 3. AGGREGATE BY PLAYER
# ============================================================
print("\n[3/6] Aggregating by player...")

id_cols = ['Player', 'Squad', 'Pos', 'Nation', 'League', 'Age', 'born', 'Season']
numeric_cols = [c for c in df.columns if c not in id_cols and df[c].dtype in ['int64', 'float64']]

# Weighted average by 90s played
def weighted_agg(group):
    result = {
        'Squad': group['Squad'].iloc[-1],
        'Pos': group['Pos'].iloc[-1],
        'Nation': group['Nation'].iloc[-1] if 'Nation' in group.columns else None,
        'League': group['League'].iloc[-1],
        'Age': group['Age'].iloc[-1],
    }
    weights = group['90s'].values if '90s' in group.columns else np.ones(len(group))
    weights = np.maximum(weights, 0.1)
    
    for col in numeric_cols:
        if col in group.columns:
            vals = group[col].values
            mask = ~np.isnan(vals)
            if mask.any():
                result[col] = np.average(vals[mask], weights=weights[mask])
            else:
                result[col] = 0
    return pd.Series(result)

df_agg = df.groupby('Player').apply(weighted_agg).reset_index()
df_agg[numeric_cols] = df_agg[numeric_cols].fillna(0)

print(f"  Unique players: {len(df_agg)}")

# ============================================================
# 4. COMPUTE FEATURES & SIMILARITY
# ============================================================
print("\n[4/6] Computing similarity matrix...")

# Key features for radar chart - using correct column names from merged data
# Column names come from header=1 loading
RADAR_STATS = {
    # --- OFFENSIVE ---
    'Gls': 'Goals',
    'Ast': 'Assists',
    'npxG': 'Non-Penalty xG',
    'xAG': 'xAG',
    'Sh': 'Shots',
    'SoT': 'Shots on Target',
    'G/Sh': 'Goals per Shot',
    'Dist': 'Avg Shot Distance',
    'PK': 'Penalty Kicks Made',
    'PKatt': 'Penalty Kicks Attempted',

    # --- PASSING ---
    'PrgP': 'Progressive Passes',  # Specific Request
    'KP': 'Key Passes',
    '1/3': 'Passes into Final 1/3',
    'PPA': 'Passes into Penalty Area',
    'CrsPA': 'Crosses into Penalty Area',
    'Att': 'Passes Attempted',  # Total
    'Cmp': 'Passes Completed',  # Total
    # Note: Percentage columns might be tricky if not pre-calculated in df_agg correctly (weighted avg of %). 
    # We'll rely on sums of Cmp/Att if needed, but for now let's see if we can use the raw columns if they exist.
    # Often 'Total' is the column name for Total Cmp/Att in passing.csv header=0. 
    # Let's trust the merge suffixes. checking headers: passing has 'Total' for Cmp? No, usually Cmp, Att.
    # header=0 for passing often gives: 'Cmp', 'Att', 'Cmp%', 'TotDist', 'PrgDist', etc.
    # Let's use specific known columns from our debug list: 'PrgP', 'PrgP_passing' (check which one is populated).
    # We will try standard names first.
    
    # --- POSSESSION ---
    'PrgR': 'Prog Passes Received', # Specific Request
    'PrgC': 'Progressive Carries',
    'Succ': 'Successful Dribbles',
    'Att_possession': 'Dribbles Attempted', # Check actual col name
    'Touches': 'Touches',
    'Dis': 'Dispossessed',
    'Mis': 'Miscontrols',
    'Recov': 'Ball Recoveries',

    # --- DEFENSIVE ---
    'Tkl': 'Tackles',
    'TklW': 'Tackles Won',
    'Int': 'Interceptions',
    'Blocks': 'Blocks',
    'Clr': 'Clearances',
    'Err': 'Errors Leading to Shot',
    
    # --- AERIAL / MISC ---
    'Won': 'Aerials Won',
    'Lost': 'Aerials Lost',
    #'Won%': 'Aerial Win %', # Might need weighted avg care
    'Fls': 'Fouls Committed',
    'Fld': 'Fouls Drawn',
    'Off': 'Offsides',
    'CrdY': 'Yellow Cards',
    'CrdR': 'Red Cards',
}

RADAR_FEATURES = [f for f in RADAR_STATS.keys() if f in df_agg.columns]
RADAR_LABELS = [RADAR_STATS[f] for f in RADAR_FEATURES]

# All numeric features for similarity
FEATURES = [c for c in numeric_cols if c in df_agg.columns]
X = df_agg[FEATURES].fillna(0).values

# Scale and compute similarity
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
similarity = cosine_similarity(X_scaled)
similarity = (similarity + 1) / 2  # Normalize to 0-1

print(f"  Features: {len(FEATURES)}")
print(f"  Radar features: {len(RADAR_FEATURES)}")

# ============================================================
# 5. EXPORT TO JSON
# ============================================================
print("\n[5/6] Exporting to JSON...")

# Players
players = []
for i, row in df_agg.iterrows():
    players.append({
        'id': int(i),
        'name': str(row['Player']),
        'squad': str(row['Squad']) if pd.notna(row['Squad']) else None,
        'position': str(row['Pos']) if pd.notna(row['Pos']) else None,
        'age': int(row['Age']) if pd.notna(row['Age']) else None,
        'nation': str(row['Nation']) if pd.notna(row['Nation']) else None,
        'league': str(row['League']) if pd.notna(row['League']) else None,
    })

# Stats (radar features only, per-90)
nineties = df_agg['90s'].replace(0, np.nan) if '90s' in df_agg.columns else None
stats = []
for i, row in df_agg.iterrows():
    for j, feat in enumerate(RADAR_FEATURES):
        # Convert to per-90 (All stats are Totals now so we divide)
        val = row[feat]
        if nineties is not None:
            n90 = nineties.iloc[i] if pd.notna(nineties.iloc[i]) else 1
            val = val / max(n90, 0.1) if not pd.isna(val) else 0
        
        # Get min/max for normalization
        all_vals = df_agg[feat].dropna()
        min_val = float(np.percentile(all_vals, 5))
        max_val = float(np.percentile(all_vals, 95))
        
        stats.append({
            'player_id': int(i),
            'stat_name': RADAR_LABELS[j],
            'stat_key': feat,
            'value': round(float(val) if pd.notna(val) else 0, 3),
            'min_range': round(min_val, 3),
            'max_range': round(max_val, 3)
        })

# Similarities (top 50 per player)
similarities = []
for i in range(len(df_agg)):
    sim_scores = similarity[i]
    # Get top 50 (excluding self)
    top_indices = np.argsort(sim_scores)[-51:-1][::-1]
    for rank, j in enumerate(top_indices, 1):
        if i != j:
            similarities.append({
                'player_id': int(i),
                'similar_player_id': int(j),
                'similarity': round(float(sim_scores[j]), 4),
                'rank': rank
            })

# Save
with open(DATA_DIR / 'export_players.json', 'w', encoding='utf-8') as f:
    json.dump(players, f, indent=2)
    
with open(DATA_DIR / 'export_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f)
    
with open(DATA_DIR / 'export_similarities.json', 'w', encoding='utf-8') as f:
    json.dump(similarities, f)

print(f"  Players: {len(players)}")
print(f"  Stats: {len(stats)}")
print(f"  Similarities: {len(similarities)}")

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("EXPORT COMPLETE!")
print("=" * 60)
print(f"\nFiles created in {DATA_DIR}/:")
print(f"  • export_players.json     ({len(players)} players)")
print(f"  • export_stats.json       ({len(stats)} stat records)")
print(f"  • export_similarities.json ({len(similarities)} records)")
print(f"\nPosition breakdown:")
for pos, count in df_agg['Pos'].value_counts().head(10).items():
    print(f"  {pos:10s}: {count:4d}")
print(f"\nTop nations:")
for nation, count in df_agg['Nation'].value_counts().head(10).items():
    print(f"  {nation:10s}: {count:4d}")
print("\nUpload these files to Supabase to complete setup!")
