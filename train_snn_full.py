"""
Full-Feature SNN Training and Inference Pipeline
================================================
This script:
1. Loads ALL raw multiseason data
2. Merges and processes ALL features (not just 12!)
3. Trains a Siamese Neural Network on ALL features
4. Computes similarity using the trained model
5. Exports players, stats, and similarities to JSON

Run: python train_snn_full.py
"""

import os
import json
import pandas as pd
import numpy as np
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
N_EPOCHS = 100
PATIENCE = 10
EMB_DIM = 128  # Higher dimension for more features
VAL_SPLIT = 0.2

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "multiseason_data"
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 1. LOAD AND MERGE ALL DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: LOADING ALL RAW DATA")
print("=" * 60)

# Load all tables with proper headers
# Use header=1 for most files
standard = pd.read_csv(RAW_DIR / "merged_standard.csv", header=1).drop_duplicates()
defense = pd.read_csv(RAW_DIR / "merged_defense.csv", header=1).drop_duplicates()
# Rename colliding columns in Defense
defense.rename(columns={'Sh': 'Sh_Blocked', 'Att': 'Tkl_Att'}, inplace=True)

possession = pd.read_csv(RAW_DIR / "merged_possession.csv", header=1).drop_duplicates()
# Rename colliding columns in Possession
possession.rename(columns={'Att': 'Drib_Att'}, inplace=True)

shooting = pd.read_csv(RAW_DIR / "merged_shooting.csv", header=1).drop_duplicates()
misc = pd.read_csv(RAW_DIR / "merged_misc.csv", header=1).drop_duplicates()

# PASSING DATA SPECIAL HANDLING: Mixed headers (some in row 0, some in row 1)
# 1. Read with header=1 to get Cmp, Att, etc.
passing = pd.read_csv(RAW_DIR / "merged_passing.csv", header=1).drop_duplicates()
# 2. Read header=0 to get KP, PPA, etc (which are Unnamed in header=1)
passing_h0 = pd.read_csv(RAW_DIR / "merged_passing.csv", header=0, nrows=0)
h0_cols = passing_h0.columns.tolist()
# 3. Patch Unnamed columns
new_cols = []
for i, col in enumerate(passing.columns):
    if "Unnamed" in col and i < len(h0_cols):
        # Use header from row 0 if row 1 is unnamed
        new_cols.append(h0_cols[i])
    else:
        new_cols.append(col)
passing.columns = new_cols

renames = {
    'Unnamed: 0': 'league', 'Unnamed: 1': 'season', 'Unnamed: 2': 'team',
    'Unnamed: 3': 'player', 'Unnamed: 4': 'nation', 'Unnamed: 5': 'pos',
    'Unnamed: 6': 'age', 'Unnamed: 7': 'born', 'Unnamed: 8': '90s_dup'
}

# Apply renames
for d in [standard, passing, defense, possession, shooting, misc]:
    if isinstance(d, pd.DataFrame):
        d.rename(columns=renames, inplace=True)

print(f"  Standard: {len(standard)} rows, {len(standard.columns)} cols")
print(f"  Passing:  {len(passing)} rows, {len(passing.columns)} cols")
print(f"  Defense:  {len(defense)} rows, {len(defense.columns)} cols")
print(f"  Possession: {len(possession)} rows, {len(possession.columns)} cols")
print(f"  Shooting: {len(shooting)} rows, {len(shooting.columns)} cols")
print(f"  Misc:     {len(misc)} rows, {len(misc.columns)} cols")

# Merge all tables
KEY_COLS = ['player', 'team', 'season', 'league']
df = standard.copy()

for name, table in [('passing', passing), ('defense', defense), 
                    ('possession', possession), ('shooting', shooting), ('misc', misc)]:
    existing = set(df.columns)
    new_cols = [c for c in table.columns if c not in existing or c in KEY_COLS]
    # Ensure keys are present in both
    if not all(k in table.columns for k in KEY_COLS):
        print(f"Warning: {name} table missing key columns!")
        continue
    df = df.merge(table[new_cols], on=KEY_COLS, how='left', suffixes=('', f'_{name}'))

print(f"\n  Merged: {df.shape[0]} rows, {df.shape[1]} columns")

# ============================================================
# 2. CLEAN DATA
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: CLEANING DATA")
print("=" * 60)

# Rename to standard format
df = df.rename(columns={
    'player': 'Player', 'team': 'Squad', 'pos': 'Pos',
    'age': 'Age', 'nation': 'Nation', 'league': 'League', 'season': 'Season'
})

# Clean age
def clean_age(age_str):
    if pd.isna(age_str): return None
    try:
        if isinstance(age_str, str) and '-' in age_str:
            return int(age_str.split('-')[0])
        return int(float(age_str))
    except: return None

df['Age'] = df['Age'].apply(clean_age)

# Filter by playing time
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
print("\n" + "=" * 60)
print("STEP 3: AGGREGATING BY PLAYER")
print("=" * 60)

id_cols = ['Player', 'Squad', 'Pos', 'Nation', 'League', 'Age', 'born', 'Season']
numeric_cols = [c for c in df.columns if c not in id_cols and df[c].dtype in ['int64', 'float64']]
print(f"  Found {len(numeric_cols)} numeric columns")

def weighted_agg(group):
    result = {
        'Squad': group['Squad'].iloc[-1],
        'Pos': group['Pos'].iloc[-1],
        'Nation': group['Nation'].iloc[-1] if 'Nation' in group.columns else None,
        'League': group['League'].iloc[-1],
        'Age': group['Age'].iloc[-1],
    }
    # Be more robust looking for the 90s column (could be '90s' or '90s_standard' etc)
    n90_col = next((c for c in group.columns if '90s' in str(c).lower()), '90s')
    weights = group[n90_col].values if n90_col in group.columns else np.ones(len(group))
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
# 4. PREPARE FEATURES FOR SNN
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: PREPARING FEATURES")
print("=" * 60)

# Use ALL numeric features for similarity
FEATURES = [c for c in numeric_cols if c in df_agg.columns and not c.startswith('Unnamed')]
# Remove percentage columns (keep raw counts for more meaningful comparison)
FEATURES = [f for f in FEATURES if '%' not in f and 'Unnamed' not in f]
print(f"  Using {len(FEATURES)} features for SNN training")

# Scale features
X = df_agg[FEATURES].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"  Feature matrix shape: {X_scaled.shape}")

# Compute ground truth similarity (cosine)
print("  Computing ground truth similarity matrix...")
ground_truth = cosine_similarity(X_scaled)
ground_truth = (ground_truth + 1) / 2  # Normalize to 0-1

# ============================================================
# 5. SIAMESE NETWORK
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: DEFINING SIAMESE NETWORK")
print("=" * 60)

class PairDataset(Dataset):
    def __init__(self, features, sim_matrix, indices, n_pairs=10000, seed=SEED):
        self.features = torch.FloatTensor(features)
        self.sim = sim_matrix
        self.indices = indices
        self.n_pairs = n_pairs
        self.rng = np.random.RandomState(seed)
        
    def __len__(self): return self.n_pairs
    
    def __getitem__(self, _):
        i, j = self.rng.choice(self.indices, 2, replace=False)
        return self.features[i], self.features[j], torch.FloatTensor([self.sim[i, j]])


class SiameseNet(nn.Module):
    def __init__(self, in_dim, emb_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, emb_dim)
        )
        self.fc = nn.Sequential(
            nn.Linear(emb_dim * 3, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        e1, e2 = self.enc(x1), self.enc(x2)
        combined = torch.cat([torch.abs(e1 - e2), e1 * e2, (e1 + e2) / 2], dim=1)
        return self.fc(combined), e1, e2
    
    def embed(self, x):
        self.eval()
        with torch.no_grad():
            return self.enc(x).cpu().numpy()


# Create datasets
all_indices = np.arange(len(X_scaled))
train_indices, val_indices = train_test_split(all_indices, test_size=VAL_SPLIT, random_state=SEED)

train_ds = PairDataset(X_scaled, ground_truth, train_indices, n_pairs=20000)
val_ds = PairDataset(X_scaled, ground_truth, val_indices, n_pairs=5000, seed=SEED+1)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

print(f"  Train pairs: {len(train_ds)}")
print(f"  Val pairs: {len(val_ds)}")

# ============================================================
# 6. TRAIN MODEL
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: TRAINING SIAMESE NETWORK")
print("=" * 60)

model = SiameseNet(len(FEATURES), EMB_DIM).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
criterion = nn.MSELoss()

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(N_EPOCHS):
    # Train
    model.train()
    train_loss = 0
    for x1, x2, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS}", leave=False):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad()
        pred, _, _ = model(x1, x2)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            pred, _, _ = model(x1, x2)
            val_loss += criterion(pred, y).item()
    val_loss /= len(val_loader)
    
    scheduler.step(val_loss)
    
    print(f"  Epoch {epoch+1:2d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODELS_DIR / 'siamese_full_best.pth')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"  Early stopping at epoch {epoch+1}")
            break

# Load best model
model.load_state_dict(torch.load(MODELS_DIR / 'siamese_full_best.pth', map_location=device, weights_only=True))
print(f"\n  Best validation loss: {best_val_loss:.4f}")

# ============================================================
# 7. COMPUTE EMBEDDINGS AND SIMILARITY
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: COMPUTING SIMILARITY")
print("=" * 60)

embeddings = model.embed(torch.FloatTensor(X_scaled).to(device))
print(f"  Embeddings shape: {embeddings.shape}")

# STRICT Gaussian RBF similarity - exponential decay with distance
# This properly penalizes differences unlike linear normalization
distances = euclidean_distances(embeddings)

# Use median distance as sigma for Gaussian kernel - this is key for proper scaling
sigma = np.median(distances[distances > 0])  
# Gaussian RBF: sim = exp(-d^2 / (2 * sigma^2))
# This means similarity drops off exponentially - only truly similar players score high
similarity = np.exp(-(distances ** 2) / (2 * sigma ** 2))
np.fill_diagonal(similarity, 1.0)

print(f"  Similarity matrix: {similarity.shape}")
print(f"  Sigma (median dist): {sigma:.4f}")
print(f"  Min: {similarity.min():.4f}, Max: {similarity.max():.4f}, Mean: {similarity.mean():.4f}")

# Show distribution
percentiles = [25, 50, 75, 90, 95, 99]
print(f"  Percentiles: {dict(zip(percentiles, [np.percentile(similarity, p) for p in percentiles]))}")

# ============================================================
# 8. EXPORT TO JSON
# ============================================================
print("\n" + "=" * 60)
print("STEP 8: EXPORTING TO JSON")
print("=" * 60)

# Define ALL stats for radar chart
RADAR_STATS = {
    # --- OFFENSIVE ---
    'Gls': 'Goals', 'Ast': 'Assists', 'npxG': 'Non-Penalty xG', 'xAG': 'xAG',
    'Sh': 'Shots', 'SoT': 'Shots on Target', 'G/Sh': 'Goals per Shot', 'G/Sh': 'Goals per Shot',
    'Dist': 'Avg Shot Distance', 'PK': 'Penalty Kicks', 'PKatt': 'PKs Attempted',
    # --- PASSING ---
    'PrgP': 'Progressive Passes', 'KP': 'Key Passes', '1/3': 'Final 1/3 Passes',
    'PPA': 'Penalty Area Passes', 'CrsPA': 'Crosses into PA',
    # --- PASSING (Total/Comp) Added ---
    'Att': 'Passes Attempted', 'Cmp': 'Passes Completed',
    # --- POSSESSION ---
    'PrgR': 'Progressive Receives', 'PrgC': 'Progressive Carries',
    'Succ': 'Successful Dribbles', 'Touches': 'Touches', 'Dis': 'Dispossessed',
    'Mis': 'Miscontrols', 'Carries': 'Carries', 'TotDist': 'Carry Distance',
    # --- DEFENSIVE ---
    'Tkl': 'Tackles', 'TklW': 'Tackles Won', 'Int': 'Interceptions',
    'Blocks': 'Blocks', 'Clr': 'Clearances', 'Err': 'Errors',
    # --- MISC ---
    'Fls': 'Fouls', 'Fld': 'Fouls Drawn', 'Off': 'Offsides',
    'CrdY': 'Yellow Cards', 'CrdR': 'Red Cards',
}

RADAR_FEATURES = [f for f in RADAR_STATS.keys() if f in df_agg.columns]
RADAR_LABELS = [RADAR_STATS[f] for f in RADAR_FEATURES]
missing_radar = [f for f in RADAR_STATS.keys() if f not in df_agg.columns]
if missing_radar:
    print(f"Warning: Missing RADAR columns: {missing_radar}")
print(f"  Radar stats available: {len(RADAR_FEATURES)}")

# Per-90 conversion safe check
nineties_col = next((c for c in df_agg.columns if '90s' in str(c).lower()), None)
nineties = df_agg[nineties_col].replace(0, np.nan) if nineties_col else None

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

# Stats (all radar features, per-90)
stats = []
for i, row in df_agg.iterrows():
    # Calculate division factor for this player
    if nineties is not None:
        n90 = nineties.iloc[i] if pd.notna(nineties.iloc[i]) else 1.0
        n90 = max(n90, 0.1) # Avoid division by zero
    else:
        n90 = 1.0

    for j, feat in enumerate(RADAR_FEATURES):
        raw_val = row[feat]
        if pd.isna(raw_val):
            val = 0.0
        else:
            val = raw_val / n90

        # Calculate min/max range based on PER-90 values, not raw totals
        # This fixes the percentile bug
        all_raw = df_agg[feat].dropna()
        all_n90 = nineties.dropna() if nineties is not None else pd.Series(np.ones(len(all_raw)), index=all_raw.index)
        
        # Align indices
        common_idx = all_raw.index.intersection(all_n90.index)
        all_raw = all_raw.loc[common_idx]
        all_n90 = all_n90.loc[common_idx].clip(lower=0.1)
        
        all_per90 = all_raw / all_n90
        
        min_val = float(np.percentile(all_per90, 5))
        max_val = float(np.percentile(all_per90, 95))
        
        stats.append({
            'player_id': int(i),
            'stat_name': RADAR_LABELS[j],
            'stat_key': feat,
            'value': round(float(val), 3),
            'min_range': round(min_val, 3),
            'max_range': round(max_val, 3)
        })

# Similarities (top 50 per player)
similarities = []
for i in range(len(df_agg)):
    sim_scores = similarity[i]
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

print(f"\n  Players: {len(players)}")
print(f"  Stats: {len(stats)} ({len(RADAR_FEATURES)} stats per player)")
print(f"  Similarities: {len(similarities)}")

print("\n" + "=" * 60)
print("COMPLETE! Files saved to data/")
print("=" * 60)
print("\nNext: Run 'python upload_to_supabase.py' to upload to database")
