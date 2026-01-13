"""
SNN-Based Player Similarity Inference

This script:
1. Loads the trained Siamese Neural Network model
2. Computes player embeddings
3. Calculates Euclidean distance-based similarity (stricter than cosine)
4. Exports results to JSON for Supabase upload

Run: python run_snn_inference.py
Requires: pip install torch pandas numpy scikit-learn python-dotenv
"""

import os
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

# ============================================================
# CONFIGURATION
# ============================================================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_DIR = Path("data")
MODELS_DIR = Path("models")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# SIAMESE NETWORK ARCHITECTURE (must match trained model)
# ============================================================
class SiameseNet(nn.Module):
    def __init__(self, in_dim, emb_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, emb_dim)
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


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    """Find column by possible names."""
    for name in names:
        for col in df.columns:
            if str(col).lower() == name.lower():
                return col
    return None


def load_features(filepath: Path) -> Tuple[List[str], Dict[str, str]]:
    """Load feature names from file."""
    features = []
    feature_names = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '#' in line:
                parts = line.split('#', 1)
                raw_name = parts[0].strip()
                readable_name = parts[1].strip()
                if raw_name and readable_name:
                    features.append(raw_name)
                    feature_names[raw_name] = readable_name
    return features, feature_names


def compute_euclidean_similarity(embeddings: np.ndarray) -> np.ndarray:
    """
    Convert Euclidean distances to similarity scores (0-1).
    
    Unlike cosine similarity which only measures direction,
    Euclidean distance measures actual distance in feature space.
    This means 0.5 goals/90 vs 0.8 goals/90 will show LOWER similarity.
    """
    distances = euclidean_distances(embeddings)
    
    # Normalize to 0-1 (closer = higher similarity)
    max_dist = distances.max()
    if max_dist > 0:
        similarity = 1 - (distances / max_dist)
    else:
        similarity = np.ones_like(distances)
    
    # Ensure diagonal is 1 (self-similarity)
    np.fill_diagonal(similarity, 1.0)
    
    return similarity


# ============================================================
# MAIN INFERENCE
# ============================================================
def main():
    print("=" * 60)
    print("SNN-BASED PLAYER SIMILARITY INFERENCE")
    print("Using Euclidean Distance (Stricter Similarity)")
    print("=" * 60)
    
    # ---- 1. Load Data ----
    print("\n[1/5] Loading data...")
    
    allpos_file = DATA_DIR / "players_allpos_multiseason.csv"
    forwards_file = DATA_DIR / "forwards_multiseason.csv"
    
    if allpos_file.exists():
        df = pd.read_csv(allpos_file)
        SOURCE = "allpos_multiseason"
        print(f"  ALL POSITIONS: {len(df)} players")
    elif forwards_file.exists():
        df = pd.read_csv(forwards_file)
        SOURCE = "forwards_multiseason"
        print(f"  Forwards only: {len(df)} players")
    else:
        raise FileNotFoundError("No data found! Run 00_process_allpos.ipynb first.")
    
    # Find column names
    PLAYER_COL = find_col(df, ['player', 'Player', 'name'])
    TEAM_COL = find_col(df, ['squad', 'Squad', 'team', 'Team'])
    AGE_COL = find_col(df, ['age', 'Age'])
    POS_COL = find_col(df, ['pos', 'Pos', 'position'])
    NINETIES_COL = find_col(df, ['90s', '90s_x', '90s_y'])
    
    print(f"  Columns found: Player={PLAYER_COL}, Team={TEAM_COL}, Age={AGE_COL}, Pos={POS_COL}")
    
    # ---- 2. Load Features ----
    print("\n[2/5] Loading features...")
    
    allpos_features = DATA_DIR / 'clustering_features_allpos.txt'
    multiseason_features = DATA_DIR / 'clustering_features_multiseason.txt'
    
    if allpos_features.exists():
        FEATURES, FEATURE_NAMES = load_features(allpos_features)
    elif multiseason_features.exists():
        FEATURES, FEATURE_NAMES = load_features(multiseason_features)
    else:
        raise FileNotFoundError("No features file found!")
    
    FEATURES = [f for f in FEATURES if f in df.columns]
    print(f"  Loaded {len(FEATURES)} features")
    
    # Convert to per-90
    RATE_FEATURES = [
        'Per 90 Minutes', 'Per 90 Minutes.1', 'Per 90 Minutes.2', 'Per 90 Minutes.3',
        'Per 90 Minutes.4', 'Per 90 Minutes.5', 'Per 90 Minutes.6', 'Per 90 Minutes.7',
        'Per 90 Minutes.8', 'Per 90 Minutes.9',
        'Standard.2', 'Standard.3', 'Standard.4', 'Standard.5', 'Standard.6', 'Standard.11',
        'Total.2', 'Short.2', 'Medium.2', 'Long.2',
        'Take-Ons.2', 'Take-Ons.4', 'Aerial Duels.2', 'Challenges.2', 'Expected.4'
    ]
    
    df_per90 = df.copy()
    if NINETIES_COL and NINETIES_COL in df.columns:
        nineties = df[NINETIES_COL].replace(0, np.nan)
        for feat in FEATURES:
            if feat not in RATE_FEATURES and feat != '90s':
                df_per90[feat] = df[feat] / nineties
        df_per90[FEATURES] = df_per90[FEATURES].fillna(0)
        print("  Converted to per-90 format")
    
    # Scale features
    X = df_per90[FEATURES].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"  Feature matrix: {X_scaled.shape}")
    
    # ---- 3. Load Model and Compute Embeddings ----
    print("\n[3/5] Loading SNN model and computing embeddings...")
    
    EMB_DIM = 64
    model = SiameseNet(len(FEATURES), EMB_DIM).to(device)
    
    model_path = MODELS_DIR / 'siamese_allpos_best.pth'
    if model_path.exists():
        try:
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        except TypeError:
            model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    embeddings = model.embed(torch.FloatTensor(X_scaled).to(device))
    print(f"  Embeddings shape: {embeddings.shape}")
    
    # ---- 4. Compute Euclidean-Based Similarity ----
    print("\n[4/5] Computing Euclidean distance-based similarity...")
    
    similarity = compute_euclidean_similarity(embeddings)
    print(f"  Similarity matrix: {similarity.shape}")
    print(f"  Min: {similarity.min():.4f}, Max: {similarity.max():.4f}, Mean: {similarity.mean():.4f}")
    
    # ---- 5. Export to JSON ----
    print("\n[5/5] Exporting to JSON...")
    
    # Radar chart configuration - ALL STATS
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
        'PrgP': 'Progressive Passes',
        'KP': 'Key Passes',
        '1/3': 'Passes into Final 1/3',
        'PPA': 'Passes into Penalty Area',
        'CrsPA': 'Crosses into Penalty Area',
        'Att': 'Passes Attempted',
        'Cmp': 'Passes Completed',

        # --- POSSESSION ---
        'PrgR': 'Prog Passes Received',
        'PrgC': 'Progressive Carries',
        'Succ': 'Successful Dribbles',
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
        'Tkl+Int': 'Tackles + Interceptions',

        # --- AERIAL / MISC ---
        'Won': 'Aerials Won',
        'Lost': 'Aerials Lost',
        'Fls': 'Fouls Committed',
        'Fld': 'Fouls Drawn',
        'Off': 'Offsides',
        'CrdY': 'Yellow Cards',
        'CrdR': 'Red Cards',
    }
    
    RADAR_FEATURES = [f for f in RADAR_STATS.keys() if f in FEATURES]
    RADAR_LABELS = [RADAR_STATS[f] for f in RADAR_FEATURES]
    RADAR_LOW = [np.percentile(df_per90[f].dropna(), 5) for f in RADAR_FEATURES]
    RADAR_HIGH = [np.percentile(df_per90[f].dropna(), 95) for f in RADAR_FEATURES]
    
    # Players
    players = []
    for i, row in df.iterrows():
        players.append({
            'id': int(i),
            'name': str(row[PLAYER_COL]),
            'squad': str(row[TEAM_COL]) if pd.notna(row[TEAM_COL]) else None,
            'position': str(row[POS_COL]) if POS_COL and pd.notna(row[POS_COL]) else None,
            'age': int(row[AGE_COL]) if AGE_COL and pd.notna(row[AGE_COL]) else None,
            'nation': str(row.get('Nation')) if pd.notna(row.get('Nation')) else None,
            'league': str(row.get('League')) if pd.notna(row.get('League')) else None,
        })
    
    # Stats
    stats = []
    for i, row in df_per90.iterrows():
        for j, feat in enumerate(RADAR_FEATURES):
            stats.append({
                'player_id': int(i),
                'stat_name': RADAR_LABELS[j],
                'stat_key': feat,
                'value': round(float(row[feat]), 3) if pd.notna(row[feat]) else 0,
                'min_range': round(RADAR_LOW[j], 3),
                'max_range': round(RADAR_HIGH[j], 3)
            })
    
    # Similarities (top 50 per player using EUCLIDEAN)
    similarities = []
    for i in range(len(df)):
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
    
    print(f"\n  Players: {len(players)}")
    print(f"  Stats: {len(stats)}")
    print(f"  Similarities: {len(similarities)}")
    
    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nFiles saved to {DATA_DIR}/:")
    print(f"  • export_players.json")
    print(f"  • export_stats.json")
    print(f"  • export_similarities.json")
    print("\nNext: Run 'python upload_to_supabase.py' to upload to database")


if __name__ == "__main__":
    main()
