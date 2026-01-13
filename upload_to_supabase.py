"""
Simple script to upload player data to Supabase.
Run: python upload_to_supabase.py

Requires: pip install python-dotenv supabase
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables from .env file
load_dotenv()

# ============================================================
# CONFIGURATION - Loaded from .env file
# ============================================================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file")

DATA_DIR = Path("data")

# ============================================================
# CONNECT TO SUPABASE
# ============================================================
print("Connecting to Supabase...")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Connected!\n")

# ============================================================
# LOAD JSON FILES
# ============================================================
print("Loading data files...")

with open(DATA_DIR / "export_players.json", "r", encoding="utf-8") as f:
    players = json.load(f)

with open(DATA_DIR / "export_stats.json", "r", encoding="utf-8") as f:
    stats = json.load(f)

with open(DATA_DIR / "export_similarities.json", "r", encoding="utf-8") as f:
    similarities = json.load(f)

print(f"  Players: {len(players)}")
print(f"  Stats: {len(stats)}")
print(f"  Similarities: {len(similarities)}")

# ============================================================
# UPLOAD PLAYERS
# ============================================================
print("\n[1/3] Uploading players...")

# Upload in batches of 500
BATCH_SIZE = 500
for i in range(0, len(players), BATCH_SIZE):
    batch = players[i:i + BATCH_SIZE]
    supabase.table("players").upsert(batch).execute()
    print(f"  Uploaded {min(i + BATCH_SIZE, len(players))}/{len(players)}")

print("  [DONE] Players uploaded!")

# ============================================================
# UPLOAD STATS
# ============================================================
print("\n[2/3] Uploading stats...")

print("  Clearing existing stats (fixing duplicates)...")
# Delete all stats where player_id is not -1 (effectively all)
try:
    supabase.table("player_stats").delete().neq("player_id", -1).execute()
    print("  [CLEANED] Old stats removed.")
except Exception as e:
    print(f"  [WARNING] Could not clear stats: {e}")

for i in range(0, len(stats), BATCH_SIZE):
    batch = stats[i:i + BATCH_SIZE]
    supabase.table("player_stats").upsert(batch).execute()
    print(f"  Uploaded {min(i + BATCH_SIZE, len(stats))}/{len(stats)}")

print("  [DONE] Stats uploaded!")

# ============================================================
# UPLOAD SIMILARITIES
# ============================================================
print("\n[3/3] Uploading similarities...")

for i in range(0, len(similarities), BATCH_SIZE):
    batch = similarities[i:i + BATCH_SIZE]
    supabase.table("player_similarity").upsert(batch, on_conflict="player_id,similar_player_id").execute()
    print(f"  Uploaded {min(i + BATCH_SIZE, len(similarities))}/{len(similarities)}")

print("  [DONE] Similarities uploaded!")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 50)
print("[SUCCESS] ALL DATA UPLOADED SUCCESSFULLY!")
print("=" * 50)
