"""
STEP 1 — Data Preprocessing
============================
- Load raw CSV
- Drop duplicates & nulls
- Engineer 'is_hit' label (popularity >= 60)
- Encode genre labels
- Save cleaned data + label encoders
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# ── Paths ──────────────────────────────────────────────
RAW_PATH   = "data/spotify_songs.csv"
CLEAN_PATH = "data/cleaned.csv"
MODEL_DIR  = "models"
os.makedirs("data", exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Audio features used by models ─────────────────────
AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms"
]
 
def preprocess():
    print(" Loading data...")
    df = pd.read_csv(RAW_PATH)
    print(f"Raw shape: {df.shape}")
 
# ── 1. Drop duplicates & nulls ─────────────────────
    df.drop_duplicates(subset="track_id", inplace=True)
    df.dropna(subset=["track_name", "track_artist", "playlist_genre"], inplace=True)
    print(f"   After cleaning: {df.shape}")
 
# ── 2. Create hit label (popularity >= 60 = hit) ───
    df["is_hit"] = (df["track_popularity"] >= 60).astype(int)
    hit_rate = df["is_hit"].mean() * 100
    print(f"   Hit rate: {hit_rate:.1f}% of songs are hits")

# ── 3. Encode genre ────────────────────────────────
    le_genre = LabelEncoder()
    df["genre_encoded"] = le_genre.fit_transform(df["playlist_genre"])
    print(f"   Genres: {list(le_genre.classes_)}")

# ── 4. Save ────────────────────────────────────────
    df.to_csv(CLEAN_PATH, index=False)
    joblib.dump(le_genre, f"{MODEL_DIR}/label_encoder_genre.pkl")
    print(f"\n✅ Saved cleaned data → {CLEAN_PATH}")
    print(f"✅ Saved genre encoder → {MODEL_DIR}/label_encoder_genre.pkl")

# ── 5. Quick EDA summary ───────────────────────────
    print("\n── Feature Summary ──────────────────────────")
    print(df[AUDIO_FEATURES + ["track_popularity", "is_hit"]].describe().round(3).to_string())
 
    print("\n── Hit rate per genre ───────────────────────")
    print(df.groupby("playlist_genre")["is_hit"].mean().mul(100).round(1).to_string())
 
    return df

if __name__ == "__main__":
    preprocess()