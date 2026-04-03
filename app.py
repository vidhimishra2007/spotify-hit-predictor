"""
Spotify Genre Hit Predictor — Aesthetic UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import subprocess

# ── Page config ─────────────────────────────────────────
st.set_page_config(
    page_title="Hit Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Master CSS ───────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #080810 !important;
    color: #e2e2f0 !important;
    font-family: 'Syne', sans-serif !important;
}

#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="collapsedControl"] { display: none !important; }

[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
.block-container { padding: 0 2.5rem 3rem !important; max-width: 1300px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0e0e1a; }
::-webkit-scrollbar-thumb { background: #1db954; border-radius: 2px; }

[data-testid="stSlider"] > div > div > div { background: #1e1e2e !important; }
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #1db954 !important;
    border: 2px solid #080810 !important;
    box-shadow: 0 0 8px rgba(29,185,84,0.6) !important;
    width: 18px !important; height: 18px !important;
}
[data-testid="stSlider"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important; color: #6b6b8a !important;
}
[data-testid="stSlider"] label p {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important; color: #a0a0c0 !important; font-weight: 500 !important;
}
[data-testid="stRadio"] label {
    font-family: 'Syne', sans-serif !important;
    font-size: 13px !important; color: #a0a0c0 !important;
}
[data-testid="stRadio"] [aria-checked="true"] div:first-child {
    border-color: #1db954 !important; background: #1db954 !important;
}
hr { border: none !important; border-top: 1px solid #1e1e2e !important; margin: 2rem 0 !important; }
[data-testid="column"] { padding: 0 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    s1 = joblib.load("models/stage1_genre.pkl")
    s2 = joblib.load("models/stage2_hit.pkl")
    le = joblib.load("models/label_encoder_genre.pkl")
    df = pd.read_csv("data/cleaned.csv")
    return s1, s2, le, df

AUDIO_FEATURES = [
    "danceability","energy","key","loudness","mode",
    "speechiness","acousticness","instrumentalness",
    "liveness","valence","tempo","duration_ms"
]

GENRE_META = {
    "pop":   {"emoji":"🎤","color":"#ff6b9d","label":"POP"},
    "rap":   {"emoji":"🎙️","color":"#ffd93d","label":"RAP"},
    "rock":  {"emoji":"🎸","color":"#ff6b35","label":"ROCK"},
    "latin": {"emoji":"💃","color":"#c77dff","label":"LATIN"},
    "r&b":   {"emoji":"🎷","color":"#4cc9f0","label":"R&B"},
    "edm":   {"emoji":"🎧","color":"#1db954","label":"EDM"},
}

def engineer_features(X_df):
    df = X_df.copy()
    df["energy_dance"]  = df["energy"] * df["danceability"]
    df["acoustic_inst"] = df["acousticness"] * df["instrumentalness"]
    df["mood_energy"]   = df["valence"] * df["energy"]
    df["loudness_norm"] = df["loudness"] + 60
    df["speech_ratio"]  = df["speechiness"] / (df["energy"] + 1e-6)
    df["tempo_bucket"]  = pd.cut(df["tempo"], bins=[0,90,120,150,999],
                                 labels=[0,1,2,3]).astype(float).fillna(0)
    return df

try:
    stage1, stage2, le, df = load_models()
    ok = True
except Exception as e:
    ok = False
    st.error(f"Run `1_preprocess.py` and `2_train.py` first.\n\n{e}")

# ── Hero ─────────────────────────────────────────────────
st.markdown("""
<div style="padding: 48px 0 32px; position: relative; overflow: hidden;">
  <div style="position:absolute;top:-80px;left:-100px;width:500px;height:400px;border-radius:50%;
    background:radial-gradient(circle,rgba(29,185,84,0.07) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="position:absolute;top:-40px;right:-60px;width:350px;height:350px;border-radius:50%;
    background:radial-gradient(circle,rgba(76,201,240,0.05) 0%,transparent 70%);pointer-events:none;"></div>
  <div style="position:relative;z-index:1;">
    <div style="display:inline-flex;align-items:center;gap:8px;
      background:rgba(29,185,84,0.08);border:1px solid rgba(29,185,84,0.2);
      border-radius:100px;padding:5px 14px;margin-bottom:20px;">
      <span style="width:6px;height:6px;border-radius:50%;background:#1db954;
        display:inline-block;box-shadow:0 0 6px #1db954;"></span>
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
        letter-spacing:0.18em;color:#1db954;text-transform:uppercase;">Billboard 100 Predictor</span>
    </div>
    <h1 style="font-family:'Syne',sans-serif;font-size:clamp(2.8rem,5vw,4.5rem);
      font-weight:800;line-height:1;margin:0 0 12px;letter-spacing:-0.02em;
      background:linear-gradient(135deg,#ffffff 0%,#a0a0c0 100%);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
      SPOTIFY<br>
      <span style="background:linear-gradient(135deg,#1db954 0%,#4ade80 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">HIT PREDICTOR</span>
    </h1>
    <p style="font-family:'Syne',sans-serif;font-size:15px;color:#6b6b8a;
      margin:0;max-width:480px;line-height:1.6;font-weight:400;">
      Dial in your audio features. Our two-stage ML pipeline predicts genre
      and whether your track will chart on the Billboard 100.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.stop()

# ── Layout ───────────────────────────────────────────────
col_left, col_right = st.columns([1.5, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
        letter-spacing:0.2em;color:#1db954;text-transform:uppercase;">01 — Controls</span>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,#1e1e2e,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        danceability     = st.slider("Danceability",     0.0, 1.0, 0.65, 0.01)
        energy           = st.slider("Energy",           0.0, 1.0, 0.70, 0.01)
        valence          = st.slider("Valence",          0.0, 1.0, 0.50, 0.01)
        acousticness     = st.slider("Acousticness",     0.0, 1.0, 0.15, 0.01)
        speechiness      = st.slider("Speechiness",      0.0, 1.0, 0.06, 0.01)
        liveness         = st.slider("Liveness",         0.0, 1.0, 0.12, 0.01)
    with c2:
        instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00, 0.01)
        tempo            = st.slider("Tempo (BPM)",      50.0, 220.0, 120.0, 1.0)
        loudness         = st.slider("Loudness (dB)",    -60.0, 0.0, -7.0, 0.5)
        key              = st.slider("Key",              0, 11, 5)
        mode             = st.radio("Mode", [0, 1],
                            format_func=lambda x: "Minor" if x == 0 else "Major",
                            horizontal=True)
        duration_s       = st.slider("Duration (sec)",  30, 600, 210)

# ── Predictions ──────────────────────────────────────────
raw = pd.DataFrame([[
    danceability, energy, key, loudness, int(mode),
    speechiness, acousticness, instrumentalness,
    liveness, valence, tempo, duration_s * 1000
]], columns=AUDIO_FEATURES)
features = engineer_features(raw).values

X1    = stage1["reducer"].transform(stage1["scaler"].transform(features))
g_enc = stage1["model"].predict(X1)[0]
g_name = le.inverse_transform([g_enc])[0]
g_conf = (stage1["model"].predict_proba(X1)[0].max() * 100
          if hasattr(stage1["model"], "predict_proba") else None)

feat_aug = np.column_stack([features, [[g_enc]]])
X2       = stage2["reducer"].transform(stage2["scaler"].transform(feat_aug))
if hasattr(stage2["model"], "predict_proba"):
    hit_prob = stage2["model"].predict_proba(X2)[0][1] * 100
    threshold = stage2.get("threshold", 0.5) * 100
    hit_pred = 1 if hit_prob >= threshold else 0
else:
    hit_pred = stage2["model"].predict(X2)[0]
    hit_prob = 50.0

gm = GENRE_META.get(g_name, {"emoji":"🎵","color":"#1db954","label":g_name.upper()})

with col_right:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:20px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
        letter-spacing:0.2em;color:#1db954;text-transform:uppercase;">02 — Output</span>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,#1e1e2e,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    # Genre card
    st.markdown(f"""
    <div style="background:#0e0e1a;border:1px solid #1e1e2e;
      border-top:2px solid {gm['color']};border-radius:16px;
      padding:24px 20px;margin-bottom:14px;position:relative;overflow:hidden;">
      <div style="position:absolute;top:-40px;right:-40px;width:140px;height:140px;border-radius:50%;
        background:radial-gradient(circle,{gm['color']}18 0%,transparent 70%);"></div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
        letter-spacing:0.2em;color:#3a3a5a;text-transform:uppercase;margin-bottom:10px;">Predicted Genre</div>
      <div style="display:flex;align-items:center;gap:14px;">
        <span style="font-size:2rem">{gm['emoji']}</span>
        <div>
          <div style="font-family:'Syne',sans-serif;font-size:2rem;font-weight:800;
            color:{gm['color']};line-height:1;letter-spacing:-0.02em;">{gm['label']}</div>
          {'<div style="font-family:JetBrains Mono,monospace;font-size:11px;color:#3a3a5a;margin-top:4px;">' + f'{g_conf:.0f}% confidence</div>' if g_conf else ''}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Hit card
    if hit_pred == 1:
        bg, bdr, glow = "linear-gradient(135deg,#0a2e16,#0d3b1e)", "#1db954", "rgba(29,185,84,0.15)"
        icon, verdict, vcolor = "🚀", "HIT", "#1db954"
        sub = "High likelihood of charting"
    else:
        bg, bdr, glow = "linear-gradient(135deg,#1a0a0a,#220d0d)", "#e63946", "rgba(230,57,70,0.12)"
        icon, verdict, vcolor = "📉", "NOT A HIT", "#e63946"
        sub = "Low likelihood of charting"

    st.markdown(f"""
    <div style="background:{bg};border:1px solid {bdr}44;border-top:2px solid {bdr};
      border-radius:16px;padding:28px 24px;margin-bottom:14px;text-align:center;
      position:relative;overflow:hidden;">
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);
        width:200px;height:200px;border-radius:50%;
        background:radial-gradient(circle,{glow} 0%,transparent 70%);pointer-events:none;"></div>
      <div style="font-size:2.8rem;margin-bottom:10px;position:relative;">{icon}</div>
      <div style="font-family:'Syne',sans-serif;font-size:2.2rem;font-weight:800;
        color:{vcolor};letter-spacing:0.04em;line-height:1;position:relative;">{verdict}</div>
      <div style="font-family:'JetBrains Mono',monospace;font-size:11px;
        color:#6b6b8a;margin-top:8px;letter-spacing:0.05em;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

    # Probability bar
    bc = "#1db954" if hit_pred == 1 else "#e63946"
    st.markdown(f"""
    <div style="margin-bottom:24px;">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
        <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
          color:#3a3a5a;letter-spacing:0.15em;text-transform:uppercase;">Hit Probability</span>
        <span style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{bc};">{hit_prob:.1f}%</span>
      </div>
      <div style="background:#1e1e2e;border-radius:100px;height:6px;overflow:hidden;">
        <div style="width:{hit_prob:.1f}%;height:100%;border-radius:100px;
          background:linear-gradient(90deg,{bc},{bc}aa);"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Similar songs
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
      <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
        letter-spacing:0.2em;color:#1db954;text-transform:uppercase;">03 — Similar Tracks</span>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,#1e1e2e,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

    genre_df = df[df["playlist_genre"] == g_name].copy()
    feat_arr = genre_df[AUDIO_FEATURES].values
    from sklearn.preprocessing import MinMaxScaler
    sc_sim = MinMaxScaler()
    normed = sc_sim.fit_transform(np.vstack([feat_arr, features[0]]))
    dists  = np.linalg.norm(normed[:-1] - normed[-1], axis=1)
    genre_df["_dist"] = dists
    top5 = genre_df.nsmallest(5, "_dist")[["track_name","track_artist","track_popularity","is_hit"]]

    for _, row in top5.iterrows():
        is_h  = row["is_hit"] == 1
        dc    = "#1db954" if is_h else "#e63946"
        pop   = int(row["track_popularity"])
        st.markdown(f"""
        <div style="background:#0e0e1a;border:1px solid #1a1a2e;border-radius:12px;
          padding:14px 16px;margin-bottom:8px;position:relative;overflow:hidden;">
          <div style="position:absolute;left:0;top:0;bottom:0;width:3px;
            background:{dc};border-radius:12px 0 0 12px;"></div>
          <div style="display:flex;justify-content:space-between;align-items:flex-start;padding-left:8px;">
            <div style="flex:1;min-width:0;">
              <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:600;
                color:#e2e2f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                {row['track_name']}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                color:#3a3a5a;margin-top:2px;">{row['track_artist']}</div>
            </div>
            <div style="text-align:right;margin-left:12px;flex-shrink:0;">
              <div style="font-family:'Syne',sans-serif;font-size:13px;font-weight:700;color:{dc};">{pop}</div>
              <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#3a3a5a;letter-spacing:0.1em;">
                {'Hit' if is_h else 'Miss'}</div>
            </div>
          </div>
          <div style="margin-top:10px;padding-left:8px;">
            <div style="background:#1a1a2e;border-radius:100px;height:3px;overflow:hidden;">
              <div style="width:{pop}%;height:100%;
                background:linear-gradient(90deg,{dc}80,{dc}30);border-radius:100px;"></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Genre stats ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:24px;">
  <span style="font-family:'JetBrains Mono',monospace;font-size:10px;
    letter-spacing:0.2em;color:#1db954;text-transform:uppercase;">04 — Genre Intelligence</span>
  <div style="flex:1;height:1px;background:linear-gradient(90deg,#1e1e2e,transparent);"></div>
</div>
""", unsafe_allow_html=True)

genre_counts = df["playlist_genre"].value_counts()
genre_hits   = df[df["is_hit"]==1]["playlist_genre"].value_counts()
cols_g       = st.columns(6)

for i, (g, cnt) in enumerate(genre_counts.items()):
    hit_pct   = int(genre_hits.get(g, 0) / cnt * 100)
    gmi       = GENRE_META.get(g, {"emoji":"🎵","color":"#1db954","label":g.upper()})
    is_active = (g == g_name)
    border    = f"1px solid {gmi['color']}60" if is_active else "1px solid #1a1a2e"
    bg        = f"linear-gradient(135deg,{gmi['color']}0d,#0e0e1a)" if is_active else "#0e0e1a"
    with cols_g[i]:
        st.markdown(f"""
        <div style="background:{bg};border:{border};border-radius:14px;
          padding:20px 14px;text-align:center;position:relative;overflow:hidden;">
          {'<div style="position:absolute;top:0;left:0;right:0;height:2px;background:' + gmi['color'] + ';"></div>' if is_active else ''}
          <div style="font-size:1.6rem;margin-bottom:8px;">{gmi['emoji']}</div>
          <div style="font-family:'Syne',sans-serif;font-size:11px;font-weight:700;
            letter-spacing:0.12em;color:{'#e2e2f0' if is_active else '#4a4a6a'};
            margin-bottom:10px;">{gmi['label']}</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.6rem;
            font-weight:800;color:{gmi['color']};line-height:1;">{hit_pct}%</div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
            color:#2a2a4a;letter-spacing:0.1em;margin-top:3px;text-transform:uppercase;">hit rate</div>
          <div style="background:#1a1a2e;border-radius:100px;height:3px;margin-top:10px;overflow:hidden;">
            <div style="width:{hit_pct}%;height:100%;background:{gmi['color']};border-radius:100px;"></div>
          </div>
          <div style="font-family:'JetBrains Mono',monospace;font-size:9px;
            color:#2a2a4a;margin-top:8px;">{cnt:,} tracks</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:48px 0 16px;font-family:'JetBrains Mono',monospace;
  font-size:10px;letter-spacing:0.15em;color:#1e1e2e;text-transform:uppercase;">
  Spotify Hit Predictor &nbsp;·&nbsp; Random Forest + LDA/PCA &nbsp;·&nbsp; 28,352 tracks &nbsp;·&nbsp; 6 genres
</div>
""", unsafe_allow_html=True)
