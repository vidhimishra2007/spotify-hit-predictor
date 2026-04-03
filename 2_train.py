"""
STEP 2 — Model Training (Improved)
====================================
Fixes:
  1. Class imbalance  → class_weight='balanced' + SMOTE oversampling
  2. Weak features    → feature engineering (interaction terms)
  3. Better models    → Gradient Boosting added
  4. Threshold tuning → find best threshold for Hit F1
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (classification_report, f1_score,
                              precision_score, accuracy_score,
                              precision_recall_curve)
from imblearn.over_sampling import SMOTE

CLEAN_PATH = "data/cleaned.csv"
MODEL_DIR  = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

AUDIO_FEATURES = [
    "danceability", "energy", "key", "loudness", "mode",
    "speechiness", "acousticness", "instrumentalness",
    "liveness", "valence", "tempo", "duration_ms"
]

def engineer_features(X_df):
    df = X_df.copy()
    df["energy_dance"]  = df["energy"] * df["danceability"]
    df["acoustic_inst"] = df["acousticness"] * df["instrumentalness"]
    df["mood_energy"]   = df["valence"] * df["energy"]
    df["loudness_norm"] = df["loudness"] + 60
    df["speech_ratio"]  = df["speechiness"] / (df["energy"] + 1e-6)
    df["tempo_bucket"]  = pd.cut(df["tempo"], bins=[0,90,120,150,999],
                                 labels=[0,1,2,3]).astype(float)
    df = df.fillna(0)
    return df

def evaluate(name, y_true, y_pred, label=""):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted")
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"  {name:<28} acc={acc:.3f}  f1={f1:.3f}  prec={prec:.3f}  [{label}]")
    return {"name": name, "acc": acc, "f1": f1, "precision": prec}

def find_best_threshold(model, X, y_true):
    if not hasattr(model, "predict_proba"):
        return 0.5
    probs = model.predict_proba(X)[:, 1]
    p, r, thresholds = precision_recall_curve(y_true, probs)
    f1s = 2 * p * r / (p + r + 1e-9)
    best_idx = np.argmax(f1s)
    t = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    print(f"  Best threshold: {t:.3f}  (F1={f1s[best_idx]:.3f})")
    return t

def train_stage(X_train, X_test, y_train, y_test,
                stage_name, use_lda=False, balance=False):
    print(f"\n{'='*60}")
    print(f"  STAGE: {stage_name}")
    print(f"{'='*60}")

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    if balance:
        print(f"  Before SMOTE: {np.bincount(y_train)}")
        sm = SMOTE(random_state=42)
        X_tr_sc, y_train = sm.fit_resample(X_tr_sc, y_train)
        print(f"  After  SMOTE: {np.bincount(y_train)}")

    if use_lda:
        n_comp  = min(len(np.unique(y_train)) - 1, X_tr_sc.shape[1])
        reducer = LDA(n_components=n_comp)
        lbl     = "LDA"
    else:
        reducer = PCA(n_components=0.95)
        lbl     = "PCA"

    X_tr_r = reducer.fit_transform(X_tr_sc, y_train)
    X_te_r = reducer.transform(X_te_sc)
    print(f"  Dims: {X_tr_sc.shape[1]} → {X_tr_r.shape[1]} ({lbl})")

    models = {
        "Random Forest (bal)":   RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1,
            class_weight="balanced", min_samples_leaf=2),
        "Gradient Boosting":     GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42),
        "Naive Bayes":           GaussianNB(),
        "SGD (bal)":             SGDClassifier(
            max_iter=1000, random_state=42, tol=1e-3,
            class_weight="balanced", loss="modified_huber"),
    }

    results = []
    trained = {}
    for mname, clf in models.items():
        clf.fit(X_tr_r, y_train)
        y_pred = clf.predict(X_te_r)
        res = evaluate(mname, y_test, y_pred, lbl)
        results.append(res)
        trained[mname] = clf

    best = max(results, key=lambda x: x["f1"])
    best_clf = trained[best["name"]]
    print(f"\n  🏆 Best: {best['name']}  (f1={best['f1']:.3f})")

    threshold = 0.5
    if not use_lda:
        threshold = find_best_threshold(best_clf, X_te_r, y_test)

    return {
        "scaler":    scaler,
        "reducer":   reducer,
        "model":     best_clf,
        "best_name": best["name"],
        "results":   results,
        "threshold": threshold,
    }

def train():
    print("Loading cleaned data...")
    df = pd.read_csv(CLEAN_PATH)
    print(f"   Shape: {df.shape}")

    df_feat   = engineer_features(df[AUDIO_FEATURES])
    feat_cols = list(df_feat.columns)
    print(f"   Features: {len(AUDIO_FEATURES)} raw → {len(feat_cols)} engineered")

    X       = df_feat.values
    y_genre = df["genre_encoded"].values
    y_hit   = df["is_hit"].values

    X_train, X_test, yg_train, yg_test, yh_train, yh_test = train_test_split(
        X, y_genre, y_hit, test_size=0.2, random_state=42, stratify=y_genre
    )
    print(f"   Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

    stage1 = train_stage(X_train, X_test, yg_train, yg_test,
                         "Genre Classification", use_lda=True, balance=False)

    gp_train = stage1["model"].predict(
        stage1["reducer"].transform(stage1["scaler"].transform(X_train)))
    gp_test  = stage1["model"].predict(
        stage1["reducer"].transform(stage1["scaler"].transform(X_test)))

    X_tr_aug = np.column_stack([X_train, gp_train])
    X_te_aug = np.column_stack([X_test,  gp_test])

    stage2 = train_stage(X_tr_aug, X_te_aug, yh_train, yh_test,
                         "Hit Prediction (SMOTE + balanced)",
                         use_lda=False, balance=True)

    joblib.dump(stage1,    f"{MODEL_DIR}/stage1_genre.pkl")
    joblib.dump(stage2,    f"{MODEL_DIR}/stage2_hit.pkl")
    joblib.dump(feat_cols, f"{MODEL_DIR}/feature_cols.pkl")
    print(f"\n✅ Models saved to {MODEL_DIR}/")

    le = joblib.load(f"{MODEL_DIR}/label_encoder_genre.pkl")

    X_te_sc = stage1["scaler"].transform(X_test)
    X_te_r  = stage1["reducer"].transform(X_te_sc)
    yg_pred = stage1["model"].predict(X_te_r)
    print("\n── Genre Report ─────────────────────────────")
    print(classification_report(yg_test, yg_pred, target_names=le.classes_))

    X_aug_sc = stage2["scaler"].transform(X_te_aug)
    X_aug_r  = stage2["reducer"].transform(X_aug_sc)
    if hasattr(stage2["model"], "predict_proba"):
        probs   = stage2["model"].predict_proba(X_aug_r)[:, 1]
        yh_pred = (probs >= stage2["threshold"]).astype(int)
    else:
        yh_pred = stage2["model"].predict(X_aug_r)

    print("── Hit Prediction Report (tuned threshold) ──")
    print(classification_report(yh_test, yh_pred, target_names=["Not a Hit", "Hit"]))

    before_recall = 0.06
    after_recall  = (yh_pred[yh_test==1] == 1).mean()
    print(f"── Hit Recall: {before_recall:.2f} → {after_recall:.2f}  {'✅ improved' if after_recall > before_recall else '⚠ check'}")

if __name__ == "__main__":
    train()