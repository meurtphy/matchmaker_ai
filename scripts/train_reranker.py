"""
train_reranker.py  – V2 re‑ranking model
=======================================
Entraîne un modèle LightGBM (fallback : GradientBoostingClassifier) à partir
 des paires \*pairs_train_*.parquet et en utilisant :

• similarité cosinus embeddings MiniLM (déjà pré‑calculés via embed_and_index.py)
• features simples : égalité CP, distance CP (différence des 2 premiers digits),
  longueur des textes, différence de longueur.

Le script supporte :
    python -m scripts.train_reranker --sample 1_200_000 --seed 13 \
        --out data/model/reranker_v2.joblib

Le modèle et la liste des features sont sauvegardés avec joblib.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    import lightgbm as lgb  # type: ignore

    _HAS_LGB = True
except ImportError:  # graceful fallback
    from sklearn.ensemble import GradientBoostingClassifier  # type: ignore

    _HAS_LGB = False

# ---------------------------------------------------------------------------
# Local paths (change if ta structure diffère)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/processed")
PAIR_GLOB = "pairs_train_*.parquet"
EMBED_DIR = Path("data/embeddings")  # contient *_vecs.f32 + *_meta.parquet
DEFAULT_OUT = Path("data/model/reranker_v2.joblib")

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def load_embeddings(dataset: str) -> Tuple[np.memmap, Dict[str, int]]:
    """Charge le memmap de vecteurs + construit un dict id → index."""
    vec_path = EMBED_DIR / f"{dataset}_vecs.f32"
    meta_path = EMBED_DIR / f"{dataset}_meta.parquet"
    if not vec_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Embeddings manquants pour {dataset}.")

    meta = pd.read_parquet(meta_path, columns=["id"])
    id_to_idx = {str(_id): i for i, _id in enumerate(meta["id"].astype(str))}
    # (n, 384) float32, mode="r" = read‑only → zéro copie RAM
    vecs = np.memmap(vec_path, mode="r", dtype="float32")
    dim = 384
    vecs = vecs.reshape(-1, dim)
    return vecs, id_to_idx


def cosine_batch(idx_a: np.ndarray, idx_b: np.ndarray, v_a: np.ndarray, v_b: np.ndarray) -> np.ndarray:
    """Produit cosinus pour 2 ensembles déjà normalisés L2 (inner‑product)."""
    # sélection → (n, dim)
    a = v_a[idx_a]
    b = v_b[idx_b]
    return (a * b).sum(axis=1)


def jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def build_features(df: pd.DataFrame, emb_rna, map_rna, emb_sir, map_sir) -> pd.DataFrame:
    # Cosine similarity
    idx_rna = df["id_rna"].astype(str).map(map_rna).to_numpy()
    idx_sir = df["id_sir"].astype(str).map(map_sir).to_numpy()
    cos = cosine_batch(idx_rna, idx_sir, emb_rna, emb_sir)

    # basic lexical lengths
    len_rna = df["texte_rna"].str.len()
    len_sir = df["texte_sir"].str.len()

    # CP exact match + region distance (2 premiers digits)
    cp_rna = df["cp_rna"].str[:5].fillna("00000")
    cp_sir = df["cp_sir"].str[:5].fillna("00000")
    exact_cp = (cp_rna == cp_sir).astype(int)
    region_dist = (cp_rna.str[:2].astype(int) - cp_sir.str[:2].astype(int)).abs()

    # Minimal jaccard token overlap (optional – un peu coûteux mais OK)
    jac = [jaccard(a, b) for a, b in zip(df["texte_rna"], df["texte_sir"])]

    feats = pd.DataFrame(
        {
            "cos": cos,
            "exact_cp": exact_cp,
            "region_dist": region_dist,
            "len_rna": len_rna,
            "len_sir": len_sir,
            "len_diff": (len_rna - len_sir).abs(),
            "jaccard": jac,
        }
    )
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Train reranker (LightGBM)")
    ap.add_argument("--sample", type=int, default=None, help="max lignes pour l'entraînement")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Charge tous les fichiers paires
    files = sorted(DATA_DIR.glob(PAIR_GLOB))
    if not files:
        raise FileNotFoundError("Aucun pairs_train_*.parquet trouvé")

    print("🚚 lecture", len(files), "fichiers pairs …")
    df_list = [pd.read_parquet(f, columns=[
        "id_rna", "id_sir", "texte_rna", "texte_sir", "cp_rna", "cp_sir", "label"
    ]) for f in files]
    pairs = pd.concat(df_list, ignore_index=True)

    # échantillonnage optionnel
    if args.sample and args.sample < len(pairs):
        pairs = pairs.sample(args.sample, random_state=args.seed)
    pairs = pairs.reset_index(drop=True)
    print("✅ paires chargées :", len(pairs))

    # 2. Embeddings memmap
    emb_rna, map_rna = load_embeddings("rna")
    emb_sir, map_sir = load_embeddings("sirene")

    # 3. Features
    feats = build_features(pairs, emb_rna, map_rna, emb_sir, map_sir)
    X = feats.values.astype(np.float32)
    y = pairs["label"].astype(int).values

    # 4. Split train/valid
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=args.seed)

    # 5. Model
    if _HAS_LGB:
        print("🧠 LightGBM …")
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        params = {
            "objective": "binary",
            "boosting_type": "gbdt",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": args.seed,
        }
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            early_stopping_rounds=30,
            verbose_eval=50,
        )
        best_iter = model.best_iteration
        print("🏁 best_iter", best_iter)
    else:
        print("⚠️ LightGBM absent → GradientBoostingClassifier")
        from sklearn.ensemble import GradientBoostingClassifier

        model = GradientBoostingClassifier(random_state=args.seed)
        model.fit(X_train, y_train)

    # 6. Eval
    if _HAS_LGB:
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    else:
        y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    print(f"🔍 AUC validation : {auc:.4f}")

    # 7. Persist
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": feats.columns.tolist(),
        "dim_emb": emb_rna.shape[1],
    }, args.out)
    print("✅ modèle enregistré →", args.out)


if __name__ == "__main__":
    main()
