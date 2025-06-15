#!/usr/bin/env python3
"""
train_reranker.py  – v2.3 (robuste LightGBM 4.x)
================================================
Reclasse les paires RNA ↔ SIRENE à l’aide d’un **LightGBM Ranker**.
Optimisé pour gros volumes : échantillonnage PyArrow, float32 compact, logs.

Usage minimal
-------------
```bash
python v2/train_reranker.py --sample_rows 5_000_000
```

Options clés
------------
--sample_rows N   limite les lignes chargées (RAM)
--gpu             active l’entraînement GPU (si LightGBM GPU dispo)
--num_leaves, --learning_rate, etc. → hyper‑params habituels.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.model_selection import train_test_split

# ──────────────────────────── Chemins ────────────────────────────
DATA_PROC = Path("data/processed")
MODEL_DIR = Path("data/model")

PAIRS_TRAIN = DATA_PROC / "pairs_train.parquet"
PAIRS_EVAL  = DATA_PROC / "pairs_eval.parquet"
RNA_CLEAN   = DATA_PROC / "rna_clean.parquet"
SIRENE_CLEAN= DATA_PROC / "sirene_clean.parquet"
MODEL_PATH  = MODEL_DIR / "reranker_lgbm.txt"

# ──────────────────────────── Helpers ────────────────────────────

def _token_len(txt: str) -> int:  # lightning‑fast token len
    return len(str(txt).split())


def _read_sample(parquet: Path, n_rows: int | None) -> pd.DataFrame:
    if n_rows is None:
        return pd.read_parquet(parquet)
    logging.info(f"◆ lecture échantillon {n_rows:,} → {parquet.name}")
    dset = ds.dataset(parquet)
    table = dset.head(n_rows)  # Arrow 11 hot‑path
    return table.to_pandas()


def _build_feats(pairs: pd.DataFrame, rna: pd.DataFrame, sir: pd.DataFrame) -> pd.DataFrame:
    rna_meta = rna[["id", "cp", "texte"]].rename(columns={"id": "id_rna", "cp": "cp_rna", "texte": "txt_rna"})
    sir_meta = sir[["siret", "cp", "texte"]].rename(columns={"cp": "cp_sir", "texte": "txt_sir"})
    df = pairs.merge(rna_meta, on="id_rna").merge(sir_meta, on="siret")
    df["len_rna"]  = df["txt_rna"].map(_token_len, na_action="ignore")
    df["len_sir"]  = df["txt_sir"].map(_token_len, na_action="ignore")
    df["len_diff"] = (df["len_rna"] - df["len_sir"]).abs()
    df["same_cp"]  = (df["cp_rna"] == df["cp_sir"]).astype(np.int8)
    return df


def _prep_lgbm(df: pd.DataFrame, cols: List[str]):
    X = df[cols].astype(np.float32).values
    y = df["label"].astype(np.int8).values
    group = df.groupby("id_rna", sort=False).size().to_numpy(np.int32)
    return X, y, group

# ──────────────────────────── Train ────────────────────────────

def train(params: dict, test_frac: float, sample_rows: int | None, gpu: bool):
    logging.info("🚀 Launch training")
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    pairs_tr = _read_sample(PAIRS_TRAIN, sample_rows)
    pairs_ev = _read_sample(PAIRS_EVAL,  sample_rows//4 if sample_rows else None)
    rna_df   = pd.read_parquet(RNA_CLEAN, columns=["id", "cp", "texte"])
    sir_df   = pd.read_parquet(SIRENE_CLEAN, columns=["siret", "cp", "texte"])

    feats = ["cos_sim", "len_rna", "len_sir", "len_diff", "same_cp"]

    logging.info("🔧 feature eng train…")
    df_tr = _build_feats(pairs_tr, rna_df, sir_df)
    logging.info("🔧 feature eng eval…")
    df_ev = _build_feats(pairs_ev, rna_df, sir_df)

    # Split by query id — keeps group structure intact
    q_ids = df_tr["id_rna"].unique()
    q_tr, q_val = train_test_split(q_ids, test_size=test_frac, random_state=42)
    df_train = df_tr[df_tr["id_rna"].isin(q_tr)]
    df_valid = df_tr[df_tr["id_rna"].isin(q_val)]

    X_tr, y_tr, g_tr = _prep_lgbm(df_train, feats)
    X_val, y_val, g_val = _prep_lgbm(df_valid, feats)

    if gpu:
        params |= {"device_type": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}

    lgb_tr = lgb.Dataset(X_tr, y_tr, group=g_tr, free_raw_data=False)
    lgb_val= lgb.Dataset(X_val, y_val, group=g_val, reference=lgb_tr, free_raw_data=False)

    booster = lgb.train(
        params,
        lgb_tr,
        valid_sets=[lgb_tr, lgb_val],
        valid_names=["train", "valid"],
        num_boost_round=params["num_boost_round"],
        callbacks=[
            lgb.early_stopping(params["early_stopping_rounds"], verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )

    booster.save_model(str(MODEL_PATH))
    logging.info(f"💾 saved → {MODEL_PATH}")

    # Global eval on df_ev
    X_ev, y_ev, g_ev = _prep_lgbm(df_ev, feats)
    y_hat = booster.predict(X_ev, num_iteration=booster.best_iteration)
    ap = average_precision_score(y_ev, y_hat)
    splits = np.split(y_hat, np.cumsum(g_ev)[:-1])
    y_true_grp = np.split(y_ev, np.cumsum(g_ev)[:-1])
    ndcg = np.mean([ndcg_score([yt], [yp]) for yt, yp in zip(y_true_grp, splits)])
    logging.info(f"✅ Global eval — AP={ap:.4f} | NDCG={ndcg:.4f}")

# ──────────────────────────── CLI ────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--boosting", default="gbdt", choices=["gbdt", "dart", "goss"])
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_boost_round", type=int, default=500)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--sample_rows", type=int, default=None, help="Max lignes pairs_train")
    p.add_argument("--gpu", action="store_true", help="GPU LightGBM")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    lgb_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "boosting": args.boosting,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "verbose": -1,
        "num_boost_round": args.num_boost_round,
        "early_stopping_rounds": args.early_stopping_rounds,
        "seed": 42,
    }

    train(lgb_params, test_frac=args.test_size,
          sample_rows=args.sample_rows, gpu=args.gpu)
