#!/usr/bin/env python3
"""
train_reranker.py – v2.5
------------------------
• Early‑stopping LightGBM 4.x (callback)
• Échantillonnage **stratifié** : on conserve *tous* les positifs, puis on ajoute
  des négatifs aléatoires jusqu’à `--sample_rows` lignes (ou tout le fichier).
• Logs clairs sur la répartition pos/neg.
• Robuste : métriques ignorent les groupes sans positifs ou d’une seule doc.
"""
from __future__ import annotations
import argparse, logging, os
from pathlib import Path
from typing import List
import lightgbm as lgb, numpy as np, pandas as pd, pyarrow.dataset as ds
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.model_selection import train_test_split

DATA = Path("data/processed")
PAIRS_TRAIN, PAIRS_EVAL = DATA/"pairs_train.parquet", DATA/"pairs_eval.parquet"
RNA_CLEAN, SIRENE_CLEAN = DATA/"rna_clean.parquet", DATA/"sirene_clean.parquet"
MODEL_PATH = Path("data/model/reranker_lgbm.txt")

def _token_len(t: str) -> int: return len(str(t).split())

def _read_sample(path: Path, n: int | None) -> pd.DataFrame:
    """Lit *path*; garde tous les positifs, complète en négatifs jusqu’à n."""
    df = ds.dataset(path).to_table().to_pandas() if n else pd.read_parquet(path)
    if n is None: return df
    pos = df[df.label == 1]
    n_pos = len(pos)
    if n_pos == 0:
        logging.warning("fichier %s : 0 positif", path.name)
        return df.sample(n=min(n, len(df)), random_state=42)
    neg_pool = df[df.label == 0]
    n_neg = max(min(n - n_pos, len(neg_pool)), 0)
    neg = neg_pool.sample(n=n_neg, random_state=42)
    logging.info("   ↪︎ échantillon : %d pos | %d neg", n_pos, n_neg)
    return pd.concat([pos, neg], ignore_index=True)

def _build_feats(p, r, s):
    r = r.rename(columns={"id":"id_rna","cp":"cp_rna","texte":"txt_rna"})
    s = s.rename(columns={"cp":"cp_sir","texte":"txt_sir"})
    d = p.merge(r, on="id_rna").merge(s, on="siret")
    d["len_rna"] = d["txt_rna"].map(_token_len)
    d["len_sir"] = d["txt_sir"].map(_token_len)
    d["len_diff"] = (d["len_rna"]-d["len_sir"]).abs()
    d["same_cp"] = (d["cp_rna"]==d["cp_sir"]).astype(np.int8)
    return d

def _prep(df, cols):
    X = df[cols].astype(np.float32).values
    y = df.label.astype(np.int8).values
    g = df.groupby("id_rna", sort=False).size().to_numpy(np.int32)
    return X, y, g

def train(params, test_frac, sample, gpu):
    logging.info("🚀 training")
    pt = _read_sample(PAIRS_TRAIN, sample)
    pe = _read_sample(PAIRS_EVAL,  sample//4 if sample else None)
    r = pd.read_parquet(RNA_CLEAN, columns=["id","cp","texte"])
    s = pd.read_parquet(SIRENE_CLEAN, columns=["siret","cp","texte"])

    feats=["cos_sim","len_rna","len_sir","len_diff","same_cp"]
    dt, de = _build_feats(pt,r,s), _build_feats(pe,r,s)
    q = dt.id_rna.unique(); qt,qv = train_test_split(q,test_size=test_frac,random_state=42)
    dtr,dval = dt[dt.id_rna.isin(qt)], dt[dt.id_rna.isin(qv)]
    Xtr,ytr,gtr = _prep(dtr,feats); Xv,yv,gv = _prep(dval,feats)

    if gpu: params |= {"device_type":"gpu"}
    bst = lgb.train(params,
        lgb.Dataset(Xtr,ytr,group=gtr),
        valid_sets=[lgb.Dataset(Xv,yv,group=gv)],
        callbacks=[lgb.early_stopping(params["early_stopping_rounds"], verbose=True),
                   lgb.log_evaluation(50)])
    bst.save_model(MODEL_PATH)
    logging.info("💾 saved → %s", MODEL_PATH)

    Xe,ye,ge = _prep(de,feats); yp = bst.predict(Xe, num_iteration=bst.best_iteration)
    ap = average_precision_score(ye,yp) if ye.sum()>0 else float("nan")
    splits = np.split(yp,np.cumsum(ge)[:-1]); ysplit = np.split(ye,np.cumsum(ge)[:-1])
    ndcgs=[ndcg_score([yt],[yp_]) for yt,yp_ in zip(ysplit,splits) if yt.sum()>0 and len(yt)>1]
    ndcg = float(np.mean(ndcgs)) if ndcgs else float("nan")
    logging.info("✅ Eval  AP=%.4f | NDCG=%.4f", ap, ndcg)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--num_leaves", type=int, default=63)
    p.add_argument("--num_boost_round", type=int, default=500)
    p.add_argument("--early_stopping_rounds", type=int, default=50)
    p.add_argument("--test_size", type=float, default=0.1)
    p.add_argument("--sample_rows", type=int, default=None)
    p.add_argument("--gpu", action="store_true")
    a = p.parse_args(); logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    params = {"objective":"lambdarank","metric":"ndcg","boosting":"gbdt","learning_rate":a.learning_rate,
              "num_leaves":a.num_leaves,"verbose":-1,"num_boost_round":a.num_boost_round,
              "early_stopping_rounds":a.early_stopping_rounds,"seed":42}
    train(params,a.test_size,a.sample_rows,a.gpu)
