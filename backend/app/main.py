#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
FastAPI server for MatchMaker AI v2.1

Endpoints
---------
- GET  /health       → liveness probe
- POST /match        → top-k entreprises pertinentes
- GET  /metrics      → renvoie AP/DNCG sur un petit jeu test (optionnel)

Usage:
  uvicorn backend.app.main:app --reload
"""
import os
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import lightgbm as lgb

try:
    import faiss  # type: ignore
except ImportError:
    raise SystemExit("pip install faiss-cpu")

# ──────────────────── Config & Paths ────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("matchmaker")

DATA_DIR   = Path("data")
V2_DIR     = DATA_DIR / "v2"
PROC_DIR   = DATA_DIR / "processed"
MODEL_DIR  = DATA_DIR / "model"

FAISS_INDEX = V2_DIR / "sirene_faiss.index"
SIR_IDS_NPY = V2_DIR / "ids_sirene.npy"
SIR_META    = PROC_DIR / "sirene_clean.parquet"
LGBM_MODEL  = MODEL_DIR / "reranker_lgbm.txt"

# GPU if available
USE_CUDA = os.getenv("MM_USE_CUDA", "1") == "1" and torch.cuda.is_available()
DEVICE   = "cuda" if USE_CUDA else "cpu"

SBERT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_FAISS   = 300  # nombre de voisins bruts

# ───────────────────── Pydantic ─────────────────────
class MatchRequest(BaseModel):
    texte: str = Field(..., min_length=3, description="Besoin de l'association")
    cp:   str = Field(..., pattern=r"^\d{5}$", description="Code postal sur 5 chiffres")
    k:    Optional[int] = Field(5, ge=1, le=50, description="Nb résultats (défaut=5)")

class EntrepriseOut(BaseModel):
    enseigne: str
    commune: str
    score:   float

# ─────────────────── Lazy loaders ───────────────────
@lru_cache(maxsize=1)
def load_faiss():
    logger.info("🔄 Chargement FAISS index & ids")
    idx = faiss.read_index(str(FAISS_INDEX))
    ids = np.load(SIR_IDS_NPY, allow_pickle=True).astype(str)
    return idx, ids

@lru_cache(maxsize=1)
def load_sbert():
    logger.info(f"🔄 Chargement SBERT model ({DEVICE})")
    return SentenceTransformer(SBERT_MODEL, device=DEVICE)

@lru_cache(maxsize=1)
def load_lgbm():
    logger.info("🔄 Chargement modèle LightGBM")
    return lgb.Booster(model_file=str(LGBM_MODEL))

@lru_cache(maxsize=1)
def load_meta():
    logger.info("🔄 Chargement métadata entreprises")
    cols = ["siret","enseigne","commune","texte","cp"]
    df = pd.read_parquet(SIR_META, columns=cols)
    df = df.set_index("siret")
    return df

# ─────────────────── Feature engineering ───────────────────
@lru_cache(maxsize=32768)
def _token_len(text: str) -> int:
    return len(text.split())

def make_features(
    cos_sim: np.ndarray,
    query_txt: str,
    meta_slice: pd.DataFrame,
    cp_rna: str
) -> np.ndarray:
    len_r = _token_len(query_txt)
    len_s = meta_slice["texte"].map(_token_len).to_numpy()
    diff  = np.abs(len_r - len_s)
    same  = (meta_slice["cp"].values == cp_rna).astype(np.int8)
    X = np.vstack([cos_sim, np.full_like(cos_sim, len_r), len_s, diff, same]).T
    return X.astype(np.float32)

# ───────────────────── FastAPI ─────────────────────
app = FastAPI(title="MatchMaker AI", version="2.1.0")

@app.get("/health")
async def health():
    return {"status": "ok", "device": DEVICE}

@app.post("/match", response_model=List[EntrepriseOut])
async def match(req: MatchRequest):
    k = req.k or 5
    logger.info(f"🔍 Requête match: texte='{req.texte[:20]}…', cp={req.cp}, k={k}")

    # 1. Embedding
    sbert = load_sbert()
    q_emb = sbert.encode([req.texte], normalize_embeddings=True).astype("float32")

    # 2. FAISS search
    idx, ids = load_faiss()
    sims, idxs = idx.search(q_emb, TOP_FAISS)
    sims, idxs = sims[0], idxs[0]
    mask = idxs >= 0
    sims, idxs = sims[mask], idxs[mask]
    siret_cands = ids[idxs]

    if siret_cands.size == 0:
        return []

    # 3. Metadata slice
    meta = load_meta()
    # keep original order
    slice_df = meta.loc[siret_cands].reset_index()

    # 4. Features + reranking
    X = make_features(sims, req.texte, slice_df, req.cp)
    booster = load_lgbm()
    scores = booster.predict(X)
    order = np.argsort(-scores)

    # 5. Collect top-k
    topn = min(k, len(scores))
    out = []
    for i in order[:topn]:
        row = slice_df.iloc[i]
        out.append(EntrepriseOut(
            enseigne=row["enseigne"],
            commune = row["commune"],
            score   = float(scores[i])
        ))
    return out

@app.get("/metrics")
async def metrics():
    """
    (Optionnel) Retourne des métriques AP/NDCG sur un petit jeu de test
    fourni dans data/processed/pairs_eval.parquet
    """
    from sklearn.metrics import average_precision_score, ndcg_score
    pairs = pd.read_parquet(PROC_DIR/"pairs_eval.parquet")
    meta  = load_meta()
    sbert = load_sbert()
    idx, ids = load_faiss()
    booster  = load_lgbm()

    # sample un petit nombre pour aller vite
    sample = pairs.sample(n=10000, random_state=42)
    y_true = []
    y_pred = []
    for _, row in sample.iterrows():
        q_emb = sbert.encode([row["texte"]], normalize_embeddings=True).astype("float32")
        sims, idxs = idx.search(q_emb, TOP_FAISS)
        sims, idxs = sims[0], idxs[0]
        # repère le rang du bon siret
        ranks = {ids[s]: sims[i] for i, s in enumerate(idxs) if sims[i]>=0}
        score = ranks.get(row["siret"], 0.0)
        y_true.append(row["label"])
        y_pred.append(score)
    ap   = average_precision_score(y_true, y_pred)
    ndcg = ndcg_score([y_true], [y_pred])
    return {"AP": f"{ap:.4f}", "NDCG": f"{ndcg:.4f}"}
