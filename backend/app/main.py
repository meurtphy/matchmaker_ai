from __future__ import annotations

"""FastAPI server for MatchMaker AI v2.

Endpoints
---------
- GET  /health        → simple liveness probe
- POST /match         → returns top-k entreprises matching the association need

Request model (POST /match):
    {
      "texte": "ballons basket",
      "cp": "34090",
      "k": 5                # optional, default 5
    }

Response model:
    [
      {"enseigne": "Intersport Montpellier", "commune": "Montpellier", "score": 0.93},
      ...
    ]
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

try:
    import faiss  # type: ignore
except ImportError as e:
    raise SystemExit("faiss-cpu doit être installé (pip install faiss-cpu)") from e

# ---------------------------------------------------------------------------
# Constantes chemin et config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
V2_DIR = DATA_DIR / "v2"
PROC_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "model"

FAISS_INDEX = V2_DIR / "sirene_faiss.index"
SIR_IDS_NPY = V2_DIR / "ids_sirene.npy"
SIR_META = PROC_DIR / "sirene_clean.parquet"

LGBM_MODEL = MODEL_DIR / "reranker_lgbm.txt"

DEVICE = "cuda" if os.getenv("MM_USE_CUDA", "1") == "1" and torch.cuda.is_available() else "cpu"
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_FAISS = 300  # nb de voisins bruts avant rerank

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class MatchRequest(BaseModel):
    texte: str = Field(..., min_length=3, description="Besoin de l'association")
    cp: str = Field(..., regex=r"^\d{5}$", description="Code postal sur 5 chiffres")
    k: int = Field(5, ge=1, le=50, description="Nombre de résultats")


class EntrepriseOut(BaseModel):
    enseigne: str
    commune: str
    score: float


# ---------------------------------------------------------------------------
# Lazy global loaders (faiss, LightGBM, SBERT, metadata)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _load_faiss():
    index = faiss.read_index(str(FAISS_INDEX))
    ids = np.load(SIR_IDS_NPY)
    return index, ids


@lru_cache(maxsize=1)
def _load_sbert():
    return SentenceTransformer(SBERT_MODEL_NAME, device=DEVICE)


@lru_cache(maxsize=1)
def _load_lgbm():
    booster = lgb.Booster(model_file=str(LGBM_MODEL))
    return booster


@lru_cache(maxsize=1)
def _load_metadata():
    return pd.read_parquet(SIR_META, columns=["siret", "enseigne", "commune", "texte", "cp"])


# ---------------------------------------------------------------------------
# Feature engineering helpers (same as training)
# ---------------------------------------------------------------------------

def _token_len(txt: str) -> int:
    return len(str(txt).split())


def _build_features(query_emb: np.ndarray, cos_sim: np.ndarray, txt_rna: str, meta_slice: pd.DataFrame, cp_rna: str):
    len_rna = _token_len(txt_rna)
    len_sir = meta_slice["texte"].apply(_token_len).to_numpy()
    len_diff = np.abs(len_rna - len_sir)
    same_cp = (meta_slice["cp"].values == cp_rna).astype(int)
    features = np.vstack([cos_sim, np.full_like(cos_sim, len_rna), len_sir, len_diff, same_cp]).T
    return features.astype(np.float32)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="MatchMaker AI", version="2.0.0")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/match", response_model=List[EntrepriseOut])
async def match(req: MatchRequest):
    if not req.texte:
        raise HTTPException(status_code=400, detail="texte cannot be empty")

    # Encode la requête
    sbert = _load_sbert()
    query_emb = sbert.encode([req.texte], normalize_embeddings=True)

    # FAISS → candidats
    index, sir_ids = _load_faiss()
    sim, idx = index.search(query_emb.astype("float32"), TOP_FAISS)
    sim = sim[0]
    idx = idx[0]
    valid_mask = idx >= 0
    idx = idx[valid_mask]
    sim = sim[valid_mask]
    siret_candidates = sir_ids[idx]

    # Métadonnées des entreprises candidates
    meta = _load_metadata()
    meta_slice = meta.set_index("siret").loc[siret_candidates].reset_index()

    # Features re-ranker (align order)
    booster = _load_lgbm()
    feats = _build_features(query_emb[0], sim, req.texte, meta_slice, req.cp)
    scores = booster.predict(feats)

    # Tri final
    order = np.argsort(-scores)
    meta_slice = meta_slice.iloc[order].reset_index(drop=True)
    scores = scores[order]

    # Top-k
    results = []
    for i in range(min(req.k, len(meta_slice))):
        row = meta_slice.iloc[i]
        results.append(
            EntrepriseOut(
                enseigne=row["enseigne"],
                commune=row["commune"],
                score=float(scores[i]),
            )
        )
    return results
