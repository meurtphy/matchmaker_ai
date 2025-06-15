#!/usr/bin/env python3
"""
generate_mapping.py — v2.2
--------------------------
Génère data/processed/rna2siret.csv en combinant :
1. Recherche sémantique (FAISS) par batch.
2. Filtre mot-clés partagés.
Options :
  --k               top-k voisins FAISS
  --score_th        seuil cos_sim initial
  --faiss_batch     taille de batch pour index.search
  --max_per_assoc   max matches par association
  --relax           si aucun match strict, retente à score_th/2
"""
from __future__ import annotations
import argparse, os, re
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError:
    raise SystemExit("faiss-cpu doit être installé (pip install faiss-cpu)")

# Chemins
DATA_V2   = Path("data/v2")
DATA_P    = Path("data/processed")
RNA_EMB   = DATA_V2 / "rna_emb.npy"
RNA_IDS   = DATA_V2 / "ids_rna.npy"
SIR_IDS   = DATA_V2 / "ids_sirene.npy"
INDEX     = DATA_V2 / "sirene_faiss.index"
RNA_PQ    = DATA_P / "rna_clean.parquet"
SIR_PQ    = DATA_P / "sirene_clean.parquet"
OUT_CSV   = DATA_P / "rna2siret.csv"

# Stop-words FR+EN basique
STOPWORDS: Set[str] = {
    "de","la","le","les","des","du","un","une","et","à","au","aux","en","d","l","s","pour",
    "the","and","for","with","from","into","onto","over","under","a","an"
}
TOKEN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9']+")

def _tokset(text: str) -> Set[str]:
    return {
        w.lower()
        for w in TOKEN_RE.findall(text)
        if len(w) > 2 and w.lower() not in STOPWORDS
    }

def _open_emb(path: Path, n: int, dim: int = 384) -> np.ndarray:
    """Charge .npy standard ou memmap brut."""
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:
        return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))

def main(
    k: int,
    score_th: float,
    faiss_batch: int,
    max_per_assoc: int,
    relax: bool,
):
    os.makedirs(DATA_P, exist_ok=True)

    # charge ids + embeddings + index
    rna_ids = np.load(RNA_IDS, allow_pickle=True).astype(str)
    sir_ids = np.load(SIR_IDS, allow_pickle=True).astype(str)
    emb     = _open_emb(RNA_EMB, len(rna_ids))
    index   = faiss.read_index(str(INDEX))

    # charge textes
    rna_txt = pd.read_parquet(RNA_PQ, columns=["id","texte"]) \
               .set_index("id")["texte"].fillna("").astype(str).to_dict()
    sir_txt = pd.read_parquet(SIR_PQ, columns=["siret","texte"]) \
               .set_index("siret")["texte"].fillna("").astype(str).to_dict()

    pairs: List[tuple[str,str]] = []
    total_asso = len(rna_ids)

    # boucle par batch pour FAISS
    for start in tqdm(range(0, total_asso, faiss_batch), desc="FAISS batch"):
        end = min(start + faiss_batch, total_asso)
        sims, idxs = index.search(emb[start:end], k)
        for i, rid in enumerate(rna_ids[start:end]):
            tokens_r = _tokset(rna_txt.get(rid, ""))
            found = 0
            # premiers seuil strict
            for sim, sidx in zip(sims[i], idxs[i]):
                if sidx < 0 or sim < score_th:
                    continue
                sid = sir_ids[sidx]
                if tokens_r & _tokset(sir_txt.get(sid, "")):
                    pairs.append((rid, sid))
                    found += 1
                    if found >= max_per_assoc:
                        break
            # si aucun trouvé et relax, retente à half threshold
            if found == 0 and relax:
                for sim, sidx in zip(sims[i], idxs[i]):
                    if sidx < 0 or sim < (score_th / 2):
                        continue
                    sid = sir_ids[sidx]
                    pairs.append((rid, sid))
                    break

    # écrit CSV final
    pd.DataFrame(pairs, columns=["id_rna","siret"]).to_csv(OUT_CSV, index=False)
    print(f"✅ mapping écrit : {len(pairs):,} lignes → {OUT_CSV}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=5, help="top-k voisins FAISS")
    p.add_argument("--score_th", type=float, default=0.50, help="seuil cosine initial")
    p.add_argument("--faiss_batch", type=int, default=4096, help="taille batch FAISS")
    p.add_argument("--max_per_assoc", type=int, default=1, help="max matches par asso")
    p.add_argument("--relax", action="store_true", help="seuil secondaire à score_th/2")
    args = p.parse_args()

    main(
        args.k,
        args.score_th,
        args.faiss_batch,
        args.max_per_assoc,
        args.relax,
    )
