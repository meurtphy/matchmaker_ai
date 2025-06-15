#!/usr/bin/env python3
"""
build_pairs_from_faiss.py  ─ v2.1
===========================
Génère les couples **(association, entreprise)** pour entraîner / évaluer le
reranker LightGBM.

✔️ Points renforcés
-------------------
* **Chargement robuste** des .npy (pickle/objet ou memmap brut) → `allow_pickle=True`.
* **Progress‑bar claire** : nb batches + pairs créés.
* **Écriture incrémentale** : stream des lignes vers Parquet pour ne pas garder
  tout le DataFrame en RAM sur très gros jeux (option `--write_chunk`).
* **Option `--mapping`** : CSV positif facultatif (`id_rna,siret`).

Sorties
-------
* `data/processed/pairs_train.parquet`
* `data/processed/pairs_eval.parquet`
"""
from __future__ import annotations

import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError as e:
    raise SystemExit("faiss-cpu doit être installé (pip install faiss-cpu)") from e

# ─────────────────────────────────────────────────────────────────────────────
# Chemins
# ─────────────────────────────────────────────────────────────────────────────
DATA_V2   = Path("data/v2")
DATA_PROC = Path("data/processed")

RNA_EMB   = DATA_V2 / "rna_emb.npy"
RNA_IDS   = DATA_V2 / "ids_rna.npy"
SIR_IDS   = DATA_V2 / "ids_sirene.npy"
SIR_INDEX = DATA_V2 / "sirene_faiss.index"

OUT_TRAIN = DATA_PROC / "pairs_train.parquet"
OUT_EVAL  = DATA_PROC / "pairs_eval.parquet"

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _np_load_any(path: Path):
    """Charge .npy, qu'il soit standard ou object/pickled."""
    return np.load(path, allow_pickle=True)


def _open_emb(path: Path, ids_path: Path, dim: int = 384) -> np.ndarray:
    """Ouvre le fichier d'embeddings, normal ou memmap brut."""
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:
        print(f"⚠️  {path.name} non‑standard → memmap brut")
        n_vec = len(_np_load_any(ids_path))
        return np.memmap(path, dtype="float32", mode="r", shape=(n_vec, dim))


def _load_mapping(mapping_path: Path | None) -> Dict[Tuple[str, str], int]:
    pos: Dict[Tuple[str, str], int] = {}
    if mapping_path and mapping_path.exists():
        df = pd.read_csv(mapping_path, dtype=str, usecols=["id_rna", "siret"])
        pos = {(r.id_rna, r.siret): 1 for r in df.itertuples(index=False)}
    return pos

# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def build_pairs(k: int, batch_size: int, write_chunk: int, mapping: Path | None):
    os.makedirs(DATA_PROC, exist_ok=True)

    rna_emb = _open_emb(RNA_EMB, RNA_IDS)
    rna_ids = _np_load_any(RNA_IDS).astype(str)
    sir_ids = _np_load_any(SIR_IDS).astype(str)
    index   = faiss.read_index(str(SIR_INDEX))

    pos_pairs = _load_mapping(mapping)

    n_rna   = rna_emb.shape[0]
    rows: List[Tuple[str, str, float, int]] = []
    written = 0

    for start in tqdm(range(0, n_rna, batch_size), desc="RNA", unit="batch"):
        end = min(start + batch_size, n_rna)
        sims, idxs = index.search(rna_emb[start:end], k)
        for i, rna_idx in enumerate(range(start, end)):
            rid = rna_ids[rna_idx]
            for j in range(k):
                sidx = idxs[i, j]
                if sidx < 0:
                    continue
                sid  = sir_ids[sidx]
                label = pos_pairs.get((rid, sid), 0)
                rows.append((rid, sid, float(sims[i, j]), label))

        # écriture par chunk pour réduire RAM
        if len(rows) >= write_chunk:
            _flush(rows, OUT_TRAIN)  # temporaire, tout ira dans un Parquet unique
            written += len(rows)
            rows.clear()

    # flush final
    if rows:
        _flush(rows, OUT_TRAIN)
        written += len(rows)
        rows.clear()

    print(f"➡️  {written:,} pairs écrites temporairement → concat…")

    # concat et split train/eval
    df = pd.read_parquet(OUT_TRAIN)
    df = df.sample(frac=1.0, random_state=42)
    n_eval = math.ceil(len(df) * 0.2)
    df_eval  = df.iloc[:n_eval].reset_index(drop=True)
    df_train = df.iloc[n_eval:].reset_index(drop=True)
    df_train.to_parquet(OUT_TRAIN, index=False)
    df_eval.to_parquet(OUT_EVAL, index=False)
    print(f"✅ Pairs générés : {len(df_train):,} train | {len(df_eval):,} eval")


def _flush(buf: List[Tuple[str, str, float, int]], dest: Path):
    """Ajoute un buffer de lignes dans un Parquet (append)."""
    df = pd.DataFrame(buf, columns=["id_rna", "siret", "cos_sim", "label"])
    if dest.exists():
        df_existing = pd.read_parquet(dest)
        df_concat = pd.concat([df_existing, df], ignore_index=True)
        df_concat.to_parquet(dest, index=False)
    else:
        df.to_parquet(dest, index=False)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, default=100, help="Voisins FAISS par requête")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--write_chunk", type=int, default=1_000_000, help="Flush vers Parquet tous les N pairs")
    p.add_argument("--mapping", type=Path, default=None, help="CSV id_rna,siret pour labels 1")
    args = p.parse_args()

    build_pairs(args.k, args.batch_size, args.write_chunk, args.mapping)
