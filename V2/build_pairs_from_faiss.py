#!/usr/bin/env python3
"""
build_pairs_from_faiss.py — v2.4
================================
Crée les couples (id_rna, siret, cos_sim, label) en streaming sur disque,
puis shuffle et split train/eval sans jamais charger tout en RAM.

Usage :
  python v2/build_pairs_from_faiss.py \
    --k 100 \
    --batch_size 2048 \
    --mapping data/processed/rna2siret.csv
"""
from __future__ import annotations
import argparse, os, math
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import faiss
except ImportError:
    raise SystemExit("pip install faiss-cpu")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
DATA_V2   = Path("data/v2")
DATA_P    = Path("data/processed")
RNA_EMB   = DATA_V2 / "rna_emb.npy"
RNA_IDS   = DATA_V2 / "ids_rna.npy"
SIR_IDS   = DATA_V2 / "ids_sirene.npy"
INDEX     = DATA_V2 / "sirene_faiss.index"
MAPPING   = DATA_P  / "rna2siret.csv"
OUT_ALL   = DATA_P  / "pairs_all.parquet"
OUT_TRAIN = DATA_P  / "pairs_train.parquet"
OUT_EVAL  = DATA_P  / "pairs_eval.parquet"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _np_load(path: Path):
    return np.load(path, allow_pickle=True)

def _open_emb(path: Path, ids_path: Path, dim: int = 384) -> np.ndarray:
    try:
        return np.load(path, mmap_mode="r")
    except ValueError:
        n = len(_np_load(ids_path))
        return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(k: int, batch_size: int, mapping_path: Path | None):
    os.makedirs(DATA_P, exist_ok=True)

    # load embeddings and IDs
    emb     = _open_emb(RNA_EMB, RNA_IDS)
    rna_ids = _np_load(RNA_IDS).astype(str)
    sir_ids = _np_load(SIR_IDS).astype(str)
    index   = faiss.read_index(str(INDEX))

    # load mapping of positives
    pos: Dict[tuple[str,str],int] = {}
    if mapping_path and mapping_path.exists():
        dfm = pd.read_csv(mapping_path, dtype=str)
        pos = {(r.id_rna, r.siret):1 for r in dfm.itertuples(index=False)}

    # Parquet writer for all pairs
    writer = None
    total = 0

    # streaming by batch
    for start in tqdm(range(0, len(rna_ids), batch_size), desc="BatchFAISS"):
        end = min(start + batch_size, len(rna_ids))
        sims, idxs = index.search(emb[start:end], k)

        # build table
        rows = []
        for i, ridx in enumerate(range(start, end)):
            rid = rna_ids[ridx]
            for j in range(k):
                sidx = idxs[i, j]
                if sidx < 0:
                    break
                sid = sir_ids[sidx]
                label = pos.get((rid, sid), 0)
                rows.append((rid, sid, float(sims[i, j]), label))

        df = pd.DataFrame(rows, columns=["id_rna","siret","cos_sim","label"])
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(OUT_ALL, table.schema, compression="snappy")
        writer.write_table(table)
        total += len(df)

    # close writer
    writer.close()
    print(f"✅ pairs écrites : {total:,} → {OUT_ALL}")

    # shuffle and split
    df_all = pd.read_parquet(OUT_ALL)
    df_all = df_all.sample(frac=1.0, random_state=42)
    n_eval = math.ceil(len(df_all) * 0.2)
    df_all.iloc[n_eval:].to_parquet(OUT_TRAIN, index=False)
    df_all.iloc[:n_eval].to_parquet(OUT_EVAL,  index=False)
    print(f"✅ train: {len(df_all)-n_eval:,} | eval: {n_eval:,}")

# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--k",           type=int, default=100, help="voisins FAISS")
    p.add_argument("--batch_size",  type=int, default=2048,help="taille batch")
    p.add_argument("--mapping",     type=Path,default=None, help="CSV positifs")
    args = p.parse_args()
    main(args.k, args.batch_size, args.mapping)
