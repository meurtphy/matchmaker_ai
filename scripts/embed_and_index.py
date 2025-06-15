#!/usr/bin/env python3
"""
embed_and_index.py  –  Étape 2 de la V2
========================================================
• Génère les embeddings MiniLM pour RNA & SIRENE (fichiers .parquet nettoyés)
• Construit un index FAISS (approximate nearest‑neighbors) sur SIRENE
• Sauvegarde :
    data/embeddings/<dataset>_vecs.f32     (float32 numpy memmap –  n × 384)
    data/embeddings/<dataset>_meta.parquet (id, texte, cp, ...)
    data/faiss/sirene.index                (FAISS – IndexHNSW32, normalised L2)

CLI  ------------------------------------------------------------------------
python -m scripts.embed_and_index --dataset sirene   # embeddings + index sirene
python -m scripts.embed_and_index --dataset rna      # uniquement embeddings

Options utiles :
    --batch 512            taille de batch pour l’encodage (déf. 256)
    --model all-MiniLM-L6-v2  (override du modèle sentence‑transformers)
    --force                ignore les fichiers déjà existants
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import List

import faiss  # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Config par défaut
# ---------------------------------------------------------------------------
DATA_PROCESSED = Path("data/processed")
EMB_DIR = Path("data/embeddings")
FAISS_DIR = Path("data/faiss")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims
EMB_DIMS = 384

EMB_DIR.mkdir(parents=True, exist_ok=True)
FAISS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(ds: str) -> pd.DataFrame:
    parquet = DATA_PROCESSED / f"{ds}_clean.parquet"
    if not parquet.exists():
        raise FileNotFoundError(parquet)
    usecols = [
        "id" if ds == "rna" else "siret",
        "texte",
        "cp",
        "commune",
    ]
    return pd.read_parquet(parquet, columns=usecols, engine="pyarrow")


def encode_batches(model: SentenceTransformer, texts: List[str], batch: int) -> np.ndarray:
    """Encode par lots (-> float32, L2 normalisé)"""
    vecs = np.empty((len(texts), EMB_DIMS), dtype="float32")
    for i in tqdm(range(0, len(texts), batch), desc="🔧 embeddings", unit_scale=batch):
        chunk = texts[i : i + batch]
        v = model.encode(chunk, batch_size=min(batch, 64), show_progress_bar=False, normalize_embeddings=True)
        vecs[i : i + len(chunk)] = v
    return vecs


def save_embeddings(ds: str, meta: pd.DataFrame, vecs: np.ndarray) -> None:
    vec_path = EMB_DIR / f"{ds}_vecs.f32"
    np.memmap(vec_path, dtype="float32", mode="w+", shape=vecs.shape)[:] = vecs
    meta_path = EMB_DIR / f"{ds}_meta.parquet"
    meta.to_parquet(meta_path, index=False, compression="snappy")
    print(f"✅ embeddings enregistrés  →  {vec_path}   {vecs.shape}")


def build_faiss_index(vecs: np.ndarray) -> faiss.Index:
    # HNSW approximate, inner‑product (car embeddings normalisés)
    index = faiss.IndexHNSWFlat(EMB_DIMS, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vecs)
    return index


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["rna", "sirene"], required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--force", action="store_true", help="écrase les fichiers existants")
    args = p.parse_args()

    ds = args.dataset
    vec_file = EMB_DIR / f"{ds}_vecs.f32"
    meta_file = EMB_DIR / f"{ds}_meta.parquet"

    if vec_file.exists() and not args.force:
        print(f"ℹ️  Embeddings déjà présents : {vec_file} – rien à faire ( --force pour écraser )")
        return

    # ---------------------------------------------------------------------
    print(f"📥 Chargement {ds}_clean.parquet …")
    df = load_dataset(ds)

    model = SentenceTransformer(args.model, device="cpu")  # GPU si dispo automatiquement

    # Encodage ----------------------------------------------------------------
    vecs = encode_batches(model, df["texte"].tolist(), args.batch)

    # Sauvegarde embeddings ----------------------------------------------------
    save_embeddings(ds, df, vecs)

    # Index only for SIRENE ----------------------------------------------------
    if ds == "sirene":
        print("⚙️  Construction FAISS (HNSW32)…")
        idx = build_faiss_index(vecs)
        faiss_path = FAISS_DIR / "sirene.index"
        faiss.write_index(idx, str(faiss_path))
        print(f"✅ Index FAISS écrit : {faiss_path}  (vecteurs : {idx.ntotal:,})")

    # Libère RAM
    del vecs, df, model
    gc.collect()


if __name__ == "__main__":
    main()
