#!/usr/bin/env python3
"""
embed_and_index.py (v2)
======================
Génère les embeddings SBERT (MiniLM) pour les datasets RNA & SIRENE puis
construit un index FAISS (Flat ou compressé) **sans dépasser la RAM**.

### Nouveautés v2.1
- **--faiss_factory** : choisir la chaîne `index_factory` FAISS (ex: `Flat`,
  `IVF4096,PQ64`, `HNSW32`) – défaut: `Flat`.
- **auto‑switch** : si `faiss_factory == Flat` mais le jeu > 2 M lignes, un
  avertissement s'affiche ; l'utilisateur peut forcer un type compressé pour
  éviter le `MemoryError`.
- Training automatique si l'index le nécessite (IVF*, PQ*, HNSW…).

### Exemple pour gros jeu (> 10 M)
```bash
python v2/embed_and_index.py --datasets sirene --faiss_factory "IVF16384,PQ64" --train_size 200000
```

Sorties inchangées :
```
data/v2/{dataset}_emb.npy
           ids_{dataset}.npy
           {dataset}_faiss.index
```
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError as e:
    raise SystemExit("faiss-cpu doit être installé (pip install faiss-cpu)") from e

# ──────────────────────────────────────────────────────────────────────────────
# Constantes chemins
# ──────────────────────────────────────────────────────────────────────────────
DATA_PROCESSED = Path("data/processed")
DATA_V2 = Path("data/v2")
RNA_FILE = DATA_PROCESSED / "rna_clean.parquet"
SIRENE_FILE = DATA_PROCESSED / "sirene_clean.parquet"

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA demandé mais indisponible, bascule sur CPU.")
        return "cpu"
    return name


def _ensure_dirs() -> None:
    DATA_V2.mkdir(parents=True, exist_ok=True)


def _load_df(dataset: str) -> pd.DataFrame:
    if dataset == "rna":
        df = pd.read_parquet(RNA_FILE)
        id_col = "id"
    else:
        df = pd.read_parquet(SIRENE_FILE)
        id_col = "siret"
    required_cols = {id_col, "texte"}
    if not required_cols.issubset(df.columns):
        raise KeyError(f"Colonnes manquantes dans {dataset}: {required_cols - set(df.columns)}")
    return df[[id_col, "texte"]].reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────────
# Embedding + FAISS
# ──────────────────────────────────────────────────────────────────────────────

def _train_index_if_needed(index, emb: np.ndarray, train_size: int):
    if index.is_trained:
        return index
    n = emb.shape[0]
    size = min(train_size, n)
    sample_idx = random.sample(range(n), size)
    train_vecs = emb[sample_idx]
    print(f"🔧 FAISS training on {size:,} samples…")
    index.train(train_vecs)
    return index


def _compute_and_index(
    df: pd.DataFrame,
    dataset: str,
    model: SentenceTransformer,
    batch_size: int,
    device: str,
    faiss_factory: str,
    train_size: int,
) -> None:
    ids = df.iloc[:, 0].astype(str).values
    texts: List[str] = df["texte"].tolist()
    dim = model.get_sentence_embedding_dimension()
    n = len(df)

    emb_path = DATA_V2 / f"{dataset}_emb.npy"
    ids_path = DATA_V2 / f"ids_{dataset}.npy"
    index_path = DATA_V2 / f"{dataset}_faiss.index"

    if all(p.exists() for p in (emb_path, ids_path, index_path)):
        print(f"✔️  {dataset}: déjà généré, rien à faire.")
        return

    # warn for Flat on big data
    if faiss_factory.lower() == "flat" and n > 2_000_000:
        print(
            f"⚠️  {dataset} contient {n:,} vecteurs -> 'Flat' index risque d'exploser la RAM."
            " Envisagez --faiss_factory 'IVF16384,PQ64' (ou similaire)."
        )

    print(f"→ {dataset}: {n:,} lignes → batch {batch_size} sur {device}")

    mmap = np.memmap(emb_path, dtype="float32", mode="w+", shape=(n, dim))

    # Index creation via factory
    if faiss_factory.lower() == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.index_factory(dim, faiss_factory, faiss.METRIC_INNER_PRODUCT)

    # Encoder et alimenter index
    for start in tqdm(range(0, n, batch_size), desc=f"{dataset}  batches", unit="batch"):
        end = min(start + batch_size, n)
        batch_texts = texts[start:end]
        with torch.inference_mode():
            emb = model.encode(
                batch_texts,
                batch_size=batch_size,
                device=device,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
        mmap[start:end] = emb

        # train index lazily if needed
        if (not index.is_trained) and faiss_factory.lower() != "flat":
            # accumulate until we reach train_size
            pass  # training handled after full embedding loop
        else:
            index.add(emb)

    mmap.flush()
    np.save(ids_path, ids)

    # Train & add for compressing indexes needing training
    if not index.is_trained:
        mmap_read = np.memmap(emb_path, dtype="float32", mode="r", shape=(n, dim))
        index = _train_index_if_needed(index, mmap_read, train_size)
        print("🚚 Adding all embeddings to trained index…")
        for start in tqdm(range(0, n, batch_size), desc="add2idx", unit="batch"):
            end = min(start + batch_size, n)
            index.add(mmap_read[start:end])

    faiss.write_index(index, str(index_path))
    print(f"✅ {dataset}: embeddings → {emb_path.name} | ids → {ids_path.name} | index → {index_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Main CLI
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed & index RNA / SIRENE avec MiniLM+FAISS")
    parser.add_argument("--datasets", nargs="+", default=["rna", "sirene"], choices=["rna", "sirene"])
    parser.add_argument("--batch_size", type=int, default=512, help="Taille de batch SBERT")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device PyTorch")
    parser.add_argument(
        "--faiss_factory",
        default="Flat",
        help="Chaîne factory FAISS (Flat, IVF4096,PQ64, HNSW32 …)",
    )
    parser.add_argument("--train_size", type=int, default=100_000, help="Nb vecteurs pour entraîner l'index s'il le faut")
    args = parser.parse_args()

    _ensure_dirs()
    device = _pick_device(args.device)
    print(f"🖥️  Device sélectionné: {device}\n")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    for ds in args.datasets:
        df = _load_df(ds)
        _compute_and_index(
            df,
            ds,
            model,
            args.batch_size,
            device,
            args.faiss_factory,
            args.train_size,
        )

if __name__ == "__main__":
    main()
