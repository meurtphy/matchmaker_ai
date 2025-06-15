#!/usr/bin/env python3
"""
clean_sirene.py
===============

Nettoie le fichier StockEtablissement_utf8.csv en Parquet optimisé.
- Lecture par chunks pour limiter la RAM
- Filtre les établissements actifs (etat == 'A')
- Construit une colonne 'enseigne' en coalesçant les champs disponibles
- Génère une colonne 'texte' (enseigne + NAF) prête pour embedding

Entrées
------
RAW_FILE (CSV)  : data/raw/StockEtablissement_utf8.csv

Sorties
-------
OUT_FILE (Parquet) : data/processed/sirene_clean.parquet

Usage
-----
$ python scripts/clean_sirene.py [--in RAW] [--out OUT] [--chunk N] [--verbose]
"""
from __future__ import annotations
import argparse
import logging
import os
import unicodedata
from pathlib import Path
from typing import Dict, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# --------------------------------------------------------------------------
# Constantes par défaut
# --------------------------------------------------------------------------
DEFAULT_RAW   = Path("data/raw/StockEtablissement_utf8.csv")
DEFAULT_OUT   = Path("data/processed/sirene_clean.parquet")
DEFAULT_CHUNK = 500_000
VALID_ETAT    = {"A"}  # seules les lignes actives

# mapping colonnes CSV → noms internes
BASE_COLS: Dict[str, str] = {
    "siret": "siret",
    "activitePrincipaleEtablissement": "naf",
    "etatAdministratifEtablissement": "etat",
    "codePostalEtablissement": "cp",
    "libelleCommuneEtablissement": "commune",
}
# champs potentiels pour 'enseigne'
ENSEIGNE_RAW: List[str] = [
    "denominationUniteLegale",
    "enseigne1Etablissement",
    "enseigne2Etablissement",
    "enseigne3Etablissement",
]
# colonnes finales au format parquet
OUTPUT_COLS = ["siret", "enseigne", "naf", "etat", "cp", "commune", "texte"]

# --------------------------------------------------------------------------
# Fonctions utilitaires
# --------------------------------------------------------------------------
def strip_acc(txt: str) -> str:
    """Normalise, retire les accents et met en minuscules."""
    if not isinstance(txt, str):
        return ""
    nfkd = unicodedata.normalize("NFKD", txt)
    no_acc = "".join(c for c in nfkd if not unicodedata.combining(c))
    return " ".join(no_acc.split()).lower()

# --------------------------------------------------------------------------
# Pipeline de nettoyage
# --------------------------------------------------------------------------
def clean_sirene(raw_file: Path, out_file: Path, chunk_size: int) -> None:
    # configuration logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s", level=level
    )
    logger = logging.getLogger(__name__)

    if not raw_file.exists():
        logger.error(f"Fichier introuvable : {raw_file}")
        raise FileNotFoundError(raw_file)

    os.makedirs(out_file.parent, exist_ok=True)
    # détection des colonnes enseigne présentes
    first_cols = pd.read_csv(raw_file, sep=",", nrows=0).columns.tolist()
    ens_in_file = [c for c in ENSEIGNE_RAW if c in first_cols]
    if not ens_in_file:
        logger.warning("Aucun champ d'enseigne détecté, la colonne 'enseigne' sera vide.")
    usecols = list(BASE_COLS.keys()) + ens_in_file

    writer: pq.ParquetWriter | None = None
    total_written = 0

    logger.info(
        f"Lecture de {raw_file.name} par chunks de {chunk_size:,} lignes (colonnes : {usecols})"
    )
    for chunk in tqdm(
        pd.read_csv(
            raw_file,
            sep=",",
            dtype=str,
            usecols=usecols,
            chunksize=chunk_size,
            low_memory=False,
        ),
        desc="chunks",
        unit="chunk",
    ):
        # filtre actifs
        chunk = chunk[chunk.get("etatAdministratifEtablissement", "").isin(VALID_ETAT)]
        if chunk.empty:
            continue

        # coalesce des champs enseigne
        if ens_in_file:
            enseigne_series = (
                chunk[ens_in_file]
                .bfill(axis=1)
                .iloc[:, 0]
                .fillna("")
            )
        else:
            enseigne_series = pd.Series("", index=chunk.index)
        chunk["enseigne"] = enseigne_series

        # renommage des colonnes de base
        chunk = chunk.rename(columns=BASE_COLS)

        # normalisation et nettoyage
        chunk["cp"] = chunk["cp"].fillna("").astype(str).str.zfill(5)
        chunk["enseigne"] = chunk["enseigne"].apply(strip_acc)
        chunk["naf"] = chunk["naf"].fillna("").astype(str)
        chunk["texte"] = (chunk["enseigne"] + " " + chunk["naf"]).str.strip()

        # préparation de l'écriture
        missing = set(OUTPUT_COLS) - set(chunk.columns)
        for col in missing:
            chunk[col] = ""
        table = pa.Table.from_pandas(chunk[OUTPUT_COLS], preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                out_file, table.schema, compression="snappy", use_dictionary=True
            )
        writer.write_table(table)
        total_written += len(chunk)

    if writer:
        writer.close()
        logger.info(f"✅ Nettoyage terminé : {total_written:,} lignes écrites → {out_file}")
    else:
        logger.warning("Aucun établissement actif n'a été écrit. Vérifier le filtre d'état.")

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Nettoie SIRENE : CSV → Parquet optimisé"
    )
    parser.add_argument(
        "--in", dest="raw", type=Path, default=DEFAULT_RAW,
        help="Fichier CSV source (défaut: %(default)s)"
    )
    parser.add_argument(
        "--out", dest="out", type=Path, default=DEFAULT_OUT,
        help="Parquet de sortie (défaut: %(default)s)"
    )
    parser.add_argument(
        "--chunk", dest="chunk", type=int, default=DEFAULT_CHUNK,
        help="Taille du chunk en nombre de lignes"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Mode debug (logging=DEBUG)"
    )
    args = parser.parse_args()

    clean_sirene(args.raw, args.out, args.chunk)
