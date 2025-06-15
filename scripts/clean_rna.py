"""
Nettoie le CSV/Parquet RNA  ➜  data/processed/rna_clean.parquet
"""

import os, unicodedata
from pathlib import Path
import pandas as pd

# --- Localisation robuste ---------------------------------------------
ROOT = Path(__file__).resolve().parents[1]          # <repo_root>
DATA  = ROOT / "data"
IN_CSV = DATA / "interim" / "rna_full.csv"
IN_PARQ = DATA / "interim" / "rna_full.parquet"     # au cas où
OUT_FILE = DATA / "processed" / "rna_clean.parquet"
COLS_MAP = {
    "id": "id",
    "titre": "titre",
    "objet": "objet",
    "adrs_codepostal": "cp",
    "adrs_libcommune": "commune",
    "siteweb": "siteweb",
}

# -----------------------------------------------------------------------

def strip_accents(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    return " ".join(txt.split()).lower()

def clean_rna() -> None:
    # --- lecture -------------------------------------------------------
    print("🔄 Lecture brut RNA…")
    if IN_PARQ.exists():
        df = pd.read_parquet(IN_PARQ, engine="pyarrow")
    elif IN_CSV.exists():
        df = pd.read_csv(
            IN_CSV,
            sep=",",
            dtype=str,
            usecols=lambda c: c in COLS_MAP or c == "position",
            low_memory=False,
        )
    else:
        raise FileNotFoundError("Aucun RNA brut trouvé dans data/interim/")

    # --- filtrage ------------------------------------------------------
    df = df[df["position"] == "A"].copy()

    # --- sélection / renommage ----------------------------------------
    df = df[list(COLS_MAP.keys())].rename(columns=COLS_MAP)

    # --- nettoyage texte ----------------------------------------------
    for col in ("titre", "objet"):
        df[col] = df[col].fillna("").apply(strip_accents)

    # --- code postal normalisé ----------------------------------------
    df["cp"] = df["cp"].astype(str).str.zfill(5)

    # --- colonne texte concaténée -------------------------------------
    df["texte"] = (df["titre"] + " " + df["objet"]).str.strip()

    # --- sauvegarde ----------------------------------------------------
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, engine="pyarrow", compression="snappy", index=False)
    print(f"✅ RNA nettoyé : {len(df):,} lignes → {OUT_FILE.relative_to(ROOT)}")

if __name__ == "__main__":
    clean_rna()
