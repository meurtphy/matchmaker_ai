# scripts/clean_rna.py
import os
import unicodedata
import pandas as pd

IN_FILE = "data/interim/rna_full.csv"
OUT_FILE = "data/processed/rna_clean.parquet"

COLS_MAP = {
    "id": "id",
    "titre": "titre",
    "objet": "objet",
    "adrs_codepostal": "cp",
    "adrs_libcommune": "commune",
    "siteweb": "siteweb",
    "position": "etat"
}

def strip_accents(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split()).lower()

def clean_rna():
    print("🔄 Lecture CSV brut RNA…")
    df = pd.read_csv(IN_FILE, sep=",", dtype=str, usecols=lambda c: c in COLS_MAP)
    df = df.rename(columns=COLS_MAP)

    df = df[df["etat"] == "A"].copy()

    for col in ["titre", "objet"]:
        df[col] = df[col].fillna("").apply(strip_accents)

    df["cp"] = df["cp"].fillna("").astype(str).str.zfill(5)
    df["texte"] = (df["titre"] + " " + df["objet"]).str.strip()

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_parquet(OUT_FILE, engine="pyarrow", compression="snappy", index=False)
    print(f"✅ RNA nettoyé : {len(df):,} lignes  → {OUT_FILE}")

if __name__ == "__main__":
    clean_rna()
