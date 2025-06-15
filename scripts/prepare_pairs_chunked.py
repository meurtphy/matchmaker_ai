"""
Génère les paires RNA ↔ SIRENE par tranches de CP
Usage :  python -m scripts.prepare_pairs_chunked 0   # 0, 1, 2 ou 3
"""

import sys, os, gc, re, random
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ---------- FICHIERS ----------
DATA_DIR      = Path("data/processed")
RNA_FILE      = DATA_DIR / "rna_clean.parquet"
SIRENE_FILE   = DATA_DIR / "sirene_clean.parquet"
OUT_TRAIN_FMT = DATA_DIR / "pairs_train_{chunk}.parquet"
OUT_EVAL_FMT  = DATA_DIR / "pairs_eval_{chunk}.parquet"

# ---------- PARAMS ----------
CHUNK_MAP = {
    0: ("0", "1"),
    1: ("2", "3"),
    2: ("4", "5"),
    3: ("6", "7", "8", "9"),
}
NEG_PER_POS = 3   # nombre de paires négatives par positive
TEST_RATIO  = .2  # 20 % pour l’éval

# ---------- OUTILS ----------
def load_subset(prefixes):
    """Charge uniquement les lignes dont le CP commence par prefixes"""
    # Colonnes minimales
    rna_cols = ["id", "texte", "cp"]
    sir_cols = ["siret", "texte", "cp"]

    print("  ↳ charge RNA…")
    rna = pd.read_parquet(RNA_FILE, columns=rna_cols)
    rna = rna[rna["cp"].str.startswith(prefixes)].reset_index(drop=True)

    print("  ↳ charge SIRENE…")
    sir = pd.read_parquet(SIRENE_FILE, columns=sir_cols)
    sir = sir[sir["cp"].str.startswith(prefixes)].reset_index(drop=True)
    return rna, sir


def jaccard(a: str, b: str) -> float:
    sa, sb = set(a.split()), set(b.split())
    inter   = len(sa & sb)
    return inter / (len(sa) + len(sb) - inter or 1)


def build_pairs(chunk_id: int):
    prefixes = CHUNK_MAP[chunk_id]
    print(f"📦 Tranche {chunk_id} → CP commençant par {prefixes}")

    rna, sir = load_subset(prefixes)

    # ---------------- PAIRES POSITIVES ----------------
    # On suppose que le label positif = même CP ET forte similarité Jaccard
    pos_rows = []
    print("⛳ positifs:")
    for _, r in tqdm(rna.iterrows(), total=len(rna)):
        same_cp = sir[sir["cp"] == r["cp"]]
        if same_cp.empty:
            continue
        # Meilleur match par jaccard
        sims = same_cp["texte"].apply(lambda t: jaccard(r["texte"], t))
        best_idx = sims.idxmax()
        pos_rows.append(
            {
                "id_rna": r["id"],
                "id_sir": same_cp.at[best_idx, "siret"],
                "texte_rna": r["texte"],
                "texte_sir": same_cp.at[best_idx, "texte"],
                "label": 1,
            }
        )

    # ---------------- PAIRES NÉGATIVES ----------------
    print("🚫 négatifs:")
    neg_rows = []
    random.seed(0)
    for row in tqdm(pos_rows):
        for _ in range(NEG_PER_POS):
            rand_sir = sir.sample(1).iloc[0]
            neg_rows.append(
                {
                    "id_rna": row["id_rna"],
                    "id_sir": rand_sir["siret"],
                    "texte_rna": row["texte_rna"],
                    "texte_sir": rand_sir["texte"],
                    "label": 0,
                }
            )

    pairs = pd.DataFrame(pos_rows + neg_rows)
    print(f"Pairs total : {len(pairs):,} (positifs {len(pos_rows)} / négatifs {len(neg_rows)})")

    # ---------------- SPLIT TRAIN / EVAL ----------------
    pairs = pairs.sample(frac=1, random_state=0).reset_index(drop=True)
    n_eval = int(len(pairs) * TEST_RATIO)
    eval_df = pairs.iloc[:n_eval]
    train_df = pairs.iloc[n_eval:]

    # ---------------- SAVE ----------------
    train_df.to_parquet(OUT_TRAIN_FMT.format(chunk=chunk_id), index=False, compression="snappy")
    eval_df.to_parquet(OUT_EVAL_FMT.format(chunk=chunk_id), index=False, compression="snappy")
    print(f"✅ train : {len(train_df):,} → {OUT_TRAIN_FMT.format(chunk=chunk_id)}")
    print(f"✅ eval  : {len(eval_df):,} → {OUT_EVAL_FMT.format(chunk=chunk_id)}")

    # Libère la RAM
    del rna, sir, pairs, train_df, eval_df
    gc.collect()


# ---------- MAIN ----------
if __name__ == "__main__":
    if len(sys.argv) != 2 or int(sys.argv[1]) not in CHUNK_MAP:
        print("Usage :  python -m scripts.prepare_pairs_chunked <0|1|2|3>")
        sys.exit(1)
    build_pairs(int(sys.argv[1]))
