scripts/train_reranker.py
But : entraîner le re-ranker V2
(LightGBM si dispo, sinon GradientBoosting) sur tous les pairs_train_*.parquet.

Features calculées

feature	explication
cos	similarité cosinus entre embeddings MiniLM (RNA vs Sirene)
exact_cp	1 si CP exact identique
region_dist	écart absolu sur les 2 premiers digits (≈ départements)
len_rna / len_sir / len_diff	longueurs des textes
jaccard	recouvrement lexical brut (token set)

Embeddings : chargés en zero-copy depuis
data/embeddings/{rna|sirene}_vecs.f32 (+ leur méta Parquet).

Usage minimal

bash
Copier le code
# entraînement complet
python -m scripts.train_reranker           # modèle → data/model/reranker_v2.joblib

# échantillonner 1.2 M lignes (plus rapide)
python -m scripts.train_reranker --sample 1200000 --seed 13 \
                                 --out data/model/reranker_v2_s1M.joblib
Sortie :

AUC validation affiché

Fichier .joblib contenant le modèle, la liste des features, la dimension d’embedding.

