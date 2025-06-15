Prochaines actions (étape 3)
Re-ranking : entraîner le modèle LightGBM sur ton pairs_train_* en ajoutant la similarité cosinus embeddings + features CP.

API FastAPI : endpoint /match qui :

encode la requête de l’asso,

interroge FAISS (top 300),

applique le reranker,

renvoie le top 5 JSON.

Ajout d’un petit script de test pour simuler la requête “ballons de basket…”.

Je reste dispo si tu veux :

un script de re-ranking prêt‐à‐l’emploi (phase 3-modèle),

l’app FastAPI céblée (phase 4),

ou des conseils pour pousser ça en prod (Docker, Cloud, etc.).






