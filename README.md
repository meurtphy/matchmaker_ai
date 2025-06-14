MatchMaker AI — v2 Roadmap (Ordre 66)

✨ Vision & But initial

MatchMaker AI connecte :

Associations (besoins, projets, événements)

Entreprises (dons matériels, compétences, mécénat)

Exemple : un club basket écrit : « On cherche des ballons de basket – besoin urgent – 34090 ».

L’API renvoie :

[
  {"enseigne":"Intersport Montpellier","commune":"34090","score":0.93},
  {"enseigne":"Décathlon Pro","commune":"34740","score":0.91},
  {"enseigne":"Recyc'Sport","commune":"34000","score":0.87}
]

Aucune entreprise n’a explicitement écrit « je donne des ballons », mais le moteur sémantique comprend l’affinité.

📜 Historique rapide

Étape

 Contenu

 Statut

v1

 Nettoyage CSV → Parquet, Jaccard/TF‑IDF, LightGBM de base

✅ Fonctionnel (RAM ~8 Go)

Ordre 66

 Prouve qu’on peut matcher sans gros moyens

✅

v2 (en cours)

 Embeddings MiniLM + index FAISS, re‑ranking LightGBM, API FastAPI

🚧

v3 (plus tard)

 Cross‑encoder, feedback utilisateur, fine‑tuning

🔮

🏗️ Architecture v2 proposée

matchmaker_ai/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI
│   │   ├── api/v1/endpoints/  # /match, /health
│   │   ├── services/          # match_service.py
│   │   ├── models/            # Pydantic schemas
│   │   ├── core/              # config, logging
│   │   └── db/                # loaders (FAISS, parquet)
│   └── tests/
├── data/
│   ├── raw/       # zips téléchargés
│   ├── interim/   # full.parquet
│   ├── processed/ # *_clean.parquet
│   ├── indexes/   # faiss index & npy
│   └── samples/   # petits jeux jouet
├── scripts/
│   ├── embed_and_index.py  # → FAISS
│   ├── train_reranker.py   # LightGBM sur pairs*
│   └── demo_query.py       # requête locale
├── templates/              # html Jinja2 (optionnel)
└── static/                 # css, js, images

🔬 Pipeline v2 détaillé

Phase

Script

 Sortie

 Notes

➡️ 0. Pré‑requis

pip install -r requirements.txt

–

faiss‑cpu, sentence‑transformers, lightgbm, fastapi, uvicorn

1. Embeddings

scripts/embed_and_index.py

data/indexes/{rna,sirene}.index + .npy

batch 50 k, MiniLM (384 d)

2. Rerank

scripts/train_reranker.py

data/model/reranker.joblib

uses cosine, len_diff, CP_proximity

3. API

uvicorn backend.app.main:app --reload

Swagger at /docs

endpoint POST /match

🚀 Run de test (toy)

python scripts/demo_query.py \
  --texte "ballons basket" \
  --cp 34090 \
  --top 5

Retour :

[
  {"enseigne":"Intersport Montpellier","distance":0.07},
  {"enseigne":"Décathlon Pro","distance":0.09},
  ...
]

📈 Roadmap condensée



Feedback bienvenu !



📁 Fichier créé
Nom du script	Chemin	Rôle
embed_and_index.py	matchmaker_ai/scripts/embed_and_index.py	Embedding + indexation FAISS des bases RNA et SIRENE

🧠 Comment ça marche ? Détail du code
1. Chargement par morceaux (batches)
Le code lit les fichiers rna_clean.parquet et sirene_clean.parquet par tranche (ex: 5000 lignes) pour éviter de charger toute la RAM.

2. Embeddings
On utilise MiniLM (via sentence-transformers) pour transformer chaque texte (ligne) en vecteur dense de 384 dimensions :

python
Copier le code
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
vectors = model.encode(list_of_texts, show_progress_bar=True)


1