Comment ça marche ?
Étape	Détail	Où ça se passe
1. Lecture	charge data/processed/<dataset>_clean.parquet (RNA ou SIRENE).	load_dataset()
2. Embeddings	encode chaque texte avec MiniLM (384 dims, normalisés L2) par batches (--batch 256 par défaut).	encode_batches()
3. Sauvegarde	• vecteurs → data/embeddings/<dataset>_vecs.f32 (memmap float32)
• métadonnées → data/embeddings/<dataset>_meta.parquet	save_embeddings()
4. Index FAISS	Seulement pour SIRENE : construit un IndexHNSWFlat (32 voisins, inner-product).	build_faiss_index()
5. Résultat	data/faiss/sirene.index + fichiers embeddings.	sortie console

Lancer pas-à-pas
bash
Copier le code
# ➜  (venv) python -m scripts.embed_and_index --dataset rna --batch 256
# ➜  (venv) python -m scripts.embed_and_index --dataset sirene --batch 256
Options utiles :

--model pour tester un autre Sentence-Transformer.

--force écrase les fichiers déjà présents (pratique si tu relances).

--batch 128 si ta RAM est serrée (⇢ moins de conso, un peu plus long).

Ce que le script gère déjà
Reprise facile : si les fichiers d’output existent, il s’arrête (sauf --force).

RAM modeste : batch encoding, vectors écrits en memmap.

CPU / GPU : SentenceTransformer switche tout seul sur CUDA si dispo.

Index HNSW : rapide en requête (~ log n) ; paramètre efConstruction=200 pour une bonne précision (> 95 %).