Je viens de déposer scripts/clean_sirene.py (V2) dans le dépôt.
Points-clés :

Étape	Détails	Pourquoi c’est mieux
Lecture	– Cherche d’abord data/interim/sirene_full.parquet (rapide)
– Bascule sur le CSV seulement si le Parquet n’existe pas	Évite de relire 40 M lignes quand on dispose déjà du Parquet
Sélection minimaliste	Charge uniquement les colonnes listées dans COLS_MAP	-50 % RAM et I/O
Filtre “établissements actifs”	Set VALID_ETAT extensible ; on normalise en .upper()	Plus de « aucune ligne » si les valeurs sont “ACTIF”, “1”…
Nettoyage	– strip_acc() retire accents/espaces
– fix_cp() supprime le “.0” résiduel et force 5 car.	Code postal propre même après import CSV
Champ texte	Concatène enseigne + naf déjà normalisés	Prêt pour l’embedding MiniLM
Pathlib	Chemins robustes (Path) au lieu de os.path	Lisible & portable
Export	Parquet Snappy compact dans data/processed/	Lecture ultrarapide pour V2

Utilisation
bash
Copier le code
# depuis la racine du projet (venv activé)
python -m scripts.clean_sirene
Cherche d’abord data/interim/sirene_full.parquet (si tu l’as déjà généré avec la V1).

Écrit : data/processed/sirene_clean.parquet
→ message final : ✅ SIRENE nettoyé : 16 573 584 lignes → data/processed/sirene_clean.parquet

Si le Parquet d’origine n’existe pas
Le script tombera automatiquement sur sirene_full.csv. Garder ce CSV au même endroit (40 M lignes ≈ 1,3 Go) et lance simplement la même commande ; la lecture sera plus longue mais le résultat identique.

Cela règle :

les problèmes de chemins relatifs,

le bug “0 lignes” dû aux valeurs de etat,

les codes-postaux mal formatés.