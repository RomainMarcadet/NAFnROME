# # Contexte du projet NAFnROME

## Objectif
Application de matching sémantique entre codes ROME (France Travail) 
et codes NAF (INSEE) via embeddings et recherche vectorielle ChromaDB.

## Stack actuelle (repo existant)
- Python 3.10+
- SentenceTransformers (all-MiniLM-L6-v2 — À REMPLACER)
- ChromaDB PersistentClient
- Pandas
- Notebooks Jupyter (pipeline 01 → 06)

## Problèmes identifiés à corriger
1. Modèle anglais sur données 100% françaises → remplacer par
   paraphrase-multilingual-MiniLM-L12-v2
2. Limite 128 tokens non gérée → troncature silencieuse des longues 
   descriptions NAF
3. ChromaDB utilisé comme stockage uniquement, pas pour la recherche 
   → utiliser collection.query() natif
4. Logique de filtrage code_rome inversée dans search_keywords
5. Aucune instrumentation / métriques

## Architecture cible
- src/ingestion.py   : pipeline d'ingestion avec chunking
- src/search.py      : moteur de recherche via ChromaDB natif
- src/api.py         : API FastAPI
- src/metrics.py     : instrumentation Prometheus
- docker-compose.yml : API + Prometheus + Grafana
- tests/             : tests unitaires

## Contraintes machine
- MacBook M1 8Go RAM
- Modèle en local (pas d'API externe)
- ChromaDB en mode persistant local

## Personas utilisateurs
- Sophie : CIP France Travail, experte, saisit des codes ROME directs
- Karim  : Headhunter, novice, tape en langage naturel
