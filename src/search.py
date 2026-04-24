"""
search.py — Moteur de recherche NAFnROME via ChromaDB.

Fonction principale : search()
  Encode une requête en langage naturel, interroge la collection
  'naf_rome_v2', déduplique par source_idx (un résultat par document
  source), applique un boost optionnel par famille ROME, et retourne
  les résultats triés.

Retour : tuple (results, search_meta)
  results     — liste de dicts triés par score décroissant
  search_meta — {'chunks_fetched_before_dedup': int, 'chunks_after_dedup': int}
"""

import logging

import chromadb
from sentence_transformers import SentenceTransformer

from src.ingestion import encode_with_chunks
from src.metrics import (
    search_chunks_after_dedup,
    search_chunks_fetched_before_dedup,
    search_family_boost_applied,
)

logger = logging.getLogger(__name__)


def search(
    query: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    n_results: int = 5,
    family_boost: dict | None = None,
) -> tuple[list[dict], dict]:
    """Recherche sémantique dans la collection ChromaDB naf_rome_v2.

    Stratégie :
      1. Overfetch × 5 pour avoir des candidats suffisants après déduplication.
      2. Déduplication par source_idx : un seul résultat par document source,
         en conservant le chunk au meilleur score.
      3. Boost optionnel : score = score_raw × family_boost.get(famille, 1.0).
      4. Tri final par score décroissant, troncature à n_results.

    Args:
        query:        Requête en langage naturel (ex: "développeur Python").
        model:        Modèle SentenceTransformer chargé.
        collection:   Collection ChromaDB cible.
        n_results:    Nombre de résultats à retourner.
        family_boost: Dictionnaire optionnel famille ROME → multiplicateur.
                      Ex: {"J": 1.2, "D": 1.1, "H": 0.85}
                      Les familles absentes du dict gardent un facteur 1.0.

    Returns:
        (results, search_meta) où :
          results   — liste de dicts triés par score décroissant :
                      {source_idx, code_naf, code_rome, name, famille,
                       chunk_idx, n_chunks, score_raw, score}
          search_meta — {'chunks_fetched_before_dedup': int,
                         'chunks_after_dedup': int}
    """
    # Overfetch fixe × 5 : couvre à la fois la déduplication et le boost
    fetch_k = min(n_results * 5, collection.count())

    vec, _ = encode_with_chunks(query, model)
    raw = collection.query(
        query_embeddings=[vec.tolist()],
        n_results=fetch_k,
        include=['metadatas', 'distances'],
    )

    # Déduplication par source_idx — on garde le chunk au meilleur score
    seen: dict[int, dict] = {}
    for meta, dist in zip(raw['metadatas'][0], raw['distances'][0]):
        score_raw  = 1.0 - dist / 2.0  # cosine distance [0,2] → [-1,1]
        famille    = meta.get('famille', '')
        factor     = family_boost.get(famille, 1.0) if family_boost else 1.0
        score      = round(score_raw * factor, 4)
        source_idx = meta.get('source_idx', -1)

        if source_idx not in seen or score > seen[source_idx]['score']:
            seen[source_idx] = {
                'source_idx': source_idx,
                'code_naf':   meta.get('code_naf', ''),
                'code_rome':  meta.get('code_rome', ''),
                'name':       meta.get('name', ''),
                'famille':    famille,
                'chunk_idx':  meta.get('chunk_idx', 0),
                'n_chunks':   meta.get('n_chunks', 1),
                'score_raw':  round(score_raw, 4),
                'score':      score,
            }

    results = sorted(seen.values(), key=lambda r: r['score'], reverse=True)[:n_results]

    chunks_fetched = len(raw['metadatas'][0])
    chunks_deduped = len(seen)

    # Prometheus
    search_family_boost_applied.set(1.0 if family_boost else 0.0)
    search_chunks_fetched_before_dedup.set(chunks_fetched)
    search_chunks_after_dedup.set(chunks_deduped)

    if family_boost:
        logger.debug(
            "family_boost=%s — %d chunks → %d docs uniques → top %d",
            family_boost, chunks_fetched, chunks_deduped, n_results,
        )

    search_meta = {
        'chunks_fetched_before_dedup': chunks_fetched,
        'chunks_after_dedup':          chunks_deduped,
    }
    return list(results), search_meta
