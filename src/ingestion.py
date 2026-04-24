"""
ingestion.py — Ingestion du corpus NAFnROME dans ChromaDB.

Source : data/corpus_final.csv (8030 lignes)
Modèle : paraphrase-multilingual-MiniLM-L12-v2 (384 dim, 128 tokens max)
Collection : naf_rome_v2

Stratégie chunk-per-entry :
  Chaque chunk d'un document source devient une entrée ChromaDB distincte.
  Un document court (≤ max_tokens) génère exactement 1 entrée.
  Un document long génère N entrées (une par fenêtre glissante).

Comportement :
  - Si la collection existe et count() > 0 : affiche les stats,
    passe directement à la validation
  - Sinon : ingestion complète + rapport + validation

Pour ré-ingérer depuis zéro (nouveau schéma), supprimer chroma_db/ puis
relancer : python -m src.ingestion
"""

import json
import logging
import time
from datetime import datetime, timezone

import chromadb
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.metrics import (
    ingestion_avg_batch_duration_ms,
    ingestion_avg_chunks_per_doc,
    ingestion_chunked_ratio,
    ingestion_duration_seconds,
    ingestion_token_overflow_count,
    ingestion_total_chunked,
    ingestion_total_chunks,
    ingestion_total_documents,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────────

CORPUS_PATH     = 'data/corpus_final.csv'
CHROMA_PATH     = './chroma_db'
COLLECTION_NAME = 'naf_rome_v2'
MODEL_NAME      = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_DIM   = 384
BATCH_SIZE      = 500
MAX_TOKENS      = 128
OVERLAP_TOKENS  = 20
METRICS_PATH    = 'data/ingestion_metrics.json'

TEST_QUERIES = [
    "professeur yoga",
    "boulangerie artisanale",
    "développeur logiciel",
    "infirmier urgences",
    "chauffeur poids lourd",
]


# ─── Chunking ────────────────────────────────────────────────────────────────

def _iter_chunks(
    text: str,
    model: SentenceTransformer,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[tuple[np.ndarray, str, int]]:
    """Découpe un texte en chunks et encode chaque chunk séparément.

    Args:
        text:       Texte à encoder.
        model:      Modèle SentenceTransformer chargé.
        max_tokens: Tokens de contenu max par chunk (hors CLS/SEP).
        overlap:    Tokens de recouvrement entre fenêtres consécutives.

    Returns:
        Liste de (embedding float32 shape (384,), chunk_text, token_count).
        Un seul élément si le texte tient en max_tokens.
    """
    content_ids: list[int] = model.tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(content_ids)

    if total_tokens <= max_tokens:
        vec = model.encode(text, normalize_embeddings=True).astype(np.float32)
        return [(vec, text, total_tokens)]

    step = max_tokens - overlap
    windows: list[str] = []
    token_counts: list[int] = []
    start = 0
    while start < total_tokens:
        window_ids = content_ids[start: start + max_tokens]
        windows.append(model.tokenizer.decode(window_ids, skip_special_tokens=True))
        token_counts.append(len(window_ids))
        start += step

    chunk_vecs = model.encode(windows, normalize_embeddings=True)  # (n, 384)
    return [
        (chunk_vecs[i].astype(np.float32), windows[i], token_counts[i])
        for i in range(len(windows))
    ]


def encode_with_chunks(
    text: str,
    model: SentenceTransformer,
    max_tokens: int = MAX_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> tuple[np.ndarray, bool]:
    """Encode un texte en un seul vecteur (moyenne normalisée si chunké).

    Utilisé par search.py et audit_romeo.py pour encoder des requêtes courtes.
    Pour l'ingestion, utiliser _iter_chunks() qui retourne une entrée par chunk.

    Args:
        text:       Texte à encoder.
        model:      Modèle SentenceTransformer chargé.
        max_tokens: Tokens de contenu max par chunk.
        overlap:    Tokens de recouvrement.

    Returns:
        (embedding float32 shape (384,), chunked: bool)
    """
    chunks = _iter_chunks(text, model, max_tokens, overlap)
    if len(chunks) == 1:
        return chunks[0][0], False

    vecs = np.stack([c[0] for c in chunks])
    mean_vec = vecs.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.astype(np.float32), True


# ─── Ingestion ────────────────────────────────────────────────────────────────

def _run_ingestion(
    df: pd.DataFrame,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """Boucle d'ingestion : découpe en chunks + push vers ChromaDB par batches.

    Chaque chunk d'un document source devient une entrée ChromaDB distincte.

    Args:
        df:         DataFrame corpus (colonnes: code_naf, code_rome, name, text_to_encode)
        model:      Modèle d'embedding chargé.
        collection: Collection ChromaDB cible (vide).
        batch_size: Nombre d'entrées (chunks) par batch.

    Returns:
        Dictionnaire des métriques brutes.
    """
    t_start = time.perf_counter()
    batch_durations: list[float] = []

    n_source_docs    = len(df)
    n_total_chunks   = 0
    n_chunked_docs   = 0
    n_overflow       = 0  # docs dont le texte dépasse max_tokens

    ids_buf:        list[str]         = []
    embeddings_buf: list[list[float]] = []
    documents_buf:  list[str]         = []
    metadatas_buf:  list[dict]        = []

    def _flush() -> None:
        nonlocal batch_durations
        t_batch = time.perf_counter()
        collection.add(
            ids=ids_buf,
            embeddings=embeddings_buf,
            documents=documents_buf,
            metadatas=metadatas_buf,
        )
        batch_durations.append(time.perf_counter() - t_batch)
        ids_buf.clear(); embeddings_buf.clear()
        documents_buf.clear(); metadatas_buf.clear()

    for source_idx, row in df.iterrows():
        text      = str(row.get('text_to_encode') or '').strip()
        code_naf  = str(row.get('code_naf')  or '').strip()
        code_rome = (
            str(row['code_rome']).strip()
            if pd.notna(row.get('code_rome')) else ''
        )
        name = str(row.get('name') or '').strip()

        # Compter les docs qui dépassent max_tokens (avant chunking)
        raw_token_count = len(
            model.tokenizer.encode(text, add_special_tokens=False)
        )
        if raw_token_count > MAX_TOKENS:
            n_overflow += 1

        chunks = _iter_chunks(text, model)
        n_chunks = len(chunks)
        n_total_chunks += n_chunks
        if n_chunks > 1:
            n_chunked_docs += 1

        rome_part = code_rome if code_rome else 'NAF'
        famille   = code_rome[0] if code_rome else 'NAF'

        for chunk_idx, (vec, chunk_text, token_count) in enumerate(chunks):
            entry_id = f"{code_naf}__{rome_part}__{source_idx}__c{chunk_idx}"

            ids_buf.append(entry_id)
            embeddings_buf.append(vec.tolist())
            documents_buf.append(chunk_text)
            metadatas_buf.append({
                'source_idx':  int(source_idx),
                'code_naf':    code_naf,
                'code_rome':   code_rome,
                'name':        name,
                'chunk_idx':   chunk_idx,
                'n_chunks':    n_chunks,
                'token_count': token_count,
                'famille':     famille,
                'is_naf_only': not bool(code_rome),
            })

            if len(ids_buf) == batch_size:
                _flush()
                done = int(source_idx) + 1
                logger.info(
                    "  Batch ingéré : %d / %d docs (%.1f %%) — %d chunks total",
                    done, n_source_docs, done / n_source_docs * 100, n_total_chunks,
                )

    if ids_buf:
        _flush()
        logger.info("  Résidu ingéré : %d entrées.", len(ids_buf) + len(ids_buf))

    duration    = time.perf_counter() - t_start
    n_in_chroma = collection.count()
    avg_batch_ms = (
        sum(batch_durations) / len(batch_durations) * 1000
        if batch_durations else 0.0
    )
    chunked_ratio = n_chunked_docs / n_source_docs * 100 if n_source_docs > 0 else 0.0
    avg_chunks    = n_total_chunks / n_source_docs if n_source_docs > 0 else 0.0

    # Prometheus
    ingestion_total_documents.set(n_source_docs)
    ingestion_total_chunks.set(n_total_chunks)
    ingestion_total_chunked.set(n_chunked_docs)
    ingestion_chunked_ratio.set(chunked_ratio)
    ingestion_token_overflow_count.set(n_overflow)
    ingestion_duration_seconds.set(duration)
    ingestion_avg_batch_duration_ms.set(avg_batch_ms)
    ingestion_avg_chunks_per_doc.set(avg_chunks)

    return {
        'timestamp':                       datetime.now(timezone.utc).isoformat(),
        'embedding_model_name':            MODEL_NAME,
        'embedding_dimension':             EMBEDDING_DIM,
        'collection_name':                 COLLECTION_NAME,
        'ingestion_total_documents':       n_source_docs,
        'ingestion_total_chunks':          n_total_chunks,
        'ingestion_chroma_entries':        n_in_chroma,
        'ingestion_total_chunked':         n_chunked_docs,
        'ingestion_chunked_ratio':         round(chunked_ratio, 2),
        'ingestion_avg_chunks_per_doc':    round(avg_chunks, 3),
        'ingestion_token_overflow_count':  n_overflow,
        'ingestion_duration_seconds':      round(duration, 3),
        'ingestion_avg_batch_duration_ms': round(avg_batch_ms, 2),
    }


# ─── Validation ──────────────────────────────────────────────────────────────

def run_validation(
    model: SentenceTransformer,
    collection: chromadb.Collection,
    queries: list[str] = TEST_QUERIES,
    n_results: int = 3,
) -> None:
    """Exécute les requêtes de test et affiche un tableau de résultats."""
    sep = '─' * 80
    print(f'\n{sep}')
    print('  VALIDATION — REQUÊTES DE TEST')
    print(sep)

    for query in queries:
        t0 = time.perf_counter()
        q_vec, _ = encode_with_chunks(query, model)
        encode_ms = (time.perf_counter() - t0) * 1000

        t1 = time.perf_counter()
        results = collection.query(
            query_embeddings=[q_vec.tolist()],
            n_results=n_results,
            include=['metadatas', 'documents', 'distances'],
        )
        query_ms = (time.perf_counter() - t1) * 1000

        print(f'\n  Requête : "{query}"')
        print(f'  Encode : {encode_ms:.1f} ms  |  ChromaDB : {query_ms:.1f} ms')
        print(f'  {"#":<3} {"code_naf":<10} {"code_rome":<10} {"chunk":>5} {"score":>6}  name')
        print(f'  {"─"*3} {"─"*10} {"─"*10} {"─"*5} {"─"*6}  {"─"*40}')

        for rank, (meta, dist) in enumerate(
            zip(results['metadatas'][0], results['distances'][0]), 1
        ):
            score     = 1.0 - dist / 2.0
            code_naf  = meta.get('code_naf',  '—')
            code_rome = meta.get('code_rome', '') or 'NAF'
            chunk_info = f"{meta.get('chunk_idx', 0)+1}/{meta.get('n_chunks', 1)}"
            name      = meta.get('name', '—')[:40]
            print(f'  [{rank}] {code_naf:<10} {code_rome:<10} {chunk_info:>5} {score:>6.3f}  {name}')

    print(f'\n{sep}\n')


# ─── Rapport ─────────────────────────────────────────────────────────────────

def _print_report(metrics: dict, n_in_chroma: int) -> None:
    sep = '─' * 70
    print(f'\n{sep}')
    print('  RAPPORT INGESTION NAFnROME')
    print(sep)
    rows = [
        ('Modèle',                          metrics.get('embedding_model_name', '—')),
        ('Dimension vecteurs',              str(metrics.get('embedding_dimension', '—'))),
        ('Collection ChromaDB',             metrics.get('collection_name', '—')),
        ('Documents source',                str(metrics.get('ingestion_total_documents', '—'))),
        ('Entrées ChromaDB (chunks)',        str(n_in_chroma)),
        ('Documents chunkés (> 1 chunk)',   str(metrics.get('ingestion_total_chunked', '—'))),
        ('Ratio chunking (%)',               f'{metrics.get("ingestion_chunked_ratio", 0):.1f}'),
        ('Moy. chunks / document',          f'{metrics.get("ingestion_avg_chunks_per_doc", 0):.3f}'),
        ('Docs > 128 tokens (overflow)',    str(metrics.get('ingestion_token_overflow_count', '—'))),
        ('Durée totale (s)',                 f'{metrics.get("ingestion_duration_seconds", 0):.1f}'),
        ('Latence moy. par batch (ms)',     f'{metrics.get("ingestion_avg_batch_duration_ms", 0):.1f}'),
    ]
    for label, value in rows:
        print(f'  {label:<40} {value:>26}')
    print(sep)


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main(
    corpus_path: str = CORPUS_PATH,
    chroma_path: str = CHROMA_PATH,
) -> None:
    logger.info("=== INGESTION NAFnROME ===")

    logger.info("Chargement du modèle : %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)
    logger.info("  Modèle chargé. Dimension : %d", EMBEDDING_DIM)

    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    existing = collection.count()

    if existing > 0:
        logger.info(
            "Collection '%s' déjà peuplée (%d entrées). Ingestion ignorée.",
            COLLECTION_NAME, existing,
        )
        try:
            with open(METRICS_PATH, encoding='utf-8') as f:
                metrics = json.load(f)
        except FileNotFoundError:
            metrics = {
                'embedding_model_name':      MODEL_NAME,
                'embedding_dimension':       EMBEDDING_DIM,
                'collection_name':           COLLECTION_NAME,
                'ingestion_total_documents': existing,
            }
    else:
        logger.info("Chargement du corpus : %s", corpus_path)
        df = pd.read_csv(corpus_path, dtype=str)
        logger.info("  %d lignes chargées.", len(df))

        metrics = _run_ingestion(df, model, collection, batch_size=BATCH_SIZE)

        with open(METRICS_PATH, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Métriques sauvegardées → %s", METRICS_PATH)

    _print_report(metrics, collection.count())
    run_validation(model, collection)


if __name__ == '__main__':
    main()
