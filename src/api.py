"""
api.py — API FastAPI NAFnROME.

Endpoints :
  GET  /health            → statut + taille collection
  GET  /metrics           → scrape Prometheus (REGISTRY)
  GET  /rome/search       → recherche sémantique (query param q)
  GET  /rome/{code}       → fiche d'un code ROME
  POST /match             → recherche sémantique (body JSON)

IMPORTANT : /rome/search est déclaré AVANT /rome/{code} pour éviter
que FastAPI route "search" comme un code ROME.

Le modèle et la collection sont chargés une seule fois au démarrage
via le context manager lifespan (FastAPI >= 0.93).
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import chromadb
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.metrics import (
    REGISTRY,
    search_latency_seconds,
    search_score_top1,
)
from src.search import search

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────────

COLLECTION_NAME = 'naf_rome_v2'
CHROMA_PATH     = './chroma_db'
MODEL_NAME      = 'paraphrase-multilingual-MiniLM-L12-v2'

# Boost par défaut utilisé quand boost=true
FAMILY_BOOST = {"J": 1.2, "D": 1.1, "H": 0.85}


# ─── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et la collection une seule fois au démarrage."""
    logger.info("Démarrage — chargement du modèle : %s", MODEL_NAME)
    app.state.model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    app.state.collection = client.get_collection(COLLECTION_NAME)

    n = app.state.collection.count()
    logger.info("Prêt — collection '%s' (%d entrées).", COLLECTION_NAME, n)
    yield
    # Pas de cleanup nécessaire (ChromaDB PersistentClient gère la fermeture)


app = FastAPI(title="NAFnROME API", version="1.0.0", lifespan=lifespan)


# ─── Helper partagé search → réponse ─────────────────────────────────────────

def _search_response(
    query: str,
    results: list[dict],
    search_meta: dict,
    model: SentenceTransformer,
    boost_applied: bool,
    t0: float,
) -> dict:
    """Construit la réponse JSON commune à /rome/search et /match."""
    latency_ms = (time.perf_counter() - t0) * 1000
    token_count = len(model.tokenizer.encode(query, add_special_tokens=False))
    top1_score  = results[0]['score'] if results else 0.0

    # Prometheus
    search_latency_seconds.observe(latency_ms / 1000)
    search_score_top1.set(top1_score)

    return {
        "query":   query,
        "results": [{"rank": i + 1, **r} for i, r in enumerate(results)],
        "meta": {
            "chunks_fetched_before_dedup": search_meta['chunks_fetched_before_dedup'],
            "chunks_after_dedup":          search_meta['chunks_after_dedup'],
            "boost_applied":               boost_applied,
            "model_used":                  MODEL_NAME,
            "query_token_count":           token_count,
            "latency_ms":                  round(latency_ms, 1),
        },
    }


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    return {
        "status":           "ok",
        "collection_count": request.app.state.collection.count(),
        "model":            MODEL_NAME,
    }


# IMPORTANT : déclaré AVANT /rome/{code}
@app.get("/rome/search")
async def rome_search(
    request: Request,
    q: str,
    n: int = 5,
    boost: bool = False,
):
    """Recherche sémantique par texte libre.

    Params :
      q     — requête (ex: "professeur yoga")
      n     — nombre de résultats (défaut 5)
      boost — applique FAMILY_BOOST si true
    """
    t0           = time.perf_counter()
    model        = request.app.state.model
    collection   = request.app.state.collection
    family_boost = FAMILY_BOOST if boost else None

    results, search_meta = search(
        q, model, collection, n_results=n, family_boost=family_boost
    )
    return _search_response(q, results, search_meta, model, boost, t0)


@app.get("/rome/{code}")
async def rome_by_code(request: Request, code: str):
    """Fiche d'un code ROME : appellations, NAF associés, famille.

    Retourne 404 si le code est inconnu.
    """
    t0         = time.perf_counter()
    collection = request.app.state.collection

    raw = collection.get(
        where={"code_rome": {"$eq": code}},
        include=["metadatas"],
        limit=10000,
    )

    if not raw["metadatas"]:
        raise HTTPException(
            status_code=404,
            detail=f"Code ROME '{code}' introuvable dans la collection.",
        )

    # Déduplication par source_idx (plusieurs chunks → même document source)
    seen: dict = {}
    for meta in raw["metadatas"]:
        # Fallback sur id(meta) pour l'ancien schéma sans source_idx
        key = meta.get("source_idx", id(meta))
        if key not in seen:
            seen[key] = meta

    unique_docs  = list(seen.values())
    name         = unique_docs[0].get("name", "")
    famille      = code[0] if code else ""
    naf_codes    = list(dict.fromkeys(
        m.get("code_naf", "") for m in unique_docs if m.get("code_naf")
    ))

    return {
        "code_rome":         code,
        "name":              name,
        "famille":           famille,
        "appellations_count": len(unique_docs),
        "naf_codes":         naf_codes,
        "timestamp":         datetime.now(timezone.utc).isoformat(),
        "latency_ms":        round((time.perf_counter() - t0) * 1000, 1),
    }


class MatchRequest(BaseModel):
    query: str
    n:     int  = 5
    boost: bool = False


@app.post("/match")
async def match(request: Request, body: MatchRequest):
    """Recherche sémantique via body JSON.

    Body : {"query": "...", "n": 5, "boost": true}
    Même logique et structure de réponse que /rome/search.
    """
    t0           = time.perf_counter()
    model        = request.app.state.model
    collection   = request.app.state.collection
    family_boost = FAMILY_BOOST if body.boost else None

    results, search_meta = search(
        body.query, model, collection,
        n_results=body.n, family_boost=family_boost,
    )
    return _search_response(body.query, results, search_meta, model, body.boost, t0)


@app.get("/metrics")
async def metrics():
    """Endpoint de scrape Prometheus — expose le REGISTRY du projet."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
