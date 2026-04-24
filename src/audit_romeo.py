"""
audit_romeo.py — Validation qualité du moteur ChromaDB vs API ROMEO v2.

Compare notre moteur de recherche (paraphrase-multilingual-MiniLM-L12-v2 +
ChromaDB) avec l'API officielle France Travail ROMEO v2 sur 15 requêtes
couvrant les principales familles ROME.

Sortie :
  - Rapport console (tableau concordance + métriques globales)
  - data/romeo_audit.json
  - Métriques Prometheus mises à jour
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import chromadb
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from src.ingestion import encode_with_chunks
from src.metrics import (
    our_engine_avg_latency_ms,
    romeo_avg_latency_ms,
    romeo_concordance_full_rate,
    romeo_concordance_partial_rate,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ─── Constantes ──────────────────────────────────────────────────────────────

COLLECTION_NAME = 'naf_rome_v2'
CHROMA_PATH     = './chroma_db'
MODEL_NAME      = 'paraphrase-multilingual-MiniLM-L12-v2'
AUDIT_PATH      = 'data/romeo_audit.json'

TOKEN_URL = (
    'https://entreprise.francetravail.fr/connexion/oauth2/access_token'
    '?realm=%2Fpartenaire'
)
ROMEO_URL = 'https://api.francetravail.io/partenaire/romeo/v2/predictionMetiers'

REQUETES = [
    ("professeur de yoga",    "Enseignement de disciplines sportives"),
    ("boulanger artisan",     "Boulangerie et boulangerie-pâtisserie"),
    ("développeur Python",    "Programmation informatique"),
    ("infirmier urgences",    "Activités hospitalières"),
    ("chauffeur poids lourd", "Transports routiers de fret"),
    ("boucher",               "Commerce de détail de viandes"),
    ("comptable",             "Activités comptables"),
    ("plombier chauffagiste", "Travaux d'installation"),
    ("agent immobilier",      "Agences immobilières"),
    ("cuisinier restaurant",  "Restauration traditionnelle"),
    ("électricien bâtiment",  "Travaux d'installation électrique"),
    ("aide soignant",         "Hébergement médicalisé"),
    ("commercial B2B",        "Conseil pour les affaires"),
    ("data scientist",        "Programmation informatique"),
    ("mécanicien auto",       "Entretien et réparation de véhicules"),
]


# ─── Authentification OAuth2 ─────────────────────────────────────────────────

def get_token(client_id: str, client_secret: str) -> tuple[str, int]:
    """Obtient un Bearer token OAuth2 depuis l'API France Travail.

    Returns:
        (access_token, expires_in_seconds)

    Raises:
        RuntimeError si la réponse n'est pas 200.
    """
    resp = requests.post(
        TOKEN_URL,
        data={
            'grant_type':    'client_credentials',
            'client_id':     client_id,
            'client_secret': client_secret,
            'scope':         'api_romeov2',
        },
        headers={'Content-Type': 'application/x-www-form-urlencoded'},
        timeout=15,
    )
    if resp.status_code != 200:
        logger.error("Token KO %d : %s", resp.status_code, resp.text)
        raise RuntimeError(f"OAuth2 échec : HTTP {resp.status_code}")
    data = resp.json()
    logger.info("Token obtenu, expire dans %d secondes.", data['expires_in'])
    return data['access_token'], data['expires_in']


# ─── Appel ROMEO v2 ──────────────────────────────────────────────────────────

def romeo_predict(intitule: str, contexte_naf: str, token: str) -> list[dict]:
    """Interroge l'API ROMEO v2 et retourne les 3 metiersRome prédits.

    Args:
        intitule:    Intitulé de poste (ex: "développeur Python")
        contexte_naf: Libellé NAF pour contextualiser (ex: "Programmation informatique")
        token:       Bearer token OAuth2

    Returns:
        Liste de dicts {codeRome, libelleRome, libelleAppellation, scorePrediction}
    """
    body = {
        "appellations": [{
            "intitule":    intitule,
            "identifiant": "audit_001",
            "contexte":    contexte_naf,
        }],
        "options": {
            "nomAppelant":           "tp_naf_rome_audit",
            "nbResultats":           3,
            "seuilScorePrediction":  0.3,
        },
    }
    resp = requests.post(
        ROMEO_URL,
        json=body,
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type':  'application/json; charset=utf-8',
        },
        timeout=15,
    )
    resp.raise_for_status()
    metiers = resp.json()[0].get('metiersRome', [])
    return [
        {
            'codeRome':           m['codeRome'],
            'libelleRome':        m['libelleRome'],
            'libelleAppellation': m.get('libelleAppellation', ''),
            'scorePrediction':    m['scorePrediction'],
        }
        for m in metiers
    ]


# ─── Notre moteur ChromaDB ───────────────────────────────────────────────────

def notre_moteur_predict(
    intitule: str,
    model: SentenceTransformer,
    collection: chromadb.Collection,
    n: int = 3,
) -> list[dict]:
    """Interroge notre moteur ChromaDB et retourne les n résultats.

    Args:
        intitule:   Requête en langage naturel.
        model:      Modèle SentenceTransformer chargé.
        collection: Collection ChromaDB 'naf_rome_v2'.
        n:          Nombre de résultats.

    Returns:
        Liste de dicts {code_naf, code_rome, name, score}
    """
    vec, _ = encode_with_chunks(intitule, model)
    results = collection.query(
        query_embeddings=[vec.tolist()],
        n_results=n,
        include=['metadatas', 'distances'],
    )
    output = []
    for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
        score = 1.0 - dist / 2.0  # cosine distance [0,2] → score [-1,1]
        output.append({
            'code_naf':  meta.get('code_naf', ''),
            'code_rome': meta.get('code_rome', ''),
            'name':      meta.get('name', ''),
            'score':     round(score, 4),
        })
    return output


# ─── Logique de concordance ──────────────────────────────────────────────────

def _match_level(romeo_code: str, notre_code: str) -> str:
    """Calcule le niveau de concordance entre deux codes ROME.

    Returns:
        'FULL'     si codes identiques
        'PARTIAL'  si même famille (première lettre)
        'NO_MATCH' sinon
    """
    if not romeo_code or not notre_code:
        return 'NO_MATCH'
    if romeo_code == notre_code:
        return 'FULL'
    if romeo_code[0] == notre_code[0]:
        return 'PARTIAL'
    return 'NO_MATCH'


# ─── Rapport console ─────────────────────────────────────────────────────────

def _print_report(results_list: list[dict]) -> None:
    """Affiche le tableau de concordance et les métriques globales."""
    col_q  = 22
    col_r  = 18
    col_n  = 18
    col_m  = 9
    sep_h  = '─'
    sep_v  = '│'

    top    = f"┌{'─'*col_q}┬{'─'*col_r}┬{'─'*col_n}┬{'─'*col_m}┐"
    mid    = f"├{'─'*col_q}┼{'─'*col_r}┼{'─'*col_n}┼{'─'*col_m}┤"
    bot    = f"└{'─'*col_q}┴{'─'*col_r}┴{'─'*col_n}┴{'─'*col_m}┘"

    def _row(q: str, r: str, n: str, m: str) -> str:
        return (
            f"{sep_v}{q:<{col_q}}{sep_v}{r:<{col_r}}"
            f"{sep_v}{n:<{col_n}}{sep_v}{m:<{col_m}}{sep_v}"
        )

    print(f'\n{top}')
    print(_row(' Requête', ' ROMEO top-1', ' Notre top-1', ' Match'))
    print(mid)

    n_full = n_partial = 0
    romeo_scores = []
    notre_scores = []
    romeo_latencies = []
    notre_latencies = []

    for r in results_list:
        romeo_top  = r['romeo_top3'][0] if r['romeo_top3'] else {}
        notre_top  = r['notre_top3'][0] if r['notre_top3'] else {}
        romeo_code = romeo_top.get('codeRome', '—')
        notre_code = notre_top.get('code_rome', '—')
        romeo_score = romeo_top.get('scorePrediction', 0.0)
        notre_score = notre_top.get('score', 0.0)

        match = r['match']
        if match == 'FULL':
            n_full += 1
            match_str = ' FULL'
        elif match == 'PARTIAL':
            n_partial += 1
            match_str = '~ FAM'
        else:
            match_str = '  ---'

        romeo_cell = f" {romeo_code} ({romeo_score:.2f})"
        notre_cell = f" {notre_code} ({notre_score:.2f})"
        query_cell = f" {r['query'][:col_q-1]}"

        print(_row(query_cell, romeo_cell, notre_cell, f' {match_str}'))

        if romeo_score:
            romeo_scores.append(romeo_score)
        if notre_score:
            notre_scores.append(notre_score)
        romeo_latencies.append(r['romeo_latency_ms'])
        notre_latencies.append(r['notre_latency_ms'])

    print(bot)

    n_total = len(results_list)
    avg_romeo = sum(romeo_scores) / len(romeo_scores) if romeo_scores else 0.0
    avg_notre = sum(notre_scores) / len(notre_scores) if notre_scores else 0.0
    avg_romeo_ms = sum(romeo_latencies) / len(romeo_latencies) if romeo_latencies else 0.0
    avg_notre_ms = sum(notre_latencies) / len(notre_latencies) if notre_latencies else 0.0

    print(f'\n  Taux concordance FULL    : {n_full:>2} / {n_total}  ({n_full/n_total*100:.1f} %)')
    print(f'  Taux concordance PARTIEL : {n_partial:>2} / {n_total}  ({n_partial/n_total*100:.1f} %)')
    print(f'  Score ROMEO moyen        : {avg_romeo:.3f}')
    print(f'  Score notre moteur moyen : {avg_notre:.3f}')
    print(f'  Latence moy ROMEO (ms)   : {avg_romeo_ms:.0f}')
    print(f'  Latence moy notre moteur : {avg_notre_ms:.0f}')
    print()


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=== AUDIT ROMEO v2 ===")

    # Étape 0 — Credentials
    load_dotenv()
    client_id     = os.environ.get('FRANCE_TRAVAIL_CLIENT_ID', '')
    client_secret = os.environ.get('FRANCE_TRAVAIL_CLIENT_SECRET', '')
    if not client_id or not client_secret:
        raise RuntimeError(
            "Variables FRANCE_TRAVAIL_CLIENT_ID / FRANCE_TRAVAIL_CLIENT_SECRET "
            "absentes du .env"
        )
    logger.info("Credentials chargés depuis .env (client_id=%s…)", client_id[:6])

    # Étape 1 — Token
    token, _ = get_token(client_id, client_secret)

    # Charger modèle + collection une seule fois
    logger.info("Chargement du modèle : %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    client     = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    logger.info("Collection '%s' — %d docs.", COLLECTION_NAME, collection.count())

    # Étape 4 — Boucle 15 requêtes
    results_list: list[dict] = []

    for intitule, contexte in REQUETES:
        logger.info("Requête : « %s »", intitule)

        # Appel ROMEO (avec retry 401)
        t0 = time.perf_counter()
        try:
            romeo_results = romeo_predict(intitule, contexte, token)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 401:
                logger.warning("Token expiré, rafraîchissement…")
                token, _ = get_token(client_id, client_secret)
                romeo_results = romeo_predict(intitule, contexte, token)
            else:
                raise
        romeo_ms = (time.perf_counter() - t0) * 1000

        time.sleep(0.5)  # rate limiting

        # Appel notre moteur
        t1 = time.perf_counter()
        notre_results = notre_moteur_predict(intitule, model, collection)
        notre_ms = (time.perf_counter() - t1) * 1000

        # Concordance
        romeo_top_code = romeo_results[0]['codeRome'] if romeo_results else ''
        notre_top_code = notre_results[0]['code_rome'] if notre_results else ''
        match          = _match_level(romeo_top_code, notre_top_code)

        results_list.append({
            'query':            intitule,
            'contexte':         contexte,
            'romeo_top3':       romeo_results,
            'notre_top3':       notre_results,
            'match':            match,
            'romeo_latency_ms': round(romeo_ms, 1),
            'notre_latency_ms': round(notre_ms, 1),
        })

    # Étape 5 — Rapport console
    _print_report(results_list)

    # Calcul des métriques agrégées
    n_total   = len(results_list)
    n_full    = sum(1 for r in results_list if r['match'] == 'FULL')
    n_partial = sum(1 for r in results_list if r['match'] == 'PARTIAL')

    romeo_scores  = [r['romeo_top3'][0]['scorePrediction'] for r in results_list if r['romeo_top3']]
    notre_scores  = [r['notre_top3'][0]['score'] for r in results_list if r['notre_top3']]
    romeo_latencies = [r['romeo_latency_ms'] for r in results_list]
    notre_latencies = [r['notre_latency_ms'] for r in results_list]

    avg_romeo_score = sum(romeo_scores) / len(romeo_scores) if romeo_scores else 0.0
    avg_notre_score = sum(notre_scores) / len(notre_scores) if notre_scores else 0.0
    avg_romeo_ms    = sum(romeo_latencies) / len(romeo_latencies) if romeo_latencies else 0.0
    avg_notre_ms    = sum(notre_latencies) / len(notre_latencies) if notre_latencies else 0.0

    # Étape 6 — Sauvegarde JSON
    audit_data = {
        'timestamp':                 datetime.now(timezone.utc).isoformat(),
        'model_used':                MODEL_NAME,
        'concordance_full_rate':     round(n_full / n_total, 4),
        'concordance_partial_rate':  round((n_full + n_partial) / n_total, 4),
        'romeo_avg_score':           round(avg_romeo_score, 4),
        'our_avg_score':             round(avg_notre_score, 4),
        'romeo_avg_latency_ms':      round(avg_romeo_ms, 1),
        'our_avg_latency_ms':        round(avg_notre_ms, 1),
        'results':                   results_list,
    }
    with open(AUDIT_PATH, 'w', encoding='utf-8') as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False)
    logger.info("Audit sauvegardé → %s", AUDIT_PATH)

    # Prometheus
    romeo_concordance_full_rate.set(n_full / n_total * 100)
    romeo_concordance_partial_rate.set((n_full + n_partial) / n_total * 100)
    romeo_avg_latency_ms.set(avg_romeo_ms)
    our_engine_avg_latency_ms.set(avg_notre_ms)

    logger.info(
        "Audit terminé — FULL %d/%d (%.0f %%), FAM %d/%d (%.0f %%)",
        n_full, n_total, n_full / n_total * 100,
        n_partial, n_total, n_partial / n_total * 100,
    )


if __name__ == '__main__':
    main()
