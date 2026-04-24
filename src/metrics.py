from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

REGISTRY = CollectorRegistry()

corpus_total_rows = Gauge(
    'corpus_total_rows',
    'Nombre total de lignes dans le corpus après déduplication',
    registry=REGISTRY,
)
corpus_duplicates_removed = Gauge(
    'corpus_duplicates_removed',
    'Nombre de doublons supprimés lors de la déduplication',
    registry=REGISTRY,
)
corpus_rome_rows = Gauge(
    'corpus_rome_rows',
    'Nombre de lignes ROME (code_rome non null)',
    registry=REGISTRY,
)
corpus_naf_rows = Gauge(
    'corpus_naf_rows',
    'Nombre de lignes NAF (code_rome null)',
    registry=REGISTRY,
)
corpus_text_length_mean = Gauge(
    'corpus_text_length_mean',
    'Longueur moyenne de text_to_encode (en caractères)',
    registry=REGISTRY,
)
corpus_text_length_p90 = Gauge(
    'corpus_text_length_p90',
    'Percentile 90 de la longueur de text_to_encode',
    registry=REGISTRY,
)
corpus_text_length_p99 = Gauge(
    'corpus_text_length_p99',
    'Percentile 99 de la longueur de text_to_encode',
    registry=REGISTRY,
)
corpus_text_length_max = Gauge(
    'corpus_text_length_max',
    'Longueur maximale de text_to_encode',
    registry=REGISTRY,
)
corpus_token_overflow_128 = Gauge(
    'corpus_token_overflow_128',
    'Nombre estimé de lignes dépassant 128 tokens (heuristique len/3.5)',
    registry=REGISTRY,
)
corpus_token_overflow_128_ratio = Gauge(
    'corpus_token_overflow_128_ratio',
    'Ratio (%) de lignes estimées > 128 tokens — KPI qualité principal',
    registry=REGISTRY,
)
preprocessing_quality_score = Gauge(
    'preprocessing_quality_score',
    'Score qualité du preprocessing (objectif > 85.0)',
    registry=REGISTRY,
)
preprocessing_duration_seconds = Histogram(
    'preprocessing_duration_seconds',
    'Durée totale du pipeline de preprocessing en secondes',
    registry=REGISTRY,
)

# ─── Métriques enrichissement NAF ─────────────────────────────────────────────

naf_enriched_count = Gauge(
    'naf_enriched_count',
    'Nombre de codes NAF dont text_to_encode a été enrichi',
    registry=REGISTRY,
)
naf_avg_text_length_chars = Gauge(
    'naf_avg_text_length_chars',
    'Longueur moyenne (chars) de text_to_encode pour les lignes NAF après enrichissement',
    registry=REGISTRY,
)
naf_coverage_referenced_by_rome = Gauge(
    'naf_coverage_referenced_by_rome',
    '% de codes NAF liés à au moins une ligne ROME dans le corpus',
    registry=REGISTRY,
)
naf_orphan_codes_gauge = Gauge(
    'naf_orphan_codes',
    'Nombre de codes NAF non référencés par aucune ligne ROME — objectif : diminue',
    registry=REGISTRY,
)

# ─── Métriques ingestion ChromaDB ─────────────────────────────────────────────

ingestion_total_documents = Gauge(
    'ingestion_total_documents',
    'Nombre total de documents ingérés dans ChromaDB',
    registry=REGISTRY,
)
ingestion_total_chunked = Gauge(
    'ingestion_total_chunked',
    'Nombre de documents ayant nécessité un chunking (> max_tokens)',
    registry=REGISTRY,
)
ingestion_chunked_ratio = Gauge(
    'ingestion_chunked_ratio',
    'Ratio (%) de documents chunkés',
    registry=REGISTRY,
)
ingestion_token_overflow_count = Gauge(
    'ingestion_token_overflow_count',
    'Nombre de documents dépassant max_tokens avant chunking',
    registry=REGISTRY,
)
ingestion_duration_seconds = Gauge(
    'ingestion_duration_seconds',
    'Durée totale de l\'ingestion en secondes',
    registry=REGISTRY,
)
ingestion_avg_batch_duration_ms = Gauge(
    'ingestion_avg_batch_duration_ms',
    'Latence moyenne par batch en millisecondes',
    registry=REGISTRY,
)
ingestion_total_chunks = Gauge(
    'ingestion_total_chunks',
    'Nombre total d\'entrées ChromaDB générées (1 par chunk)',
    registry=REGISTRY,
)
ingestion_avg_chunks_per_doc = Gauge(
    'ingestion_avg_chunks_per_doc',
    'Nombre moyen de chunks par document source',
    registry=REGISTRY,
)

# ─── Métriques audit ROMEO ────────────────────────────────────────────────────

romeo_concordance_full_rate = Gauge(
    'romeo_concordance_full_rate',
    'Taux de concordance exacte (même code_rome) avec ROMEO v2 (%)',
    registry=REGISTRY,
)
romeo_concordance_partial_rate = Gauge(
    'romeo_concordance_partial_rate',
    'Taux de concordance famille (même lettre ROME) avec ROMEO v2 (%)',
    registry=REGISTRY,
)
romeo_avg_latency_ms = Gauge(
    'romeo_avg_latency_ms',
    'Latence moyenne des appels API ROMEO v2 (ms)',
    registry=REGISTRY,
)
our_engine_avg_latency_ms = Gauge(
    'our_engine_avg_latency_ms',
    'Latence moyenne de notre moteur ChromaDB (ms)',
    registry=REGISTRY,
)

# ─── Métriques search ─────────────────────────────────────────────────────────

search_family_boost_applied = Gauge(
    'search_family_boost_applied',
    '1.0 si le dernier appel à search() utilisait family_boost, 0.0 sinon',
    registry=REGISTRY,
)
search_chunks_fetched_before_dedup = Gauge(
    'search_chunks_fetched_before_dedup',
    'Nombre de chunks récupérés depuis ChromaDB avant déduplication',
    registry=REGISTRY,
)
search_chunks_after_dedup = Gauge(
    'search_chunks_after_dedup',
    'Nombre de documents uniques après déduplication par source_idx',
    registry=REGISTRY,
)
search_latency_seconds = Histogram(
    'search_latency_seconds',
    'Latence de l\'endpoint de recherche (encode + ChromaDB + dedup)',
    registry=REGISTRY,
)
search_score_top1 = Gauge(
    'search_score_top1',
    'Score cosinus du résultat top-1 (après boost éventuel)',
    registry=REGISTRY,
)
search_results_returned = Gauge(
    'search_results_returned',
    'Nombre de résultats retournés par le dernier appel search()',
    registry=REGISTRY,
)
search_query_token_length = Gauge(
    'search_query_token_length',
    'Longueur en tokens de la dernière requête envoyée à search()',
    registry=REGISTRY,
)
search_requests = Counter(
    'search_requests',
    'Nombre total de recherches exécutées, ventilées par endpoint/persona/boost',
    ['endpoint', 'persona', 'boost'],
    registry=REGISTRY,
)
search_zero_results = Counter(
    'search_zero_results',
    'Nombre total de recherches ne retournant aucun résultat',
    ['endpoint', 'persona'],
    registry=REGISTRY,
)
search_top1_band = Counter(
    'search_top1_band',
    'Répartition des bandes de confiance du résultat top-1',
    ['band', 'endpoint', 'persona'],
    registry=REGISTRY,
)
search_top1_code_rome = Counter(
    'search_top1_code_rome',
    'Nombre de fois où un code ROME ressort en top-1',
    ['code_rome', 'name', 'persona'],
    registry=REGISTRY,
)
search_top1_code_naf = Counter(
    'search_top1_code_naf',
    'Nombre de fois où un code NAF ressort en top-1',
    ['code_naf', 'persona'],
    registry=REGISTRY,
)
search_top1_famille = Counter(
    'search_top1_famille',
    'Nombre de fois où une famille ROME ressort en top-1',
    ['famille', 'persona'],
    registry=REGISTRY,
)
search_latency_by_persona_seconds = Histogram(
    'search_latency_by_persona_seconds',
    'Latence de recherche ventilée par endpoint/persona',
    ['endpoint', 'persona'],
    registry=REGISTRY,
)
search_query_token_length_histogram = Histogram(
    'search_query_token_length_histogram',
    'Distribution de la taille des requêtes en tokens',
    buckets=(1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256),
    registry=REGISTRY,
)
search_top1_score_distribution = Histogram(
    'search_top1_score_distribution',
    'Distribution des scores du résultat top-1',
    ['endpoint', 'persona'],
    buckets=(0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

# ─── Métriques HTTP par endpoint ──────────────────────────────────────────────

collection_count = Gauge(
    'collection_count',
    'Nombre d\'entrées dans la collection ChromaDB (mis à jour au démarrage)',
    registry=REGISTRY,
)
http_requests_total_rome_search = Counter(
    'http_requests_total_rome_search',
    'Nombre total de requêtes GET /rome/search (Sophie)',
    registry=REGISTRY,
)
http_requests_total_match = Counter(
    'http_requests_total_match',
    'Nombre total de requêtes POST /match (Karim)',
    registry=REGISTRY,
)
