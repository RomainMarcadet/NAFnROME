from prometheus_client import Gauge, Histogram, CollectorRegistry

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
