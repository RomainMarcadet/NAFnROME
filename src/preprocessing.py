"""
preprocessing.py — Nettoyage et préparation du corpus NAFnROME.

Produit data/corpus_clean.csv à partir de fusion_naf_rome_001_allMiniLM_L6_v2.csv.
Ce fichier est l'unique source d'entrée pour src/ingestion.py.
"""

import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.metrics import (
    corpus_duplicates_removed,
    corpus_naf_rows,
    corpus_rome_rows,
    corpus_text_length_max,
    corpus_text_length_mean,
    corpus_text_length_p90,
    corpus_text_length_p99,
    corpus_token_overflow_128,
    corpus_token_overflow_128_ratio,
    corpus_total_rows,
    naf_orphan_codes_gauge,
    preprocessing_duration_seconds,
    preprocessing_quality_score,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ─── Pattern compilés une seule fois ──────────────────────────────────────────

_OGR_SPLIT = re.compile(r'\s*\(code OGR:\d+\)\s*libelle:\s*')
_NAF_CODE_PREFIX = re.compile(r'^\d{2}\.\d{2}[A-Z]?\s+')
_CPF_CLASSIFICATION = re.compile(r'Classification des Produits Française[^\n]*')
_CPF_REV = re.compile(r'CPF\s+r[eé]v\.?\s*[\d.]+\s*\d*')
_PRODUITS_ASSOCIES = re.compile(r'Produits associés\s*:[^\n]*')
_CPF_SUBCODE = re.compile(r'\b\d{2}\.\d{2}\.\d+[a-z]?\b')
_CLASSIF_MARKER = re.compile(r'\b(CC|CA|NC)\s*:\s*')
_CF_REF = re.compile(r'\(cf\.[^)]*\)')
_NEWLINES = re.compile(r'[\r\n]+')
_MULTI_SPACES = re.compile(r' +')


# ─── Étape 1 : nettoyage descriptions ROME ────────────────────────────────────

def clean_rome_desc(text: str) -> str:
    """Extrait les appellations métier lisibles depuis une description ROME brute.

    Le format brut contient des répétitions via le marqueur
    "(code OGR:XXXXX) libelle:". Cette fonction :
      a) Splitpe sur ce pattern
      b) Filtre les segments vides
      c) Déduplique en conservant l'ordre (dict.fromkeys)
      d) Rejoint avec ' | '
      e) Normalise les espaces
    """
    if not text or not isinstance(text, str):
        return ''

    raw_segments = _OGR_SPLIT.split(text)
    # Chaque segment peut contenir " | Titre suivant" hérité du format brut
    parts: list[str] = []
    for seg in raw_segments:
        parts.extend(p.strip() for p in seg.split(' | ') if p.strip())
    parts = list(dict.fromkeys(parts))
    result = ' | '.join(parts)
    return _MULTI_SPACES.sub(' ', result).strip()


# ─── Étape 2 : nettoyage descriptions NAF ────────────────────────────────────

def clean_naf_desc(text: str, code_naf: str = '') -> str:
    """Supprime le bruit classificatoire d'une description NAF brute.

    Conserve le contenu sémantique utile ("Cette sous-classe comprend",
    listes de produits en français, etc.). Les substitutions sont
    appliquées dans un ordre strict.
    """
    if not text or not isinstance(text, str):
        return ''

    # a) Code NAF en début de texte
    text = _NAF_CODE_PREFIX.sub('', text)

    # b) Blocs "Classification des Produits Française..."
    text = _CPF_CLASSIFICATION.sub('', text)

    # c) "CPF rév. X.X NNN"
    text = _CPF_REV.sub('', text)

    # d) Listes "Produits associés : ..."
    text = _PRODUITS_ASSOCIES.sub('', text)

    # e) Sous-codes CPF numériques isolés (ex: 56.10.10)
    text = _CPF_SUBCODE.sub('', text)

    # f) Marqueurs classificatoires : CC :, CA :, NC :
    text = _CLASSIF_MARKER.sub('', text)

    # g) Bullets • → " - "
    text = text.replace('•', ' - ')

    # h) Références (cf. XX.XXZ)
    text = _CF_REF.sub('', text)

    # i) Sauts de ligne multiples → un espace
    text = _NEWLINES.sub(' ', text)

    # j) Espaces multiples
    text = _MULTI_SPACES.sub(' ', text)

    # k) Strip
    return text.strip()


# ─── Étape 3 : construction du texte à encoder ───────────────────────────────

def build_embedding_text(row: pd.Series) -> str:
    """Construit le texte final que le modèle d'embedding encodera.

    - Source ROME (code_rome non null) : name + clean_rome_desc(desc)
    - Source NAF (code_rome null)      : name + clean_naf_desc(desc, code_naf)
    - Limité à 1500 caractères maximum (garde-fou avant tokenisation)
    """
    name = str(row.get('name', '') or '').strip()
    desc = str(row.get('desc', '') or '').strip()
    code_naf = str(row.get('code_naf', '') or '').strip()

    if pd.notna(row.get('code_rome')):
        cleaned = clean_rome_desc(desc)
    else:
        cleaned = clean_naf_desc(desc, code_naf)

    text = f"{name}. {cleaned}" if cleaned else name
    return text[:1500]


# ─── Étape 4 : pipeline complet ──────────────────────────────────────────────

def prepare_corpus(input_csv: str, output_csv: str) -> dict:
    """Charge, nettoie, audite et sauvegarde le corpus.

    Args:
        input_csv: Chemin vers fusion_naf_rome_001_allMiniLM_L6_v2.csv
        output_csv: Chemin de sortie (data/corpus_clean.csv)

    Returns:
        Dictionnaire des métriques calculées.
    """
    start = time.perf_counter()

    # 4a — Chargement ──────────────────────────────────────────────────────────
    logger.info("Chargement de %s …", input_csv)
    df = pd.read_csv(input_csv, dtype=str)
    initial_count = len(df)
    logger.info("  %d lignes chargées.", initial_count)

    # 4b — Déduplication ───────────────────────────────────────────────────────
    df_dedup = df.drop_duplicates(subset=['code_naf', 'code_rome'], keep='first')
    n_duplicates = initial_count - len(df_dedup)
    df = df_dedup.reset_index(drop=True)
    logger.info("  %d doublons supprimés → %d lignes restantes.", n_duplicates, len(df))
    corpus_duplicates_removed.set(n_duplicates)

    # 4b-bis — Codes NAF orphelins (non référencés par aucune ligne ROME) ────────
    naf_codes_all = set(df.loc[df['code_rome'].isna(), 'code_naf'].dropna())
    naf_codes_with_rome = set(df.loc[df['code_rome'].notna(), 'code_naf'].dropna())
    naf_orphan_count = len(naf_codes_all - naf_codes_with_rome)
    naf_coverage_rome_pct = (
        len(naf_codes_all & naf_codes_with_rome) / len(naf_codes_all) * 100
        if naf_codes_all else 0.0
    )
    naf_orphan_codes_gauge.set(naf_orphan_count)
    logger.info(
        "  Codes NAF orphelins (sans ROME) : %d / %d  (couverture ROME : %.1f %%)",
        naf_orphan_count, len(naf_codes_all), naf_coverage_rome_pct,
    )

    # 4c — Nettoyage des descriptions ──────────────────────────────────────────
    logger.info("Nettoyage des descriptions …")
    df['text_to_encode'] = df.apply(build_embedding_text, axis=1)
    logger.info("  %d lignes traitées.", len(df))

    # 4d — Audit qualité ───────────────────────────────────────────────────────
    lengths = df['text_to_encode'].str.len()
    n_rome = int(df['code_rome'].notna().sum())
    n_naf = int(df['code_rome'].isna().sum())
    n_total = len(df)

    mean_len = float(lengths.mean())
    p90 = float(np.percentile(lengths, 90))
    p99 = float(np.percentile(lengths, 99))
    max_len = int(lengths.max())

    overflow_mask = (lengths / 3.5) > 128
    n_overflow = int(overflow_mask.sum())
    overflow_ratio = (n_overflow / n_total * 100) if n_total > 0 else 0.0
    quality_score = (1 - overflow_ratio / 100) * 100

    # Mise à jour Prometheus
    corpus_total_rows.set(n_total)
    corpus_rome_rows.set(n_rome)
    corpus_naf_rows.set(n_naf)
    corpus_text_length_mean.set(mean_len)
    corpus_text_length_p90.set(p90)
    corpus_text_length_p99.set(p99)
    corpus_text_length_max.set(max_len)
    corpus_token_overflow_128.set(n_overflow)
    corpus_token_overflow_128_ratio.set(overflow_ratio)
    preprocessing_quality_score.set(quality_score)

    duration = time.perf_counter() - start
    preprocessing_duration_seconds.observe(duration)

    metrics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'corpus_total_rows': n_total,
        'corpus_duplicates_removed': n_duplicates,
        'corpus_rome_rows': n_rome,
        'corpus_naf_rows': n_naf,
        'corpus_text_length_mean': round(mean_len, 2),
        'corpus_text_length_p90': round(p90, 2),
        'corpus_text_length_p99': round(p99, 2),
        'corpus_text_length_max': max_len,
        'corpus_token_overflow_128': n_overflow,
        'corpus_token_overflow_128_ratio': round(overflow_ratio, 2),
        'quality_score': round(quality_score, 2),
        'naf_orphan_count': naf_orphan_count,
        'naf_coverage_rome_pct': round(naf_coverage_rome_pct, 2),
        'preprocessing_duration_seconds': round(duration, 3),
    }

    # 4e — Sauvegarde ──────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    df[['code_naf', 'code_rome', 'name', 'text_to_encode']].to_csv(output_csv, index=False)
    logger.info("Corpus sauvegardé → %s", output_csv)

    metrics_path = os.path.join(os.path.dirname(os.path.abspath(output_csv)), 'preprocessing_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métriques sauvegardées → %s", metrics_path)

    # 4f — Rapport console ─────────────────────────────────────────────────────
    _print_report(df, metrics, initial_count, n_duplicates)

    return metrics


def _print_report(df: pd.DataFrame, metrics: dict, initial_count: int, n_duplicates: int) -> None:
    """Affiche le rapport récapitulatif dans la console."""
    sep = '─' * 70

    print(f'\n{sep}')
    print('  RAPPORT PREPROCESSING NAFnROME')
    print(sep)
    print(f'  {"Métrique":<40} {"Avant":>10} {"Après":>10}')
    print(sep)
    print(f'  {"Lignes totales":<40} {initial_count:>10} {metrics["corpus_total_rows"]:>10}')
    print(f'  {"Doublons supprimés":<40} {"—":>10} {n_duplicates:>10}')
    print(f'  {"Lignes ROME (code_rome non null)":<40} {"—":>10} {metrics["corpus_rome_rows"]:>10}')
    print(f'  {"Lignes NAF (code_rome null)":<40} {"—":>10} {metrics["corpus_naf_rows"]:>10}')
    print(f'  {"Longueur moyenne (chars)":<40} {"—":>10} {metrics["corpus_text_length_mean"]:>10.1f}')
    print(f'  {"P90 longueur":<40} {"—":>10} {metrics["corpus_text_length_p90"]:>10.1f}')
    print(f'  {"P99 longueur":<40} {"—":>10} {metrics["corpus_text_length_p99"]:>10.1f}')
    print(f'  {"Longueur max":<40} {"—":>10} {metrics["corpus_text_length_max"]:>10}')
    print(f'  {"Lignes > 128 tokens (estimé)":<40} {"—":>10} {metrics["corpus_token_overflow_128"]:>10}')
    print(f'  {"Ratio overflow (%) ← KPI":<40} {"—":>10} {metrics["corpus_token_overflow_128_ratio"]:>10.1f}')
    print(f'  {"Quality score":<40} {"—":>10} {metrics["quality_score"]:>10.1f}')
    print(f'  {"NAF orphelins (sans ROME)":<40} {"—":>10} {metrics["naf_orphan_count"]:>10}')
    print(f'  {"Couverture NAF→ROME (%)":<40} {"—":>10} {metrics["naf_coverage_rome_pct"]:>10.1f}')
    print(f'  {"Durée pipeline (s)":<40} {"—":>10} {metrics["preprocessing_duration_seconds"]:>10.3f}')
    print(sep)

    lengths = df['text_to_encode'].str.len()

    print('\n  TOP 5 textes les plus LONGS :')
    top_long = lengths.nlargest(5).index
    for idx in top_long:
        row = df.loc[idx]
        code = row.get('code_naf', '') or row.get('code_rome', '') or '—'
        txt = row['text_to_encode']
        print(f'    [{code}] len={len(txt):4d}  "{txt[:80]}…"')

    print('\n  TOP 5 textes les plus COURTS :')
    top_short = lengths.nsmallest(5).index
    for idx in top_short:
        row = df.loc[idx]
        code = row.get('code_naf', '') or row.get('code_rome', '') or '—'
        txt = row['text_to_encode']
        print(f'    [{code}] len={len(txt):4d}  "{txt[:80]}"')

    print(f'\n{sep}')
    if metrics['quality_score'] < 85:
        print(f'  ⚠ AVERTISSEMENT : quality_score = {metrics["quality_score"]:.1f} < 85.0')
        print(f'    {metrics["corpus_token_overflow_128"]} lignes dépassent la limite de 128 tokens.')
        print(f'    Envisager un chunking dans src/ingestion.py.')
    else:
        print(f'  OK quality_score = {metrics["quality_score"]:.1f} ≥ 85.0')
    print(sep)

    print('\n  10 premières valeurs de text_to_encode :')
    for i, text in enumerate(df['text_to_encode'].head(10)):
        print(f'    [{i:2d}] {text[:100]}')
    print()


# ─── Point d'entrée ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    prepare_corpus(
        input_csv='fusion_naf_rome_001_allMiniLM_L6_v2.csv',
        output_csv='data/corpus_clean.csv',
    )
