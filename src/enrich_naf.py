"""
enrich_naf.py — Enrichissement des descriptions NAF du corpus.

Croise trois sources pour améliorer la qualité des 732 lignes NAF
de data/corpus_clean.csv :
  1. data/raw/temps/nafs/int_courts_naf_rev_2.xls  ← intitulés officiels INSEE
  2. naf_codes_001_desc.csv                          ← descriptions longues source prof
  3. data/corpus_clean.csv                           ← corpus actuel

Produit : data/corpus_v2.csv (remplace corpus_clean.csv pour la suite).

Chemins configurables via les constantes MODULE-LEVEL ci-dessous.
Si vous déposez un fichier à l'emplacement demandé dans le prompt
(data/naf_insee_intitules.xls, uploads/naf_codes_001_desc.csv),
modifiez simplement les constantes XLS_PATH et DESC_SRC_PATH.
"""

import json
import logging
import os
import re
from datetime import datetime, timezone

import pandas as pd

from src.metrics import (
    naf_avg_text_length_chars,
    naf_coverage_referenced_by_rome,
    naf_enriched_count,
)
from src.preprocessing import clean_naf_desc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)

# ─── Chemins (adapter si besoin) ─────────────────────────────────────────────

XLS_PATH = 'data/raw/temps/nafs/int_courts_naf_rev_2.xls'
DESC_SRC_PATH = 'naf_codes_001_desc.csv'
CORPUS_IN = 'data/corpus_clean.csv'
CORPUS_OUT = 'data/corpus_v2.csv'
INSEE_PARSED_PATH = 'data/naf_insee_parsed.csv'
METRICS_PATH = 'data/enrich_metrics.json'

MAX_CHARS = 600

# Pattern feuille NAF : exactement XX.XXZ (lettre finale obligatoire)
_LEAF_CODE = re.compile(r'^\d{2}\.\d{2}[A-Z]$')


# ─── Utilitaires ─────────────────────────────────────────────────────────────

def _count_significant_words(text: str) -> int:
    """Compte les mots de longueur >= 4 (proxy de richesse sémantique)."""
    return sum(1 for w in text.split() if len(w) >= 4)


def _is_richer(candidate: str, baseline: str) -> bool:
    """Retourne True si candidate est plus long ET plus riche que baseline."""
    return (
        len(candidate) > len(baseline)
        and _count_significant_words(candidate) > _count_significant_words(baseline)
    )


# ─── Étape 1 — Parser le XLS INSEE ──────────────────────────────────────────

def parse_naf_xls(xls_path: str = XLS_PATH) -> pd.DataFrame:
    """Lit le fichier XLS INSEE et extrait les codes feuilles avec leurs intitulés.

    Structure du fichier (int_courts_naf_rev_2.xls) :
      col 0 : numéro de ligne  (ignoré)
      col 1 : code NAF         (sections, divisions, groupes, classes, sous-classes)
      col 2 : intitulé complet (libellé long officiel)
      col 3 : intitulé ≤ 65 chars
      col 4 : intitulé ≤ 40 chars

    Seules les sous-classes (pattern XX.XXZ) sont retenues.

    Returns:
        DataFrame avec colonnes [code_naf, libelle_long, libelle_court]
    """
    logger.info("Lecture du XLS INSEE : %s", xls_path)
    raw = pd.read_excel(xls_path, header=0)  # première ligne = noms de colonnes
    raw.columns = ['ligne', 'code', 'libelle_long', 'libelle_court_65', 'libelle_court_40']

    # Filtrer sur les codes feuilles uniquement
    mask = raw['code'].astype(str).str.match(_LEAF_CODE.pattern, na=False)
    leaves = raw.loc[mask, ['code', 'libelle_long', 'libelle_court_65']].copy()
    leaves.rename(columns={'code': 'code_naf', 'libelle_court_65': 'libelle_court'}, inplace=True)
    leaves = leaves.reset_index(drop=True)

    logger.info("  %d sous-classes NAF extraites (attendu : 732)", len(leaves))

    # Aperçu validation
    print('\n  Aperçu naf_insee_parsed.csv (5 premières lignes) :')
    print(leaves.head().to_string(index=False))
    print()

    leaves.to_csv(INSEE_PARSED_PATH, index=False)
    logger.info("  Sauvegardé → %s", INSEE_PARSED_PATH)

    return leaves


# ─── Étape 2 — Construire les descriptions enrichies ─────────────────────────

def build_enriched_desc(
    insee_df: pd.DataFrame,
    desc_src_path: str = DESC_SRC_PATH,
) -> pd.DataFrame:
    """Construit text_enriched pour chaque code NAF.

    Stratégie :
      text_enriched = libellé_long_insee + ". " + clean_naf_desc(desc_source)
      Si libellé absent : on garde text_to_encode existant (pas de dégradation).
      Limité à MAX_CHARS caractères.

    Args:
        insee_df: DataFrame issu de parse_naf_xls()
        desc_src_path: CSV source avec colonne desc (bruit résiduel à nettoyer)

    Returns:
        DataFrame code_naf → text_enriched
    """
    logger.info("Chargement de la source de descriptions : %s", desc_src_path)
    src = pd.read_csv(desc_src_path, dtype=str)

    # Conserver uniquement les lignes NAF (code_rome null)
    naf_src = src[src['code_rome'].isna()][['code_naf', 'desc']].copy()
    naf_src = naf_src.drop_duplicates('code_naf')
    logger.info("  %d descriptions NAF dans la source", len(naf_src))

    # Nettoyage desc source
    naf_src['desc_clean'] = naf_src.apply(
        lambda row: clean_naf_desc(str(row['desc'] or ''), str(row['code_naf'] or '')),
        axis=1,
    )

    # Jointure avec INSEE
    merged = insee_df.merge(naf_src[['code_naf', 'desc_clean']], on='code_naf', how='left')

    def _build(row: pd.Series) -> str:
        libelle = str(row['libelle_long']).strip() if pd.notna(row['libelle_long']) else ''
        desc = str(row['desc_clean']).strip() if pd.notna(row['desc_clean']) else ''

        if libelle and desc:
            text = f"{libelle}. {desc}"
        elif libelle:
            text = libelle
        else:
            text = desc

        return text[:MAX_CHARS]

    merged['text_enriched'] = merged.apply(_build, axis=1)
    logger.info("  text_enriched construit pour %d codes", len(merged))

    return merged[['code_naf', 'text_enriched']]


# ─── Étape 3 — Mettre à jour le corpus ───────────────────────────────────────

def update_corpus(
    enriched_df: pd.DataFrame,
    corpus_in: str = CORPUS_IN,
    corpus_out: str = CORPUS_OUT,
) -> tuple[dict, pd.DataFrame, dict, pd.Series]:
    """Remplace text_to_encode pour les lignes NAF si l'enrichi est meilleur.

    Règle d'acceptation : text_enriched est plus long ET contient plus de mots
    significatifs (>= 4 lettres) que le text_to_encode actuel.
    Les lignes ROME (code_rome non null) ne sont jamais modifiées.

    Returns:
        (metrics dict, corpus DataFrame, enriched_map dict, initial_naf_lengths Series)
    """
    logger.info("Chargement du corpus : %s", corpus_in)
    corpus = pd.read_csv(corpus_in, dtype=str)
    initial_lengths = corpus['text_to_encode'].str.len().copy()

    naf_mask = corpus['code_rome'].isna()
    rome_mask = ~naf_mask
    logger.info(
        "  %d lignes ROME (intouchées) | %d lignes NAF (candidates)",
        rome_mask.sum(), naf_mask.sum(),
    )

    # Dictionnaire code_naf → text_enriched (évite le reset d'index du merge)
    enriched_map: dict[str, str] = dict(
        zip(enriched_df['code_naf'], enriched_df['text_enriched'])
    )

    n_enriched = 0
    n_unchanged = 0

    # Itérer sur les indices réels des lignes NAF dans le corpus
    for idx in corpus.index[naf_mask]:
        code = str(corpus.at[idx, 'code_naf'] or '').strip()
        candidate = enriched_map.get(code, '')
        baseline = str(corpus.at[idx, 'text_to_encode'] or '').strip()

        if candidate and _is_richer(candidate, baseline):
            corpus.at[idx, 'text_to_encode'] = candidate
            n_enriched += 1
        else:
            n_unchanged += 1

    # ─── Métriques ────────────────────────────────────────────────────────────
    naf_lengths_after = corpus.loc[naf_mask, 'text_to_encode'].str.len()
    avg_len_after = float(naf_lengths_after.mean())

    all_codes_naf = set(corpus.loc[naf_mask, 'code_naf'].dropna())
    codes_linked_to_rome = set(corpus.loc[rome_mask, 'code_naf'].dropna())
    coverage = (
        len(all_codes_naf & codes_linked_to_rome) / len(all_codes_naf) * 100
        if all_codes_naf else 0.0
    )

    naf_enriched_count.set(n_enriched)
    naf_avg_text_length_chars.set(avg_len_after)
    naf_coverage_referenced_by_rome.set(coverage)

    metrics = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'naf_total': int(naf_mask.sum()),
        'naf_enriched_count': n_enriched,
        'naf_unchanged_count': n_unchanged,
        'naf_avg_text_length_before': round(float(initial_lengths[naf_mask].mean()), 2),
        'naf_avg_text_length_after': round(avg_len_after, 2),
        'naf_coverage_referenced_by_rome': round(coverage, 2),
    }

    # ─── Sauvegarde ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(corpus_out)), exist_ok=True)
    corpus[['code_naf', 'code_rome', 'name', 'text_to_encode']].to_csv(corpus_out, index=False)
    logger.info("corpus_v2.csv sauvegardé → %s  (%d lignes)", corpus_out, len(corpus))

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Métriques sauvegardées → %s", METRICS_PATH)

    return metrics, corpus, enriched_map, initial_lengths[naf_mask]


# ─── Étape 4 — Rapport de comparaison ────────────────────────────────────────

def print_report(
    metrics: dict,
    corpus: pd.DataFrame,
    enriched_map: dict,
    initial_naf_lengths: pd.Series,
) -> None:
    """Affiche le rapport comparatif avant/après enrichissement."""
    sep = '─' * 72

    print(f'\n{sep}')
    print('  RAPPORT ENRICHISSEMENT NAF')
    print(sep)
    print(f'  Codes NAF total          : {metrics["naf_total"]}')
    print(f'  Codes enrichis           : {metrics["naf_enriched_count"]}')
    print(f'  Codes inchangés          : {metrics["naf_unchanged_count"]}')
    print(f'  Longueur moy. avant      : {metrics["naf_avg_text_length_before"]:.1f} chars')
    print(f'  Longueur moy. après      : {metrics["naf_avg_text_length_after"]:.1f} chars')
    gain_moy = metrics["naf_avg_text_length_after"] - metrics["naf_avg_text_length_before"]
    sign = '+' if gain_moy >= 0 else ''
    print(f'  Gain moyen               : {sign}{gain_moy:.1f} chars')
    print(f'  NAF couverts par ROME    : {metrics["naf_coverage_referenced_by_rome"]:.1f} %')
    print(sep)

    # ─── TOP 5 les plus améliorés ─────────────────────────────────────────────
    naf_mask = corpus['code_rome'].isna()
    current_lens = corpus.loc[naf_mask, 'text_to_encode'].str.len()
    gain_series = current_lens - initial_naf_lengths
    top5_idx = gain_series.nlargest(5).index

    print('\n  TOP 5 codes les plus améliorés :')
    any_improved = False
    for rank, idx in enumerate(top5_idx, 1):
        g = int(gain_series.get(idx, 0))
        if g == 0:
            continue
        any_improved = True
        code = corpus.at[idx, 'code_naf']
        before_len = int(initial_naf_lengths.get(idx, 0))
        after_text = corpus.at[idx, 'text_to_encode']
        print(f'  [{rank}] {code}  +{g} chars')
        print(f'      AVANT ({before_len:3d}c) : (voir corpus_clean.csv)')
        print(f'      APRÈS ({len(after_text):3d}c) : {after_text[:90]}')
    if not any_improved:
        print('  (aucun code amélioré — sources identiques au corpus existant)')

    # ─── Codes courts — sont-ils corrigés ? ──────────────────────────────────
    print('\n  3 codes avec text_to_encode le plus court dans corpus_v2 :')
    short = corpus.loc[naf_mask].copy()
    short['_len'] = short['text_to_encode'].str.len()
    short3 = short.nsmallest(3, '_len')
    for _, row in short3.iterrows():
        code = row['code_naf']
        candidate = enriched_map.get(code, '')
        current_text = row['text_to_encode']
        print(f'  [{code}] len={row["_len"]:3d}  "{current_text}"')
        if candidate and candidate != current_text:
            print(f'    → candidat disponible mais rejeté ({len(candidate)} chars, '
                  f'pas plus riche) : "{candidate[:80]}"')
        else:
            print(f'    → aucun enrichissement disponible dans les sources')

    print(f'\n{sep}')


# ─── Point d'entrée ──────────────────────────────────────────────────────────

def enrich_corpus(
    xls_path: str = XLS_PATH,
    desc_src_path: str = DESC_SRC_PATH,
    corpus_in: str = CORPUS_IN,
    corpus_out: str = CORPUS_OUT,
) -> dict:
    """Pipeline complet d'enrichissement NAF."""
    # Étape 1
    insee_df = parse_naf_xls(xls_path)

    # Étape 2
    enriched_df = build_enriched_desc(insee_df, desc_src_path)

    # Étape 3
    metrics, corpus, enriched_map, initial_naf_lengths = update_corpus(
        enriched_df, corpus_in, corpus_out
    )

    # Étape 4
    print_report(metrics, corpus, enriched_map, initial_naf_lengths)

    return metrics


if __name__ == '__main__':
    enrich_corpus()
