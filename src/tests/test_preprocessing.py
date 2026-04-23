"""Tests unitaires pour src/preprocessing.py."""

import os

import pandas as pd
import pytest

from src.preprocessing import build_embedding_text, clean_naf_desc, clean_rome_desc

# ─── clean_rome_desc ─────────────────────────────────────────────────────────

def test_clean_rome_removes_ogr_code():
    inp = "Aide agricole (code OGR:10001) libelle:Aide agricole"
    assert clean_rome_desc(inp) == "Aide agricole"


def test_clean_rome_deduplicates():
    inp = "Plombier (code OGR:111) libelle:Plombier"
    assert clean_rome_desc(inp) == "Plombier"


def test_clean_rome_keeps_multiple_distinct():
    inp = "Aide A (code OGR:1) libelle:Aide A | Aide B (code OGR:2) libelle:Aide B"
    assert clean_rome_desc(inp) == "Aide A | Aide B"


# ─── clean_naf_desc ──────────────────────────────────────────────────────────

def test_clean_naf_removes_code_at_start():
    inp = "56.10A Restauration traditionnelle Cette sous-classe comprend les restaurants."
    result = clean_naf_desc(inp)
    assert not result.startswith("56.10A"), f"Le code NAF est encore présent en début : {result!r}"


def test_clean_naf_removes_cpf_subcodes():
    inp = "56.10.10 Services quelconques CC : - truc"
    result = clean_naf_desc(inp)
    assert "56.10.10" not in result, f"Le sous-code CPF est encore présent : {result!r}"


def test_clean_naf_removes_cc_marker():
    inp = "Texte CC : - élément 1 - élément 2"
    result = clean_naf_desc(inp)
    assert "CC :" not in result, f"Le marqueur CC : est encore présent : {result!r}"


def test_clean_naf_removes_nc_marker():
    inp = "Description NC : quelque chose NC : autre chose"
    result = clean_naf_desc(inp)
    assert "NC :" not in result, f"Le marqueur NC : est encore présent : {result!r}"


def test_clean_naf_preserves_semantic_content():
    inp = "56.10A Cette sous-classe comprend les restaurants traditionnels."
    result = clean_naf_desc(inp)
    assert "Cette sous-classe comprend" in result


# ─── build_embedding_text ─────────────────────────────────────────────────────

def test_build_embedding_text_max_length():
    row = pd.Series({
        'name': 'Métier test',
        'desc': 'X' * 5000,
        'code_naf': '01.11Z',
        'code_rome': None,
    })
    result = build_embedding_text(row)
    assert len(result) <= 1500, f"Texte trop long : {len(result)} chars"


def test_build_embedding_text_rome_source():
    row = pd.Series({
        'name': 'Conducteur',
        'desc': 'Chauffeur (code OGR:11987) libelle:Chauffeur',
        'code_naf': '01.61Z',
        'code_rome': 'A1101',
    })
    result = build_embedding_text(row)
    assert 'code OGR' not in result
    assert 'Conducteur' in result
    assert 'Chauffeur' in result


def test_build_embedding_text_naf_source():
    row = pd.Series({
        'name': 'Restauration',
        'desc': '56.10A Restauration traditionnelle Cette sous-classe comprend les restaurants.',
        'code_naf': '56.10A',
        'code_rome': None,
    })
    result = build_embedding_text(row)
    assert not result.startswith('56.10A')
    assert 'Restauration' in result


# ─── Test sur données réelles ─────────────────────────────────────────────────

REAL_CSV = 'fusion_naf_rome_001_allMiniLM_L6_v2.csv'


@pytest.mark.skipif(
    not os.path.exists(REAL_CSV),
    reason=f"{REAL_CSV} absent — test ignoré hors environnement de données",
)
def test_no_empty_text_to_encode():
    df = pd.read_csv(REAL_CSV, dtype=str)
    df['text_to_encode'] = df.apply(build_embedding_text, axis=1)

    empty_mask = df['text_to_encode'].str.len() < 5
    n_empty = int(empty_mask.sum())
    if n_empty > 0:
        samples = df.loc[empty_mask, ['code_naf', 'code_rome', 'name', 'text_to_encode']].head(5).to_string()
        pytest.fail(f"{n_empty} lignes avec text_to_encode < 5 chars :\n{samples}")
