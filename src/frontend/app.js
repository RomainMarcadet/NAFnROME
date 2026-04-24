/**
 * app.js — Logique SPA NAFnROME
 * API_BASE = '' (même origine, FastAPI sert le front)
 */

'use strict';

// ─── État global ──────────────────────────────────────────────────────────────

const state = {
  mode: 'sophie',      // 'sophie' | 'karim'
  boost: true,
  lastQuery: '',
  lastResults: [],
  lastMeta: null,
  detailOpen: false,
  currentDetail: null,
};

// ─── Raccourcis DOM ───────────────────────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ─── Utilitaires ──────────────────────────────────────────────────────────────

function scoreLabel(score) {
  const pct = score * 100;
  if (pct >= 80) return { label: 'HAUTE',   cls: 'badge--high' };
  if (pct >= 60) return { label: 'MOYENNE', cls: 'badge--medium' };
  return           { label: 'FAIBLE',  cls: 'badge--low' };
}

function fmtPct(score) {
  return (score * 100).toFixed(1) + '\u202f%';
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ─── Footer ───────────────────────────────────────────────────────────────────

function updateFooter(latencyMs) {
  if (latencyMs != null) {
    $('#footer-latency').textContent = latencyMs.toFixed(1) + '\u202fms';
  }
}

async function fetchHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) return;
    const data = await res.json();
    const count = data.collection_count;
    $('#footer-count').textContent =
      (count ? count.toLocaleString('fr-FR') : '—') + '\u00a0entrées · collection naf_rome_v2';
    const model = data.model || 'paraphrase-multilingual-MiniLM-L12-v2';
    $('#footer-model').textContent = model + '\u00a0· local';
  } catch (e) {
    // silencieux — l'API peut ne pas être démarrée
  }
}

// ─── Loader ───────────────────────────────────────────────────────────────────

function showLoader() {
  const loader = $('#loader');
  loader.hidden = false;
  loader.setAttribute('aria-label', 'Recherche en cours…');
}

function hideLoader() {
  const loader = $('#loader');
  loader.hidden = true;
}

// ─── Recherche ────────────────────────────────────────────────────────────────

async function search(query) {
  query = query.trim();
  if (!query) return;

  state.lastQuery = query;
  showLoader();
  clearResults();

  const n = state.mode === 'sophie' ? 5 : 3;

  try {
    const res = await fetch('/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, n, boost: state.boost }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Erreur serveur' }));
      showError(err.detail || 'Erreur lors de la recherche.');
      return;
    }

    const data = await res.json();
    state.lastResults = data.results || [];
    state.lastMeta    = data.meta || {};

    renderResults(state.lastResults, data.meta);
    updateFooter(data.meta.latency_ms);
  } catch (e) {
    showError('Impossible de joindre le serveur. Vérifiez que FastAPI est démarré.');
  } finally {
    hideLoader();
  }
}

function clearResults() {
  const zone = $('#results');
  zone.innerHTML = '';
}

function showError(msg) {
  const zone = $('#results');
  zone.innerHTML = `
    <div style="text-align:center; padding: 40px 24px;">
      <p style="font-family:var(--font-mono); font-size:0.78rem; color:var(--danger);">
        ${escHtml(msg)}
      </p>
    </div>`;
}

// ─── Rendu résultats ──────────────────────────────────────────────────────────

function renderResults(results, meta) {
  const zone = $('#results');
  zone.innerHTML = '';

  if (!results.length) {
    zone.innerHTML = `
      <div class="results__empty">
        <p class="results__empty-text">Aucun résultat pour « ${escHtml(state.lastQuery)} ».</p>
      </div>`;
    return;
  }

  const boostApplied = meta && meta.boost_applied;
  const chunkInfo = meta
    ? `${meta.chunks_after_dedup} docs · ${meta.chunks_fetched_before_dedup} chunks analysés`
    : '';

  const header = document.createElement('div');
  header.className = 'results__header';
  header.innerHTML = `
    <span class="results__title">${results.length} résultat${results.length > 1 ? 's' : ''} · mode ${state.mode === 'sophie' ? 'Sophie' : 'Karim'}</span>
    <span class="results__meta">${escHtml(chunkInfo)}</span>`;
  zone.appendChild(header);

  const list = document.createElement('div');
  list.className = 'results__list';

  if (state.mode === 'sophie') {
    results.forEach(r => list.appendChild(buildSophieCard(r, boostApplied)));
  } else {
    results.forEach(r => list.appendChild(buildKarimCard(r)));
  }

  zone.appendChild(list);
}

// ─── Carte Sophie ─────────────────────────────────────────────────────────────

function buildSophieCard(r, boostApplied) {
  const { label, cls } = scoreLabel(r.score);
  const boostDiff = boostApplied && Math.abs(r.score - r.score_raw) > 0.001;

  const card = document.createElement('article');
  card.className = 'card-sophie';
  card.setAttribute('aria-label', `Résultat ${r.rank} : ${r.name}`);

  card.innerHTML = `
    <span class="card-sophie__rank">#${r.rank}</span>

    <div class="card-sophie__body">
      <div class="card-sophie__codes">
        <span class="card-sophie__code" title="Code ROME">${escHtml(r.code_rome)}</span>
        <span class="card-sophie__code" style="color:var(--accent)" title="Code NAF">· ${escHtml(r.code_naf)}</span>
      </div>
      <div class="card-sophie__name" title="${escHtml(r.name)}">${escHtml(r.name)}</div>
      <div class="card-sophie__naf-label">
        Famille&nbsp;<span style="color:var(--sage); font-family:var(--font-mono)">${escHtml(r.famille)}</span>
        &nbsp;· Chunk ${r.chunk_idx + 1}/${r.n_chunks}
      </div>
      ${boostDiff ? `
      <div class="card-sophie__scores">
        <span class="card-sophie__score-raw">Brut : ${fmtPct(r.score_raw)} → Boosté : ${fmtPct(r.score)}</span>
      </div>` : ''}
    </div>

    <div class="card-sophie__aside">
      <span class="badge ${cls} badge--score" aria-label="Niveau de confiance : ${label}">${label} · ${fmtPct(r.score)}</span>
      <div class="card-sophie__actions">
        <button class="btn btn--primary" data-action="detail" data-rome="${escHtml(r.code_rome)}" data-idx="${r.rank - 1}" aria-label="Voir le détail de ${escHtml(r.code_rome)}">Détail</button>
        <button class="btn btn--secondary" data-action="copy" data-text="${escHtml(r.code_naf + ' · ' + r.code_rome)}" aria-label="Copier les codes ${escHtml(r.code_naf)} et ${escHtml(r.code_rome)}">Copier</button>
      </div>
    </div>`;

  card.addEventListener('click', handleCardAction);
  return card;
}

// ─── Carte Karim ─────────────────────────────────────────────────────────────

function buildKarimCard(r) {
  const pct = (r.score * 100).toFixed(1);

  const card = document.createElement('article');
  card.className = 'card-karim';
  card.setAttribute('aria-label', `Résultat ${r.rank} : ${r.name}`);

  const uid = `why-${r.rank}`;

  card.innerHTML = `
    <div class="card-karim__name">${escHtml(r.name)}</div>
    <div class="card-karim__sector">Secteur : <span id="sector-${r.rank}">chargement…</span></div>

    <div class="card-karim__score-bar-wrap" aria-label="Score de correspondance : ${pct}%">
      <div class="card-karim__score-bar" role="progressbar" aria-valuenow="${pct}" aria-valuemin="0" aria-valuemax="100">
        <div class="card-karim__score-fill" style="width:0%" data-target="${pct}"></div>
      </div>
      <span class="card-karim__score-pct">${pct}\u202f%</span>
    </div>

    <div class="card-karim__codes-secondary">
      ROME&nbsp;<strong>${escHtml(r.code_rome)}</strong> &middot; NAF&nbsp;<strong>${escHtml(r.code_naf)}</strong>
    </div>

    <div class="card-karim__actions">
      <button class="btn btn--primary" data-action="detail" data-rome="${escHtml(r.code_rome)}" data-idx="${r.rank - 1}" aria-label="Voir le détail de ${escHtml(r.code_rome)}">Détail</button>
      <button class="btn btn--secondary" data-action="copy" data-text="${escHtml(r.code_naf)}" aria-label="Copier le code NAF ${escHtml(r.code_naf)}">Copier NAF</button>
      <button class="btn btn--ghost" data-action="why" data-uid="${uid}" aria-expanded="false" aria-controls="${uid}">Pourquoi ?</button>
    </div>

    <div class="card-karim__why" id="${uid}" aria-hidden="true">
      <div class="card-karim__why-inner">
        <div class="card-karim__why-title">Correspondance sémantique</div>
        <div class="why-content">Famille ROME <strong>${escHtml(r.famille)}</strong> · chunk ${r.chunk_idx + 1}/${r.n_chunks}<br>
        Score brut : ${fmtPct(r.score_raw)}</div>
      </div>
    </div>`;

  card.addEventListener('click', handleCardAction);

  // Animer la barre après injection dans le DOM
  requestAnimationFrame(() => {
    const fill = card.querySelector('.card-karim__score-fill');
    if (fill) {
      setTimeout(() => { fill.style.width = fill.dataset.target + '%'; }, 80);
    }
  });

  // Charger le libellé NAF depuis /rome/{code}
  fetchNafLabel(r.code_rome, card.querySelector(`#sector-${r.rank}`));

  return card;
}

async function fetchNafLabel(codeRome, el) {
  if (!el) return;
  try {
    const res = await fetch(`/rome/${encodeURIComponent(codeRome)}`);
    if (!res.ok) { el.textContent = '—'; return; }
    const data = await res.json();
    const nafs = data.naf_codes || [];
    el.textContent = nafs.length ? nafs.slice(0, 2).join(', ') : '—';
  } catch {
    el.textContent = '—';
  }
}

// ─── Gestion des actions sur cartes ──────────────────────────────────────────

function handleCardAction(e) {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;

  const action = btn.dataset.action;

  if (action === 'detail') {
    const idx = parseInt(btn.dataset.idx, 10);
    openDetail(btn.dataset.rome, idx);
  }

  if (action === 'copy') {
    copyToClipboard(btn.dataset.text, btn);
  }

  if (action === 'why') {
    const uid = btn.dataset.uid;
    const panel = document.getElementById(uid);
    const open = panel.classList.toggle('card-karim__why--open');
    panel.setAttribute('aria-hidden', String(!open));
    btn.setAttribute('aria-expanded', String(open));
  }
}

// ─── Presse-papier ────────────────────────────────────────────────────────────

async function copyToClipboard(text, btn) {
  try {
    await navigator.clipboard.writeText(text);
    const orig = btn.textContent;
    btn.textContent = 'Copié !';
    btn.style.color = 'var(--highlight)';
    setTimeout(() => {
      btn.textContent = orig;
      btn.style.color = '';
    }, 1000);
  } catch {
    btn.textContent = 'Erreur';
    setTimeout(() => { btn.textContent = 'Copier'; }, 1500);
  }
}

// ─── Panneau détail ───────────────────────────────────────────────────────────

async function openDetail(codeRome, resultIdx) {
  const panel   = $('#detail-panel');
  const overlay = $('#panel-overlay');

  // Pré-remplir depuis les résultats en mémoire
  const r = state.lastResults[resultIdx];
  if (r) {
    populatePanelFromResult(r);
  }

  // Ouvrir immédiatement
  panel.classList.add('open');
  panel.setAttribute('aria-hidden', 'false');
  overlay.classList.add('panel-overlay--visible');
  overlay.setAttribute('aria-hidden', 'false');
  state.detailOpen = true;

  // Focus sur le bouton fermer
  $('#btn-panel-close').focus();

  // Compléter avec données /rome/{code}
  try {
    const res = await fetch(`/rome/${encodeURIComponent(codeRome)}`);
    if (res.ok) {
      const data = await res.json();
      enrichPanelFromApi(data);
    }
  } catch { /* silencieux */ }
}

function populatePanelFromResult(r) {
  $('#panel-code-rome').textContent = r.code_rome;
  $('#panel-code-naf').textContent  = r.code_naf;
  $('#panel-name').textContent      = r.name;
  $('#panel-famille').textContent   = r.famille + ' — Famille ROME';
  $('#panel-score-raw').textContent = fmtPct(r.score_raw);
  $('#panel-score').textContent     = fmtPct(r.score) + (state.boost ? ' (boosté)' : '');
  $('#panel-chunk').textContent     = `${r.chunk_idx + 1} / ${r.n_chunks}`;
  $('#panel-appellations').textContent = '—';

  // Liens
  $('#link-ft').href    = `https://candidat.francetravail.fr/metierscope/metier-detail?codeRome=${encodeURIComponent(r.code_rome)}`;
  $('#link-insee').href = `https://www.insee.fr/fr/metadonnees/nafr2/sousClasse/${encodeURIComponent(r.code_naf)}`;

  // NAF provisoire
  const naf = $('#panel-naf-list');
  naf.innerHTML = `<span class="naf-tag">${escHtml(r.code_naf)}</span>`;
}

function enrichPanelFromApi(data) {
  if (data.appellations_count != null) {
    $('#panel-appellations').textContent = data.appellations_count;
  }
  const naf = $('#panel-naf-list');
  if (data.naf_codes && data.naf_codes.length) {
    naf.innerHTML = data.naf_codes
      .map(c => `<span class="naf-tag">${escHtml(c)}</span>`)
      .join('');
  }
}

function closeDetail() {
  const panel   = $('#detail-panel');
  const overlay = $('#panel-overlay');
  panel.classList.remove('open');
  panel.setAttribute('aria-hidden', 'true');
  overlay.classList.remove('panel-overlay--visible');
  overlay.setAttribute('aria-hidden', 'true');
  state.detailOpen = false;
}

// ─── Export CSV ───────────────────────────────────────────────────────────────

function exportCSV() {
  if (!state.lastResults.length) return;
  const header = 'rank,code_naf,code_rome,name,score,score_raw';
  const rows = state.lastResults.map(r =>
    `${r.rank},${r.code_naf},${r.code_rome},"${r.name.replace(/"/g, '""')}",${r.score},${r.score_raw}`
  );
  const csv  = header + '\n' + rows.join('\n');
  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `naf_rome_${state.lastQuery.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '')}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ─── Toggle boost ─────────────────────────────────────────────────────────────

function toggleBoost() {
  state.boost = !state.boost;
  if (state.lastQuery) search(state.lastQuery);
}

// ─── Toggle mode ─────────────────────────────────────────────────────────────

function setMode(mode) {
  state.mode = mode;

  // Mise à jour des boutons
  $$('.mode-toggle__btn').forEach(btn => {
    const active = btn.dataset.mode === mode;
    btn.classList.toggle('mode-toggle__btn--active', active);
    btn.setAttribute('aria-pressed', String(active));
  });

  // Mise à jour du placeholder
  const input = $('#search-input');
  if (mode === 'sophie') {
    input.placeholder = 'Code ROME ou appellation… ex\u202f: G1209';
  } else {
    input.placeholder = 'Décrivez le métier… ex\u202f: développeur Java fintech';
  }

  // Relancer si query présente
  if (state.lastQuery) search(state.lastQuery);
}

// ─── Focus trap (aside) ───────────────────────────────────────────────────────

function getFocusables(container) {
  return Array.from(container.querySelectorAll(
    'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
  )).filter(el => !el.closest('[aria-hidden="true"]') && el.offsetParent !== null);
}

function trapFocus(e) {
  if (!state.detailOpen) return;
  const panel     = $('#detail-panel');
  const focusable = getFocusables(panel);
  if (!focusable.length) return;

  const first = focusable[0];
  const last  = focusable[focusable.length - 1];

  if (e.key === 'Tab') {
    if (e.shiftKey) {
      if (document.activeElement === first) {
        e.preventDefault();
        last.focus();
      }
    } else {
      if (document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  }
}

// ─── Init ─────────────────────────────────────────────────────────────────────

function initSearch() {
  const input  = $('#search-input');
  const btnSrc = $('#btn-search');

  btnSrc.addEventListener('click', () => search(input.value));

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      search(input.value);
    }
  });
}

function initToggle() {
  // Mode toggle
  $$('.mode-toggle__btn').forEach(btn => {
    btn.addEventListener('click', () => setMode(btn.dataset.mode));
  });

  // Boost toggle
  const boostInput = $('#toggle-boost');
  boostInput.addEventListener('change', toggleBoost);
}

function initChips() {
  $$('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const query = chip.dataset.query;
      const input = $('#search-input');
      input.value = query;
      search(query);
    });
  });
}

function initPanel() {
  // Fermeture
  $('#btn-panel-close').addEventListener('click', closeDetail);
  $('#panel-overlay').addEventListener('click', closeDetail);

  // Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.detailOpen) closeDetail();
    trapFocus(e);
  });

  // Export CSV depuis le panneau
  $('#btn-export-csv').addEventListener('click', exportCSV);
}

// ─── Bootstrap ────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initToggle();
  initSearch();
  initChips();
  initPanel();
  fetchHealth();

  // Placer le focus sur l'input au chargement
  $('#search-input').focus();
});
