'use strict';

const state = {
  boost: true,
  persona: 'recruteur',
  lastQuery: '',
  lastResults: [],
  lastMeta: null,
  detailOpen: false,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function fmtPct(score) {
  return (score * 100).toFixed(1) + '\u202f%';
}

function scoreLabel(score) {
  const pct = score * 100;
  if (pct >= 80) return { label: 'Très probable', cls: 'badge--high' };
  if (pct >= 60) return { label: 'Pertinent', cls: 'badge--medium' };
  return { label: 'À vérifier', cls: 'badge--low' };
}

function updateFooter(latencyMs) {
  if (latencyMs != null) {
    const label = `${latencyMs.toFixed(1)}\u202fms`;
    $('#footer-latency').textContent = `Latence : ${label}`;
    $('#hero-latency').textContent = label;
  }
}

function updatePageMeta(query) {
  const title = query
    ? `JOB | Résultats pour "${query}"`
    : 'JOB | Recherche ROME et NAF';
  document.title = title;
}

function setEngineStatus(label) {
  $('#engine-status').textContent = label;
}

async function fetchHealth() {
  try {
    const res = await fetch('/health');
    if (!res.ok) return;
    const data = await res.json();
    const count = data.collection_count || 0;
    const countLabel = count ? count.toLocaleString('fr-FR') : '—';
    const model = data.model || 'paraphrase-multilingual-MiniLM-L12-v2';

    $('#footer-count').textContent = `Base : ${countLabel} entrées`;
    $('#footer-model').textContent = `Modèle : ${model}`;
    $('#hero-count').textContent = `${countLabel} entrées`;
    $('#hero-model').textContent = model;
    setEngineStatus('Prêt');
  } catch {
    setEngineStatus('Indisponible');
  }
}

function showLoader() {
  const loader = $('#loader');
  loader.hidden = false;
  setEngineStatus('Analyse');
}

function hideLoader() {
  const loader = $('#loader');
  loader.hidden = true;
  setEngineStatus('Prêt');
}

function clearResults() {
  $('#results').innerHTML = '';
}

function updateResultsMeta(summary, metaText) {
  $('#results-summary').textContent = summary;
  $('#results-meta').textContent = metaText;
}

function confidenceCopy(score) {
  const pct = score * 100;
  if (pct >= 80) {
    return {
      title: 'Correspondance très crédible',
      text: 'Le moteur place ce résultat en tête avec un niveau de proximité élevé. Dans la plupart des cas, c’est un bon point de départ pour valider rapidement.',
    };
  }
  if (pct >= 60) {
    return {
      title: 'Correspondance plausible',
      text: 'Le résultat est cohérent avec la requête, mais il mérite une vérification métier avant validation finale, surtout si plusieurs activités se ressemblent.',
    };
  }
  return {
    title: 'Correspondance à confirmer',
    text: 'Le moteur voit une proximité partielle. Utilisez ce résultat comme piste, puis comparez avec les alternatives proposées avant de trancher.',
  };
}

function personaInsights(meta, results) {
  if (!results.length || !meta) {
    return [
      {
        label: 'Priorité',
        value: 'Lancez une recherche',
        text: 'Les indicateurs se personnalisent après la première requête.',
      },
      {
        label: 'Lecture',
        value: 'Aucun signal',
        text: 'Le moteur expliquera ici la qualité et la stabilité du résultat.',
      },
      {
        label: 'Action',
        value: 'En attente',
        text: 'Vous verrez ici quoi faire ensuite selon le contexte du résultat.',
      },
    ];
  }

  const top = results[0];
  const confidence = confidenceCopy(top.score);
  const stable = results.length > 1 ? Math.max((top.score - results[1].score) * 100, 0) : top.score * 100;
  const stableLabel = stable >= 8 ? 'Écart net' : stable >= 4 ? 'Écart moyen' : 'Écart serré';

  if (state.persona === 'recruteur') {
    return [
      {
        label: 'Priorité',
        value: top.name,
        text: `Le meilleur rapprochement combine ${top.code_rome} et ${top.code_naf}. ${confidence.text}`,
      },
      {
        label: 'Sécurité',
        value: stableLabel,
        text: `L’écart avec le résultat suivant est de ${stable.toFixed(1)} points. Plus l’écart est fort, plus le premier résultat est simple à défendre.`,
      },
      {
        label: 'Action',
        value: 'Ouvrir la fiche',
        text: 'Vérifiez les codes associés et la description NAF avant de réutiliser le résultat dans un dossier ou un workflow.',
      },
    ];
  }

  if (state.persona === 'analyste') {
    return [
      {
        label: 'Signal',
        value: `${fmtPct(top.score)} final`,
        text: `Le score initial est ${fmtPct(top.score_raw)}${meta.boost_applied ? `, puis ajusté à ${fmtPct(top.score)}` : ''}.`,
      },
      {
        label: 'Couverture',
        value: `${meta.chunks_after_dedup} candidats`,
        text: `${meta.chunks_fetched_before_dedup} segments ont été comparés pour construire le classement et éliminer les doublons.`,
      },
      {
        label: 'Action',
        value: `${meta.query_token_count} tokens`,
        text: 'Si le résultat semble trop large, reformulez la requête avec un intitulé métier plus précis ou un contexte sectoriel.',
      },
    ];
  }

  return [
    {
      label: 'Décision',
      value: confidence.title,
      text: `Le premier résultat atteint ${fmtPct(top.score)}. C’est le niveau de fiabilité actuellement proposé pour orienter la décision.`,
    },
    {
      label: 'Vitesse',
      value: `${meta.latency_ms} ms`,
      text: 'Le temps de réponse mesuré te permet de juger la fluidité de l’outil dans un usage réel.',
    },
    {
      label: 'Action',
      value: meta.boost_applied ? 'Classement priorisé' : 'Classement neutre',
      text: meta.boost_applied
        ? 'Le moteur applique une priorisation métier. Désactive-la si tu veux une lecture plus brute du classement.'
        : 'Le classement est présenté sans priorisation métier supplémentaire.',
    },
  ];
}

function renderPersonaInsights(meta, results) {
  const insights = personaInsights(meta, results);
  insights.forEach((item, idx) => {
    $(`#insight-${idx + 1}-label`).textContent = item.label;
    $(`#insight-${idx + 1}-value`).textContent = item.value;
    $(`#insight-${idx + 1}-text`).textContent = item.text;
  });
}

function showError(msg) {
  $('#results').innerHTML = `
    <div class="results__empty">
      <p class="results__empty-title">Recherche indisponible</p>
      <p class="results__empty-text">${escHtml(msg)}</p>
    </div>`;
  updateResultsMeta('Erreur de recherche', 'Vérifie que l’API est démarrée');
  renderPersonaInsights(null, []);
}

async function search(query) {
  query = query.trim();
  if (!query) return;

  state.lastQuery = query;
  showLoader();
  clearResults();
  updatePageMeta(query);
  updateResultsMeta(`Analyse en cours pour « ${query} »`, 'Interrogation du moteur');
  renderPersonaInsights(null, []);

  try {
    const res = await fetch('/match', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, n: 5, boost: state.boost }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Erreur serveur' }));
      showError(err.detail || 'Erreur lors de la recherche.');
      return;
    }

    const data = await res.json();
    state.lastResults = data.results || [];
    state.lastMeta = data.meta || {};
    renderResults(state.lastResults, state.lastMeta);
    updateFooter(data.meta.latency_ms);
    renderPersonaInsights(state.lastMeta, state.lastResults);
  } catch {
    showError('Impossible de joindre le moteur pour le moment.');
  } finally {
    hideLoader();
  }
}

function buildResultSummary(r, boostApplied) {
  const parts = [
    `Code ROME ${escHtml(r.code_rome)}`,
    `code NAF ${escHtml(r.code_naf)}`,
    `famille ${escHtml(r.famille || '—')}`,
  ];

  if (boostApplied && Math.abs(r.score - r.score_raw) > 0.001) {
    parts.push(`score ajusté depuis ${fmtPct(r.score_raw)}`);
  }

  return parts.join(' · ');
}

function renderResults(results, meta) {
  const zone = $('#results');
  zone.innerHTML = '';

  if (!results.length) {
    zone.innerHTML = `
      <div class="results__empty">
        <p class="results__empty-title">Aucune correspondance utile trouvée</p>
        <p class="results__empty-text">Essaie une formulation plus précise ou un métier voisin pour élargir le rapprochement sémantique.</p>
      </div>`;
    updateResultsMeta(`0 résultat pour « ${state.lastQuery} »`, 'Ajuste la formulation de la requête');
    return;
  }

  const boostApplied = meta && meta.boost_applied;
  const summary = `${results.length} résultat${results.length > 1 ? 's' : ''} pour « ${state.lastQuery} »`;
  const metaText = meta
    ? `${meta.chunks_after_dedup} correspondances uniques · ${meta.chunks_fetched_before_dedup} segments comparés`
    : 'Classement sémantique';
  updateResultsMeta(summary, metaText);

  const list = document.createElement('div');
  list.className = 'results__list';
  results.forEach((r) => list.appendChild(buildResultCard(r, boostApplied)));
  zone.appendChild(list);
}

function buildResultCard(r, boostApplied) {
  const { label, cls } = scoreLabel(r.score);
  const card = document.createElement('article');
  card.className = 'result-card';
  card.setAttribute('aria-label', `Résultat ${r.rank} : ${r.name}`);

  const boostDiff = boostApplied && Math.abs(r.score - r.score_raw) > 0.001;
  const whyBlock = boostDiff
    ? `Le classement final intègre une priorisation métier : ${fmtPct(r.score_raw)} → ${fmtPct(r.score)}.`
    : `Le classement repose directement sur la proximité sémantique mesurée par le moteur.`;

  card.innerHTML = `
    <div class="result-card__top">
      <div class="result-card__content">
        <div class="result-card__eyebrow">
          <span class="result-card__rank">#${r.rank}</span>
          <span class="result-card__code">ROME ${escHtml(r.code_rome)}</span>
          <span class="result-card__code">NAF ${escHtml(r.code_naf)}</span>
        </div>
        <h3 class="result-card__title">${escHtml(r.name)}</h3>
        <p class="result-card__summary">${buildResultSummary(r, boostApplied)}</p>
      </div>

      <div class="result-card__aside">
        <span class="badge ${cls}">${label}</span>
        <span class="result-card__score">${fmtPct(r.score)} de confiance</span>
      </div>
    </div>

    <div class="result-card__meta">
      <span class="meta-pill">Famille ${escHtml(r.famille || '—')}</span>
      <span class="meta-pill">Segment ${r.chunk_idx + 1}/${r.n_chunks}</span>
      <span class="meta-pill">Score initial ${fmtPct(r.score_raw)}</span>
      <span class="meta-pill">Score final ${fmtPct(r.score)}</span>
    </div>

    <div class="result-card__actions">
      <button class="btn btn--primary" data-action="detail" data-rome="${escHtml(r.code_rome)}" data-idx="${r.rank - 1}">
        Ouvrir la fiche
      </button>
      <button class="btn btn--secondary" data-action="copy" data-text="${escHtml(`${r.code_naf} · ${r.code_rome}`)}">
        Copier les codes
      </button>
      <button class="btn btn--ghost" data-action="why" data-why="${escHtml(whyBlock)}">
        Pourquoi ce résultat ?
      </button>
    </div>`;

  card.addEventListener('click', handleCardAction);
  return card;
}

function handleCardAction(e) {
  const btn = e.target.closest('[data-action]');
  if (!btn) return;

  const action = btn.dataset.action;

  if (action === 'detail') {
    const idx = parseInt(btn.dataset.idx, 10);
    openDetail(btn.dataset.rome, idx);
    return;
  }

  if (action === 'copy') {
    copyToClipboard(btn.dataset.text, btn);
    return;
  }

  if (action === 'why') {
    const existing = btn.closest('.result-card').querySelector('.result-card__why');
    if (existing) {
      existing.remove();
      btn.textContent = 'Pourquoi ce résultat ?';
      return;
    }

    const why = document.createElement('div');
    why.className = 'result-card__why';
    why.innerHTML = `<strong>Lecture du score.</strong> ${btn.dataset.why}`;
    btn.closest('.result-card').appendChild(why);
    btn.textContent = 'Masquer l’explication';
  }
}

async function copyToClipboard(text, btn) {
  const original = btn.textContent;
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = 'Copié';
  } catch {
    btn.textContent = 'Impossible';
  }

  setTimeout(() => {
    btn.textContent = original;
  }, 1100);
}

async function openDetail(codeRome, resultIdx) {
  const panel = $('#detail-panel');
  const overlay = $('#panel-overlay');
  const result = state.lastResults[resultIdx];

  if (result) {
    populatePanelFromResult(result);
  }

  panel.classList.add('open');
  panel.setAttribute('aria-hidden', 'false');
  overlay.classList.add('panel-overlay--visible');
  overlay.setAttribute('aria-hidden', 'false');
  state.detailOpen = true;
  $('#btn-panel-close').focus();

  try {
    const res = await fetch(`/rome/${encodeURIComponent(codeRome)}`);
    if (!res.ok) return;
    const data = await res.json();
    enrichPanelFromApi(data);
  } catch {
    // silencieux
  }
}

function populatePanelFromResult(r) {
  const confidence = confidenceCopy(r.score);
  $('#panel-code-rome').textContent = `ROME ${r.code_rome}`;
  $('#panel-code-naf').textContent = `NAF ${r.code_naf}`;
  $('#panel-name').textContent = r.name;
  $('#panel-famille').textContent = r.famille || '—';
  $('#panel-score-raw').textContent = fmtPct(r.score_raw);
  $('#panel-score').textContent = state.boost ? `${fmtPct(r.score)} (priorisé)` : fmtPct(r.score);
  $('#panel-chunk').textContent = `${r.chunk_idx + 1} / ${r.n_chunks}`;
  $('#panel-appellations').textContent = '—';
  $('#panel-confidence-title').textContent = confidence.title;
  $('#panel-confidence-text').textContent = confidence.text;

  $('#link-ft').href =
    `https://candidat.francetravail.fr/metierscope/metier-detail?codeRome=${encodeURIComponent(r.code_rome)}`;
  $('#link-insee').href =
    `https://www.insee.fr/fr/metadonnees/nafr2/sousClasse/${encodeURIComponent(r.code_naf)}`;

  $('#panel-naf-list').innerHTML = `<span class="naf-tag">${escHtml(r.code_naf)}</span>`;
  $('#panel-naf-descriptions').innerHTML = '<p class="naf-description-empty">Chargement de la description NAF…</p>';
}

function enrichPanelFromApi(data) {
  if (data.appellations_count != null) {
    $('#panel-appellations').textContent = data.appellations_count;
  }

  if (data.naf_codes && data.naf_codes.length) {
    $('#panel-naf-list').innerHTML = data.naf_codes
      .map((code) => `<span class="naf-tag">${escHtml(code)}</span>`)
      .join('');
  }

  if (data.naf_details && data.naf_details.length) {
    const withDescriptions = data.naf_details.filter((item) => item.description);
    $('#panel-naf-descriptions').innerHTML = withDescriptions.length
      ? withDescriptions.map((item) => `
        <article class="naf-description-card">
          <strong class="naf-description-card__code">${escHtml(item.code_naf)}</strong>
          <p class="naf-description-card__text">${escHtml(item.description)}</p>
        </article>`).join('')
      : '<p class="naf-description-empty">Aucune description NAF détaillée disponible pour ce résultat.</p>';
  }
}

function closeDetail() {
  $('#detail-panel').classList.remove('open');
  $('#detail-panel').setAttribute('aria-hidden', 'true');
  $('#panel-overlay').classList.remove('panel-overlay--visible');
  $('#panel-overlay').setAttribute('aria-hidden', 'true');
  state.detailOpen = false;
}

function exportCSV() {
  if (!state.lastResults.length) return;

  const header = 'rank,code_naf,code_rome,name,score,score_raw';
  const rows = state.lastResults.map((r) =>
    `${r.rank},${r.code_naf},${r.code_rome},"${r.name.replace(/"/g, '""')}",${r.score},${r.score_raw}`
  );
  const blob = new Blob(['\uFEFF' + header + '\n' + rows.join('\n')], {
    type: 'text/csv;charset=utf-8;',
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `nafnrome_${state.lastQuery.replace(/\s+/g, '_').replace(/[^a-zA-Z0-9_]/g, '')}.csv`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

function toggleBoost() {
  state.boost = !state.boost;
  if (state.lastQuery) {
    search(state.lastQuery);
  }
}

function getFocusables(container) {
  return Array.from(container.querySelectorAll(
    'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])'
  )).filter((el) => !el.closest('[aria-hidden="true"]') && el.offsetParent !== null);
}

function trapFocus(e) {
  if (!state.detailOpen || e.key !== 'Tab') return;
  const panel = $('#detail-panel');
  const focusable = getFocusables(panel);
  if (!focusable.length) return;

  const first = focusable[0];
  const last = focusable[focusable.length - 1];

  if (e.shiftKey && document.activeElement === first) {
    e.preventDefault();
    last.focus();
  } else if (!e.shiftKey && document.activeElement === last) {
    e.preventDefault();
    first.focus();
  }
}

function initSearch() {
  const input = $('#search-input');
  $('#btn-search').addEventListener('click', () => search(input.value));
  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      search(input.value);
    }
  });
}

function initToggles() {
  $('#toggle-boost').addEventListener('change', toggleBoost);

  $$('.persona-chip').forEach((btn) => {
    btn.addEventListener('click', () => {
      state.persona = btn.dataset.persona;
      $$('.persona-chip').forEach((chip) => {
        const active = chip.dataset.persona === state.persona;
        chip.classList.toggle('persona-chip--active', active);
        chip.setAttribute('aria-pressed', String(active));
      });
      renderPersonaInsights(state.lastMeta, state.lastResults);
    });
  });
}

function initChips() {
  $$('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const query = chip.dataset.query;
      $('#search-input').value = query;
      search(query);
    });
  });
}

function initPanel() {
  $('#btn-panel-close').addEventListener('click', closeDetail);
  $('#panel-overlay').addEventListener('click', closeDetail);
  $('#btn-export-csv').addEventListener('click', exportCSV);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && state.detailOpen) {
      closeDetail();
    }
    trapFocus(e);
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initSearch();
  initToggles();
  initChips();
  initPanel();
  fetchHealth();
  updatePageMeta('');
  renderPersonaInsights(null, []);
  $('#search-input').focus();
});
