// BIO comparison visualizer frontend.

const datasetSel = document.getElementById("dataset-select");
const modelSel = document.getElementById("model-select");
const entrySel = document.getElementById("entry-select");
const prevBtn = document.getElementById("prev-btn");
const nextBtn = document.getElementById("next-btn");
const gtBox = document.getElementById("gt-text");
const predBox = document.getElementById("pred-text");
const predTitle = document.getElementById("pred-title");
const metricsBox = document.getElementById("metrics");
const legendBox = document.getElementById("legend");
const statusBox = document.getElementById("status");

const state = {
  dataset: null,
  model: null,
  entries: [],
  entryId: null,
};

// Deterministic color from string (HSL).
function colorForLabel(label) {
  let h = 0;
  for (let i = 0; i < label.length; i++) {
    h = (h * 31 + label.charCodeAt(i)) >>> 0;
  }
  const hue = h % 360;
  return {
    bg: `hsl(${hue}, 70%, 85%)`,
    border: `hsl(${hue}, 60%, 55%)`,
    swatch: `hsl(${hue}, 65%, 65%)`,
  };
}

function escapeHtml(s) {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function renderTextWithSpans(text, spans) {
  // spans must be sorted, non-overlapping (they are, since produced from BIO).
  const sorted = [...spans].sort((a, b) => a.char_start - b.char_start);
  let html = "";
  let cursor = 0;
  for (const s of sorted) {
    if (s.char_start < cursor) continue; // skip overlaps defensively
    if (s.char_start > cursor) {
      html += escapeHtml(text.slice(cursor, s.char_start));
    }
    const c = colorForLabel(s.iso_label || s.raw_label || "?");
    const tip =
      `${s.iso_label}` +
      (s.raw_label && s.raw_label !== s.iso_label ? ` (raw: ${s.raw_label})` : "") +
      ` — tokens [${s.token_start}, ${s.token_end}) — chars [${s.char_start}, ${s.char_end})`;
    html +=
      `<span class="span" title="${escapeHtml(tip)}" ` +
      `style="background:${c.bg};border-bottom-color:${c.border}">` +
      escapeHtml(text.slice(s.char_start, s.char_end)) +
      `<span class="lbl">${escapeHtml(s.raw_label)}</span>` +
      `</span>`;
    cursor = s.char_end;
  }
  if (cursor < text.length) html += escapeHtml(text.slice(cursor));
  return html;
}

function renderLegend(labels) {
  legendBox.innerHTML = "";
  for (const l of labels) {
    const c = colorForLabel(l);
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.innerHTML =
      `<span class="swatch" style="background:${c.swatch}"></span>` +
      escapeHtml(l);
    legendBox.appendChild(chip);
  }
}

function fmt(n) {
  if (n === null || n === undefined) return "-";
  return (Math.round(n * 1000) / 10).toFixed(1) + "%";
}

function renderMetrics(metrics) {
  if (!metrics) {
    metricsBox.innerHTML = "";
    return;
  }
  const s = metrics.span;
  const t = metrics.token;
  metricsBox.innerHTML = `
    <span class="group">
      <b>Span exact</b>
      P=${fmt(s.precision)} R=${fmt(s.recall)} F1=${fmt(s.f1)}
      <small>(TP=${s.tp} FP=${s.fp} FN=${s.fn})</small>
    </span>
    <span class="group">
      <b>Token (ISO)</b>
      P=${fmt(t.precision)} R=${fmt(t.recall)} F1=${fmt(t.f1)}
      <small>(TP=${t.tp} FP=${t.fp} FN=${t.fn})</small>
    </span>`;
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status} ${r.statusText}`);
  return r.json();
}

async function loadDatasetsAndModels() {
  const [datasets, models] = await Promise.all([
    fetchJSON("/api/datasets"),
    fetchJSON("/api/models"),
  ]);
  datasetSel.innerHTML = datasets
    .map((d) => `<option value="${d}">${d}</option>`)
    .join("");
  modelSel.innerHTML =
    `<option value="">(no model)</option>` +
    models.map((m) => `<option value="${m}">${m}</option>`).join("");
  state.dataset = datasets[0] || null;
  state.model = models[0] || "";
  if (state.dataset) datasetSel.value = state.dataset;
  if (state.model) modelSel.value = state.model;
}

async function loadEntries() {
  if (!state.dataset) return;
  const url = new URL("/api/entries", window.location.origin);
  url.searchParams.set("dataset", state.dataset);
  if (state.model) url.searchParams.set("model", state.model);
  state.entries = await fetchJSON(url);
  entrySel.innerHTML = state.entries
    .map(
      (e) =>
        `<option value="${e.id}">${e.id} · gt=${e.gt_span_count}${
          state.model ? (e.has_prediction ? " ✓" : " ✗") : ""
        }</option>`
    )
    .join("");
  if (state.entries.length) {
    state.entryId = state.entries[0].id;
    entrySel.value = state.entryId;
  } else {
    state.entryId = null;
  }
}

async function loadEntry() {
  if (!state.dataset || !state.entryId) return;
  statusBox.textContent = "Loading...";
  try {
    const url = new URL("/api/entry", window.location.origin);
    url.searchParams.set("dataset", state.dataset);
    url.searchParams.set("id", state.entryId);
    if (state.model) url.searchParams.set("model", state.model);
    const data = await fetchJSON(url);

    gtBox.innerHTML = renderTextWithSpans(data.text, data.ground_truth.spans);
    if (data.prediction) {
      predTitle.textContent = `Prediction · ${state.model}`;
      predBox.innerHTML = renderTextWithSpans(
        data.text,
        data.prediction.spans
      );
    } else {
      predTitle.textContent = "Prediction (none for this entry)";
      predBox.textContent = data.text;
    }
    renderMetrics(data.metrics);

    const labels = new Set();
    data.ground_truth.spans.forEach((s) => labels.add(s.iso_label));
    if (data.prediction) {
      data.prediction.spans.forEach((s) => labels.add(s.iso_label));
    }
    renderLegend([...labels].sort());

    statusBox.textContent =
      `Entry: ${data.id} · tokens: ${data.tokens.length} · chars: ${data.text.length}`;
  } catch (err) {
    statusBox.textContent = `Error: ${err.message}`;
    gtBox.textContent = "";
    predBox.textContent = "";
  }
}

function navigate(delta) {
  if (!state.entries.length) return;
  const idx = state.entries.findIndex((e) => e.id === state.entryId);
  const next = (idx + delta + state.entries.length) % state.entries.length;
  state.entryId = state.entries[next].id;
  entrySel.value = state.entryId;
  loadEntry();
}

datasetSel.addEventListener("change", async () => {
  state.dataset = datasetSel.value;
  await loadEntries();
  await loadEntry();
});
modelSel.addEventListener("change", async () => {
  state.model = modelSel.value;
  await loadEntries();
  await loadEntry();
});
entrySel.addEventListener("change", () => {
  state.entryId = entrySel.value;
  loadEntry();
});
prevBtn.addEventListener("click", () => navigate(-1));
nextBtn.addEventListener("click", () => navigate(1));
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  if (e.key === "ArrowLeft") navigate(-1);
  else if (e.key === "ArrowRight") navigate(1);
});

(async function init() {
  try {
    await loadDatasetsAndModels();
    await loadEntries();
    await loadEntry();
  } catch (err) {
    statusBox.textContent = `Init error: ${err.message}`;
  }
})();
