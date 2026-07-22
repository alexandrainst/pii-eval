"use strict";

// Labels the deterministic regex model can match. Mirrors
// REGEX_FEASIBLE_LABELS in pii_eval/constants.py.
const REGEX_LABELS = [
  "BANK_ACCOUNT",
  "DATE",
  "DRIVERS_LICENSE",
  "EMAIL_ADDRESS",
  "ID_CARD_NUMBER",
  "IP_ADDRESS",
  "PASSPORT",
  "PCI",
  "PHONE_NUMBER",
  "SSN",
  "TAX_NUMBER",
  "URL",
  "VEHICLE_ID",
];

// Hue per label so highlights stay stable and distinguishable. Backgrounds use
// a translucent tint so they read on both light and dark themes.
const LABEL_HUE = {
  PERSON_NAME: 262,
  PHONE_NUMBER: 212,
  EMAIL_ADDRESS: 192,
  SSN: 352,
  BANK_ACCOUNT: 148,
  PCI: 24,
  IP_ADDRESS: 300,
  URL: 176,
  PRECISE_LOCATION: 96,
  DATE: 42,
  PASSPORT: 322,
  DRIVERS_LICENSE: 228,
  ID_CARD_NUMBER: 338,
  TAX_NUMBER: 128,
  SALARY: 72,
  USERNAME: 284,
  VEHICLE_ID: 14,
};

const DEFAULT_TEXT = `Hej team,

Jeg hedder Mette Nielsen, og I kan fange mig på 22 33 44 55 eller mette.nielsen@firma.dk.
Mit CPR-nummer er 250789-1234, og betalingen kan sendes til kontonummer 1234-5678901234.
Min hjemme-IP er 192.168.0.42, og I kan læse mere på https://mette-nielsen.dk.

Venlig hilsen,
Mette`;

// ---------- element handles ----------
const els = {
  input: document.getElementById("input"),
  highlights: document.getElementById("highlights"),
  backdrop: document.getElementById("backdrop"),
  output: document.getElementById("output"),
  legend: document.getElementById("legend"),
  status: document.getElementById("status"),
  statusText: document.getElementById("status-text"),
  labelGrid: document.getElementById("label-grid"),
  genEnabled: document.getElementById("gen-enabled"),
  genFields: document.getElementById("gen-fields"),
  genModel: document.getElementById("gen-model"),
  genBaseUrl: document.getElementById("gen-base-url"),
  genApiKey: document.getElementById("gen-api-key"),
  genMinConf: document.getElementById("gen-min-conf"),
  genExtraBody: document.getElementById("gen-extra-body"),
  genExtraBodyError: document.getElementById("gen-extra-body-error"),
  labelDescList: document.getElementById("label-desc-list"),
  addLabelDesc: document.getElementById("add-label-desc"),
  configSummary: document.getElementById("config-summary"),
  copyBtn: document.getElementById("copy-btn"),
};

// ---------- helpers ----------
function colorVars(label) {
  const hue = LABEL_HUE[label];
  const h = hue === undefined ? 220 : hue;
  const sat = hue === undefined ? 8 : 68;
  return {
    bg: `hsla(${h}, ${sat}%, 52%, 0.16)`,
    line: `hsl(${h}, ${sat}%, 52%)`,
  };
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// Keep spans sorted and non-overlapping (first match wins).
function normalizeSpans(spans) {
  const sorted = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  const out = [];
  let lastEnd = 0;
  for (const s of sorted) {
    if (s.start >= lastEnd && s.end > s.start) {
      out.push(s);
      lastEnd = s.end;
    }
  }
  return out;
}

function setStatus(state, text) {
  els.status.dataset.state = state;
  els.statusText.textContent = text;
}

// ---------- rendering ----------
function renderHighlights(text, spans) {
  let html = "";
  let cursor = 0;
  for (const span of spans) {
    html += escapeHtml(text.slice(cursor, span.start));
    const { bg, line } = colorVars(span.label);
    const style = `--pii-bg:${bg};--pii-line:${line}`;
    html +=
      `<mark class="pii" style="${style}">` +
      `<span class="tag">${escapeHtml(span.label)}</span>` +
      escapeHtml(text.slice(span.start, span.end)) +
      `</mark>`;
    cursor = span.end;
  }
  // Trailing text; keep a trailing space so the final line renders at full
  // height (mirrors how the textarea lays out a trailing newline).
  html += escapeHtml(text.slice(cursor)) + " ";
  els.highlights.innerHTML = html;
}

function renderOutput(text, spans) {
  let html = "";
  let cursor = 0;
  for (const span of spans) {
    html += escapeHtml(text.slice(cursor, span.start));
    html += `<span class="token">&lt;${escapeHtml(span.label)}&gt;</span>`;
    cursor = span.end;
  }
  html += escapeHtml(text.slice(cursor));
  els.output.innerHTML = html;
}

function renderLegend(spans) {
  const counts = new Map();
  for (const s of spans) counts.set(s.label, (counts.get(s.label) || 0) + 1);
  if (counts.size === 0) {
    els.legend.hidden = true;
    els.legend.innerHTML = "";
    return;
  }
  const items = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  els.legend.innerHTML = items
    .map(([label, count]) => {
      const { line } = colorVars(label);
      return (
        `<span class="legend-item">` +
        `<span class="legend-swatch" style="background:${line}"></span>` +
        `${escapeHtml(label)} <span class="legend-count">${count}</span>` +
        `</span>`
      );
    })
    .join("");
  els.legend.hidden = false;
}

// ---------- config ----------
// The API key is never sent to the browser; we send a blank one back and the
// server fills it in from its own config.
let serverKeyConfigured = false;

// ---------- label descriptions (label ↔ description pairs) ----------
function addLabelDescRow(label = "", description = "") {
  const row = document.createElement("div");
  row.className = "label-desc-row";
  const labelInput = document.createElement("input");
  labelInput.className = "ld-label";
  labelInput.placeholder = "LABEL_NAME";
  labelInput.value = label;
  const descInput = document.createElement("input");
  descInput.className = "ld-desc";
  descInput.placeholder = "description";
  descInput.value = description;
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "icon-btn ld-remove";
  remove.textContent = "×";
  remove.setAttribute("aria-label", `Remove ${label || "label"}`);
  row.append(labelInput, descInput, remove);
  els.labelDescList.appendChild(row);
  return row;
}

function renderLabelDescRows(descriptions) {
  els.labelDescList.innerHTML = "";
  for (const [label, description] of Object.entries(descriptions || {})) {
    addLabelDescRow(label, description);
  }
}

function collectLabelDescriptions() {
  const out = {};
  els.labelDescList.querySelectorAll(".label-desc-row").forEach((row) => {
    const label = row.querySelector(".ld-label").value.trim();
    const desc = row.querySelector(".ld-desc").value.trim();
    if (label) out[label] = desc;
  });
  return out;
}

function setExtraBodyError(message) {
  els.genExtraBody.setAttribute("aria-invalid", message ? "true" : "false");
  els.genExtraBodyError.textContent = message || "";
  els.genExtraBodyError.hidden = !message;
}

// Parse the "extra body" textarea into a JSON object, updating the inline error
// message. Returns null when empty or invalid (the request then omits it).
function readExtraBody() {
  const text = els.genExtraBody.value.trim();
  if (!text) {
    setExtraBodyError(null);
    return null;
  }
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch {
    setExtraBodyError("Invalid JSON");
    return null;
  }
  if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
    setExtraBodyError("Must be a JSON object");
    return null;
  }
  setExtraBodyError(null);
  return parsed;
}

function buildConfig() {
  const labels = [...els.labelGrid.querySelectorAll("input:checked")].map(
    (cb) => cb.value
  );
  const config = { classical: { labels }, generative: null };
  if (els.genEnabled.checked) {
    config.generative = {
      model: els.genModel.value.trim(),
      base_url: els.genBaseUrl.value.trim(),
      api_key: els.genApiKey.value,
      min_confidence: parseFloat(els.genMinConf.value) || 0,
      label_descriptions: collectLabelDescriptions(),
    };
    const extraBody = readExtraBody();
    if (extraBody) config.generative.extra_body = extraBody;
  }
  return config;
}

function setGenerativeEnabled(enabled) {
  els.genEnabled.checked = enabled;
  els.genFields.hidden = !enabled;
}

// Pre-populate the UI from the server's active config (pii_eval.yaml).
async function loadServerConfig() {
  try {
    const res = await fetch("../config");
    if (!res.ok) return;
    const data = await res.json();
    const cfg = data.config || {};

    const labels = cfg.classical && cfg.classical.labels;
    if (Array.isArray(labels)) {
      const wanted = new Set(labels);
      els.labelGrid.querySelectorAll("input[type=checkbox]").forEach((cb) => {
        cb.checked = wanted.has(cb.value);
      });
    }

    const gen = cfg.generative;
    if (gen) {
      setGenerativeEnabled(true);
      els.genModel.value = gen.model || "";
      els.genBaseUrl.value = gen.base_url || "";
      if (typeof gen.min_confidence === "number") {
        els.genMinConf.value = gen.min_confidence;
      }
      if (gen.extra_body) {
        els.genExtraBody.value = JSON.stringify(gen.extra_body, null, 2);
      }
      renderLabelDescRows(gen.label_descriptions);
    }

    serverKeyConfigured = Boolean(data.api_key_configured);
    if (serverKeyConfigured) {
      els.genApiKey.placeholder = "•••••• (configured on server)";
    }
  } catch (err) {
    console.warn("Could not load server config:", err);
  }
}

function updateConfigSummary() {
  const total = REGEX_LABELS.length;
  const checked = els.labelGrid.querySelectorAll("input:checked").length;
  const labelPart =
    checked === total ? "all labels" : `${checked}/${total} labels`;
  const modelPart = els.genEnabled.checked ? "Regex + generative" : "Regex model";
  els.configSummary.textContent = `${modelPart} · ${labelPart}`;
}

// ---------- analysis ----------
let reqCounter = 0;

async function analyse() {
  const text = els.input.value;
  const config = buildConfig();
  const myReq = ++reqCounter;
  setStatus("loading", "Analysing…");

  try {
    const res = await fetch("../identify", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, config }),
    });
    if (myReq !== reqCounter) return; // a newer request superseded this one
    if (!res.ok) {
      const detail = await res.text();
      throw new Error(`HTTP ${res.status}: ${detail.slice(0, 200)}`);
    }
    const spans = normalizeSpans(await res.json());
    renderHighlights(text, spans);
    renderOutput(text, spans);
    renderLegend(spans);
    const n = spans.length;
    setStatus("ok", n === 0 ? "No PII found" : `${n} entit${n === 1 ? "y" : "ies"} found`);
  } catch (err) {
    if (myReq !== reqCounter) return;
    console.error(err);
    // Still show the raw text so the caret stays aligned.
    renderHighlights(text, []);
    setStatus("error", "Request failed");
  }
}

let debounceTimer = null;
function scheduleAnalyse() {
  clearTimeout(debounceTimer);
  const delay = els.genEnabled.checked ? 700 : 300;
  debounceTimer = setTimeout(analyse, delay);
}

// Grow the textarea to fit its content so the whole text (and its highlight
// backdrop, which is pinned to the editor via inset:0) stays visible without an
// inner scrollbar.
function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${els.input.scrollHeight}px`;
}

// Immediately mirror the raw text into the backdrop while typing, before the
// debounced /identify request lands, so the caret never drifts from the text.
function syncOverlay() {
  renderHighlightsRaw(els.input.value);
  autoGrow();
}
function renderHighlightsRaw(text) {
  els.highlights.innerHTML = escapeHtml(text) + " ";
}

// ---------- setup ----------
function buildLabelGrid() {
  els.labelGrid.innerHTML = REGEX_LABELS.map((label) => {
    const { line } = colorVars(label);
    return (
      `<label>` +
      `<input type="checkbox" value="${label}" checked />` +
      `<span class="legend-swatch" style="background:${line}"></span>` +
      `${label}</label>`
    );
  }).join("");
}

function wireEvents() {
  els.input.addEventListener("input", () => {
    syncOverlay();
    scheduleAnalyse();
  });
  window.addEventListener("resize", autoGrow);

  els.labelGrid.addEventListener("change", () => {
    updateConfigSummary();
    scheduleAnalyse();
  });

  els.genEnabled.addEventListener("change", () => {
    setGenerativeEnabled(els.genEnabled.checked);
    updateConfigSummary();
    scheduleAnalyse();
  });

  for (const el of [
    els.genModel,
    els.genBaseUrl,
    els.genApiKey,
    els.genMinConf,
    els.genExtraBody,
  ]) {
    el.addEventListener("input", scheduleAnalyse);
  }
  // Validate the JSON on the fly, not only when the debounced request fires.
  els.genExtraBody.addEventListener("input", readExtraBody);

  els.addLabelDesc.addEventListener("click", () => {
    const row = addLabelDescRow();
    row.querySelector(".ld-label").focus();
  });

  // Editing or removing a label/description pair re-runs detection.
  els.labelDescList.addEventListener("input", scheduleAnalyse);
  els.labelDescList.addEventListener("click", (event) => {
    const remove = event.target.closest(".ld-remove");
    if (!remove) return;
    remove.closest(".label-desc-row").remove();
    scheduleAnalyse();
  });

  els.copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(els.output.textContent);
      els.copyBtn.textContent = "Copied!";
      setTimeout(() => (els.copyBtn.textContent = "Copy"), 1200);
    } catch {
      els.copyBtn.textContent = "Copy failed";
      setTimeout(() => (els.copyBtn.textContent = "Copy"), 1200);
    }
  });
}

async function init() {
  buildLabelGrid();
  wireEvents();
  els.input.value = DEFAULT_TEXT;
  syncOverlay();
  await loadServerConfig();
  updateConfigSummary();
  analyse();
}

init();
