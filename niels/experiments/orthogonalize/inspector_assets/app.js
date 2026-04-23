let manifest = null;
let currentEval = null;
let currentSnapshot = null;
let currentRow = null;
let selectedStepId = null;
const snapshotCache = new Map();

const evalSelect = document.getElementById("eval-select");
const snapshotSelect = document.getElementById("snapshot-select");
const hideExcluded = document.getElementById("hide-excluded");
const searchInput = document.getElementById("search");
const summary = document.getElementById("summary");
const tableHead = document.querySelector("#rows-table thead");
const tableBody = document.querySelector("#rows-table tbody");
const timeline = document.getElementById("timeline");
const detailBody = document.getElementById("detail-body");
const rowMeta = document.getElementById("row-meta");

function shortText(value, limit = 140) {
  if (!value) return "";
  return value.length > limit ? `${value.slice(0, limit)}…` : value;
}

function formatScore(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "null";
  return Number(value).toFixed(1);
}

function renderSummary() {
  const rows = currentSnapshot?.rows || [];
  const included = rows.filter((row) => row.included).length;
  summary.innerHTML = `
    <strong>${currentEval.eval_name}</strong>
    <div>${currentSnapshot.label}</div>
    <div>${included} included / ${rows.length} rows shown</div>
  `;
}

function filteredRows() {
  const query = searchInput.value.trim().toLowerCase();
  return (currentSnapshot?.rows || []).filter((row) => {
    if (hideExcluded.checked && !row.included) return false;
    if (!query) return true;
    const haystack = [
      row.row_id,
      row.question,
      row.primary_reference_answer,
      JSON.stringify(row.reference_answers),
    ].join(" ").toLowerCase();
    return haystack.includes(query);
  });
}

function renderTable() {
  const targets = currentEval.target_evals;
  const rows = filteredRows();
  tableHead.innerHTML = "";
  tableBody.innerHTML = "";

  const headRow = document.createElement("tr");
  ["row", "status", "question", "primary answer", ...targets].forEach((label) => {
    const th = document.createElement("th");
    th.textContent = label;
    headRow.appendChild(th);
  });
  tableHead.appendChild(headRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    if (currentRow && currentRow.lineage_id === row.lineage_id && currentRow.step_id === row.step_id) {
      tr.classList.add("selected");
    }
    tr.addEventListener("click", () => {
      currentRow = row;
      selectedStepId = row.step_id;
      renderTable();
      renderDetail();
    });

    const cells = [
      row.row_id,
      row.status,
      shortText(row.question, 120),
      shortText(row.primary_reference_answer, 120),
      ...targets.map((target) => formatScore(row.propensity_scores[target])),
    ];
    cells.forEach((value) => {
      const td = document.createElement("td");
      td.textContent = value;
      tr.appendChild(td);
    });
    tableBody.appendChild(tr);
  });
}

function renderTimeline(lineage) {
  timeline.innerHTML = "";
  lineage.history.forEach((step) => {
    const button = document.createElement("button");
    button.textContent = step.label;
    if (step.step_id === selectedStepId) {
      button.classList.add("active");
    }
    button.addEventListener("click", () => {
      selectedStepId = step.step_id;
      renderTimeline(lineage);
      renderDetailBody(step);
    });
    timeline.appendChild(button);
  });
}

function renderDetailBody(step) {
  const scoreCards = currentEval.target_evals.map((target) => `
    <div class="score-card">
      <strong>${target}</strong>
      <span>${formatScore(step.propensity_scores[target])}</span>
    </div>
  `).join("");

  const refs = Object.entries(step.reference_answers || {})
    .map(([key, value]) => `<div class="block"><strong>${key}</strong><br>${escapeHtml(value)}</div>`)
    .join("");

  const crossSummary = (step.cross_summary || []).map((item) => {
    const statusClass = item.is_violation ? "chip warn" : "chip";
    return `<span class="${statusClass}">${item.target_eval}: ${formatScore(item.primary_score)} / gap ${item.max_gap === null ? "null" : item.max_gap.toFixed(1)}</span>`;
  }).join("");

  const messages = (step.chat_history || []).map((message) => `
    <div class="message">
      <div class="message-role">${escapeHtml(message.role)}</div>
      <div>${escapeHtml(message.text || "")}</div>
    </div>
  `).join("");

  detailBody.innerHTML = `
    <section class="detail-section">
      <h3>Question</h3>
      <div class="block">${escapeHtml(step.question || "")}</div>
    </section>
    <section class="detail-section">
      <h3>Reference Answers</h3>
      <div class="score-grid">${refs}</div>
    </section>
    <section class="detail-section">
      <h3>Primary Cross-Propensity Scores</h3>
      <div class="score-grid">${scoreCards}</div>
    </section>
    <section class="detail-section">
      <h3>Cross Summary</h3>
      <div class="chips">${crossSummary || '<span class="muted">No cross-summary recorded.</span>'}</div>
    </section>
    <section class="detail-section">
      <h3>Violations</h3>
      <div class="block">${escapeHtml((step.violations || []).join("\n") || "None")}</div>
    </section>
    <section class="detail-section">
      <h3>Chat History</h3>
      ${messages || '<p class="muted">No chat history for this step.</p>'}
    </section>
    <section class="detail-section">
      <h3>Raw Score Rows</h3>
      <pre>${escapeHtml(JSON.stringify(step.score_rows || [], null, 2))}</pre>
    </section>
  `;
}

function renderDetail() {
  if (!currentRow) {
    rowMeta.textContent = "";
    timeline.innerHTML = "";
    detailBody.innerHTML = `<p class="muted">Select a row to inspect its revisions.</p>`;
    return;
  }
  const lineage = currentEval.lineages[currentRow.lineage_id];
  if (!lineage) {
    return;
  }
  const step = lineage.history.find((item) => item.step_id === selectedStepId) || lineage.history[lineage.history.length - 1];
  rowMeta.textContent = `${currentRow.row_id} · ${currentRow.status} · ${step.label}`;
  renderTimeline(lineage);
  renderDetailBody(step);
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

async function loadEval(path) {
  const response = await fetch(path);
  return response.json();
}

async function loadSnapshot(path) {
  if (!snapshotCache.has(path)) {
    snapshotCache.set(path, fetch(path).then((response) => response.json()));
  }
  return snapshotCache.get(path);
}

async function onEvalChange() {
  const entry = manifest.evals.find((item) => item.name === evalSelect.value);
  currentEval = await loadEval(entry.path);
  snapshotSelect.innerHTML = "";
  currentEval.snapshots.forEach((snapshot) => {
    const option = document.createElement("option");
    option.value = snapshot.id;
    option.textContent = snapshot.label;
    snapshotSelect.appendChild(option);
  });
  snapshotSelect.value = currentEval.snapshots[currentEval.snapshots.length - 1].id;
  await onSnapshotChange();
}

async function onSnapshotChange() {
  const snapshotMeta = currentEval.snapshots.find((snapshot) => snapshot.id === snapshotSelect.value);
  currentSnapshot = await loadSnapshot(snapshotMeta.json_path);
  currentSnapshot.id = snapshotMeta.id;
  currentRow = null;
  selectedStepId = null;
  renderSummary();
  renderTable();
  renderDetail();
}

async function bootstrap() {
  const response = await fetch("./manifest.json");
  manifest = await response.json();
  manifest.evals.forEach((entry) => {
    const option = document.createElement("option");
    option.value = entry.name;
    option.textContent = entry.name;
    evalSelect.appendChild(option);
  });
  if (manifest.evals.length > 0) {
    evalSelect.value = manifest.evals[0].name;
    await onEvalChange();
  }
}

evalSelect.addEventListener("change", onEvalChange);
snapshotSelect.addEventListener("change", onSnapshotChange);
hideExcluded.addEventListener("change", renderTable);
searchInput.addEventListener("input", renderTable);

bootstrap();
