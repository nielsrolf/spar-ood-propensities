#!/usr/bin/env python3
"""
Config-driven annotation GUI for propensity audit human evaluation.

Usage:
    python propensity_audit/annotate.py --config configs/risk_affinity.yaml
    python propensity_audit/annotate.py --config configs/alignment.yaml

Opens a browser at http://localhost:8780 with a single-page annotation tool.
5 ordinal buckets (from config) + INCOHERENT, with keyboard shortcuts.
No dependencies beyond the Python standard library + pandas.
"""

import argparse
import csv
import json
import io
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from audit_config import from_yaml, AuditConfig

PORT = 8780


def _strip_preamble(prompt: str) -> str:
    """Strip the shared orthogonality-preamble from a judge prompt so the UI
    only shows the trait-specific rubric. Falls back to the full text if no
    known preamble marker is found."""
    if not prompt:
        return prompt
    # The preamble ends with a "METRIC PROMPT" header; everything before it
    # (including the leading "# orthogonality-preamble-v1" tag and the
    # null-vs-score discussion) is shared boilerplate.
    markers = [
        "METRIC PROMPT (use the scale defined here, but apply the null rule above):",
        "METRIC PROMPT",
    ]
    for m in markers:
        idx = prompt.find(m)
        if idx >= 0:
            tail = prompt[idx + len(m):]
            # Trim leading separator lines / whitespace
            return tail.lstrip("-\n :").strip()
    return prompt.strip()


# ── Data loading ────────────────────────────────────────────────────

def find_blind_csv(config: AuditConfig) -> Path:
    """Find the blind sample CSV in the output directory."""
    candidates = sorted(config.output_dir.glob("sample_*_blind.csv"))
    if candidates:
        return candidates[-1]  # most recent
    raise FileNotFoundError(
        f"No sample_*_blind.csv found in {config.output_dir}. "
        "Run sample_for_review.py first."
    )


_RATER: str | None = None  # set in main(); per-rater scoping for save files


def _resolve_rater(explicit: str | None = None) -> str:
    """Resolve a per-rater identifier used to scope annotation save files.

    Priority: explicit arg > $AUDIT_RATER > git config user.name > $USER.
    Sanitised to a filesystem-safe slug (alnum / dash / underscore).
    """
    import os as _os
    import re as _re
    import subprocess as _sp

    candidate = explicit or _os.environ.get("AUDIT_RATER")
    if not candidate:
        try:
            out = _sp.run(
                ["git", "config", "user.name"],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0:
                candidate = out.stdout.strip()
        except Exception:
            pass
    if not candidate:
        candidate = _os.environ.get("USER") or _os.environ.get("LOGNAME") or "anon"
    slug = _re.sub(r"[^A-Za-z0-9_.-]+", "-", candidate).strip("-_.")
    return slug or "anon"


def save_path(config: AuditConfig) -> Path:
    """Per-rater annotation file. Falls back to legacy unscoped name when
    _RATER is unset (e.g. analysis scripts importing this module)."""
    if _RATER:
        return config.output_dir / f"human_annotations__{_RATER}.csv"
    return config.output_dir / "human_annotations.csv"


def load_data(blind_path: Path, annotations_path: Path):
    rows = []
    with open(blind_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    annotations = {}
    if annotations_path.exists():
        with open(annotations_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                idx = int(r["index"])
                annotations[idx] = r.get("human_label", "")

    return rows, annotations


def save_annotations(rows, annotations, path: Path):
    row_keys = [k for k in rows[0].keys() if k != "human_label"]
    fieldnames = ["index"] + row_keys + ["human_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = {"index": i, **{k: row[k] for k in row_keys}, "human_label": annotations.get(i, "")}
            writer.writerow(out)


# ── HTML generation ─────────────────────────────────────────────────

def _css_safe(label: str) -> str:
    """Convert label to CSS-safe class name."""
    return label.lower().replace(" ", "-").replace("_", "-")


def _build_button_styles(config: AuditConfig) -> str:
    """Generate CSS classes for each bucket button and dot."""
    lines = []
    for opt in config.all_options:
        cls = _css_safe(opt["label"])
        color = opt["color"]
        lines.append(f"""
  .btn-{cls} {{ background: {color}; color: #fff; }}
  .btn-{cls}:hover {{ background: {color}; filter: brightness(1.2); }}
  .btn-{cls}.selected {{ border-color: #fff; box-shadow: 0 0 12px {color}88; }}
  .dot-{cls} {{ background: {color}; }}""")
    return "\n".join(lines)


def _display_label(label: str, config: AuditConfig) -> str:
    """Friendlier UI text — but the persisted value stays as `label`."""
    if label == config.INCOHERENT_LABEL:
        return "Invalid/Incoherent"
    if label == config.NULL_LABEL:
        return "Off-topic/No-evidence"
    return label


def _build_button_html(config: AuditConfig) -> str:
    """Generate the label button elements."""
    btns = []
    for i, opt in enumerate(config.all_options):
        cls = _css_safe(opt["label"])
        key_display = opt["key"].upper()
        label = opt["label"]
        shown = _display_label(label, config)

        # Add separator gap before INCOHERENT
        if opt["label"] == config.INCOHERENT_LABEL:
            btns.append('<span style="width:24px;display:inline-block"></span>')

        btns.append(
            f'<button class="label-btn btn-{cls}" '
            f"onclick=\"label('{label}')\">{shown} ({key_display})</button>"
        )
    return "\n    ".join(btns)


def _build_color_map_json(config: AuditConfig) -> str:
    """Generate JS COLOR_MAP for minimap."""
    entries = {}
    for opt in config.all_options:
        entries[opt["label"]] = opt["color"]
    return json.dumps(entries)


def _build_key_cases(config: AuditConfig) -> str:
    """Generate JS switch cases for keyboard shortcuts."""
    lines = []
    for opt in config.all_options:
        k = opt["key"]
        label = opt["label"]
        lines.append(f"    case '{k}': case '{k.upper()}': label('{label}'); break;")
    return "\n".join(lines)


def _build_filter_buttons(config: AuditConfig) -> str:
    """Generate filter row buttons."""
    btns = [
        '<button class="filter-btn active" onclick="setFilter(\'all\')">All</button>',
        '<button class="filter-btn" onclick="setFilter(\'unlabeled\')">Unlabeled</button>',
    ]
    for opt in config.all_options:
        label = opt["label"]
        short = opt.get("short", label[:3])
        btns.append(
            f'<button class="filter-btn" onclick="setFilter(\'{label}\')">{short}</button>'
        )
    return "\n    ".join(btns)


def _build_keyboard_hint(config: AuditConfig) -> str:
    """Generate keyboard shortcut hint text."""
    parts = []
    for opt in config.all_options:
        parts.append(f"<b>{opt['key'].upper()}</b> = {_display_label(opt['label'], config)}")
    return " &middot; ".join(parts) + ' &middot; <b>&larr;/&rarr;</b> = Navigate &middot; <b>U</b> = Next unlabeled'


def _build_metadata_fields(config: AuditConfig) -> str:
    """Generate JS array of metadata column names for the reveal toggle.

    `condition` is excluded from the on-screen reveal so annotators are not
    biased by which system prompt produced the response — the column is still
    persisted in the per-eval annotations CSV.
    """
    cols = [c for c in config.metadata_columns if c != "condition"]
    return json.dumps(cols)


def build_html(config: AuditConfig) -> str:
    """Build the complete HTML page from config."""
    button_styles = _build_button_styles(config)
    button_html = _build_button_html(config)
    color_map_json = _build_color_map_json(config)
    key_cases = _build_key_cases(config)
    filter_buttons = _build_filter_buttons(config)
    keyboard_hint = _build_keyboard_hint(config)
    title = f"Propensity Audit &mdash; {config.display_name}"
    metadata_fields = _build_metadata_fields(config)
    randomize_order_js = "true" if config.randomize_order else "false"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{config.display_name} — Propensity Audit</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: #0d1117; color: #c9d1d9; padding: 0; }}

  .top-bar {{ background: #161b22; border-bottom: 1px solid #30363d;
             padding: 12px 24px; display: flex; align-items: center; gap: 16px;
             position: sticky; top: 0; z-index: 10; }}
  .top-bar h1 {{ font-size: 16px; font-weight: 600; color: #58a6ff; }}
  .progress {{ font-size: 13px; color: #8b949e; }}
  .progress-bar {{ width: 200px; height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; }}
  .progress-fill {{ height: 100%; background: #238636; border-radius: 3px; transition: width 0.3s; }}

  .container {{ max-width: 900px; margin: 0 auto; padding: 24px; }}

  .nav {{ display: flex; gap: 8px; align-items: center; margin-bottom: 16px; }}
  .nav button {{ background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 13px; }}
  .nav button:hover {{ background: #30363d; }}
  .nav input {{ width: 60px; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d;
               padding: 6px 8px; border-radius: 6px; text-align: center; font-size: 13px; }}
  .nav .total {{ color: #8b949e; font-size: 13px; }}

  .filter-row {{ display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }}
  .filter-btn {{ background: #21262d; color: #8b949e; border: 1px solid #30363d;
                padding: 4px 10px; border-radius: 12px; cursor: pointer; font-size: 12px; }}
  .filter-btn.active {{ background: #1f6feb33; color: #58a6ff; border-color: #1f6feb; }}

  .meta {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
  .meta-hidden .meta-detail {{ display: none; }}
  .tag {{ background: #21262d; border: 1px solid #30363d; padding: 3px 10px;
         border-radius: 12px; font-size: 12px; color: #8b949e; }}
  .reveal-btn {{ background: none; border: 1px solid #30363d; color: #484f58;
                padding: 3px 10px; border-radius: 12px; font-size: 12px;
                cursor: pointer; }}
  .reveal-btn:hover {{ color: #8b949e; border-color: #484f58; }}

  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 20px; margin-bottom: 16px; }}
  .card h3 {{ font-size: 13px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase;
             letter-spacing: 0.5px; }}
  .card .text {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
  .question-text {{ color: #58a6ff; }}
  .response-text {{ color: #c9d1d9; }}

  .label-buttons {{ display: flex; gap: 12px; margin-top: 20px; justify-content: center;
                    flex-wrap: wrap; }}
  .label-btn {{ padding: 12px 24px; border-radius: 8px; border: 2px solid transparent;
               font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.15s; }}
  .label-btn:hover {{ transform: translateY(-1px); }}

{button_styles}

  .keyboard-hint {{ text-align: center; margin-top: 12px; font-size: 12px; color: #484f58; }}

  .status-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                margin-right: 4px; vertical-align: middle; }}
  .dot-empty {{ background: #484f58; }}

  .minimap {{ display: flex; flex-wrap: wrap; gap: 2px; margin-top: 16px; padding: 12px;
             background: #161b22; border: 1px solid #30363d; border-radius: 8px; }}
  .minimap-cell {{ width: 10px; height: 10px; border-radius: 2px; cursor: pointer;
                  transition: transform 0.1s; }}
  .minimap-cell:hover {{ transform: scale(1.8); }}
  .minimap-cell.current {{ outline: 2px solid #58a6ff; outline-offset: 1px; }}
</style>
</head>
<body>

<div class="top-bar">
  <h1>{title}</h1>
  <div class="progress">
    <span id="progress-text">0 / 0</span>
  </div>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
</div>

<div class="container">
  <div class="nav">
    <button onclick="go(-1)" id="btn-prev">&larr; Prev</button>
    <input type="number" id="idx-input" min="1" value="1" onchange="jumpTo(this.value-1)">
    <span class="total">/ <span id="total-count">0</span></span>
    <button onclick="go(1)" id="btn-next">Next &rarr;</button>
    <span style="flex:1"></span>
    <button onclick="goToNextUnlabeled()">Next unlabeled</button>
    <button onclick="exportCSV()">Export CSV</button>
  </div>

  <div class="filter-row">
    <span style="font-size:12px;color:#8b949e;line-height:24px;">Show:</span>
    {filter_buttons}
  </div>

  <div class="meta meta-hidden" id="meta"></div>

  <div class="aux-toggles" style="display:flex;gap:8px;margin-bottom:12px;">
    <button class="reveal-btn" id="toggle-def" onclick="toggleDef()">Show trait definition</button>
    <button class="reveal-btn" id="toggle-refs" onclick="toggleRefs()">Show reference answers</button>
  </div>

  <div class="card aux-card" id="def-card" style="display:none;">
    <h3>Trait definition (judge prompt)</h3>
    <div class="text" id="def-text" style="font-size:12px;color:#8b949e;max-height:300px;overflow:auto;"></div>
  </div>

  <div class="card aux-card" id="refs-card" style="display:none;">
    <h3>Reference answers <span style="font-weight:normal;color:#8b949e;font-size:11px;" id="refs-itemid"></span></h3>
    <div id="refs-body" style="display:flex;flex-direction:column;gap:10px;"></div>
  </div>

  <div class="card">
    <h3>Question</h3>
    <div class="text question-text" id="question"></div>
  </div>

  <div class="card">
    <h3>Response</h3>
    <div class="text response-text" id="response"></div>
  </div>

  <div class="label-buttons">
    {button_html}
  </div>
  <div class="keyboard-hint">
    Keyboard: {keyboard_hint}
  </div>

  <div class="minimap" id="minimap"></div>
</div>

<script>
const COLOR_MAP = {color_map_json};
const METADATA_FIELDS = {metadata_fields};
// Default randomize-order setting from the config; can be overridden per-eval
// by injecting window.RANDOMIZE_ORDER before this script runs.
const RANDOMIZE_ORDER_DEFAULT = {randomize_order_js};

let DATA = [];
let annotations = {{}};
let currentIdx = 0;
let filteredIndices = [];
let filterMode = 'all';
let metaRevealed = false;
let DEFINITION = '';
let REFERENCES = {{}};
let defShown = false;
let refsShown = false;

// Deterministic 32-bit hash → seeded RNG so the shuffle is stable for a given
// (eval, set-of-rows). Same rater hitting reload sees the same order; clearing
// localStorage / running a fresh sample regenerates a new order.
function _hash32(s) {{
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {{
    h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  }}
  return h >>> 0;
}}
function _mulberry32(a) {{
  return function() {{
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }};
}}
function _shuffleStable(arr, seedKey) {{
  const rng = _mulberry32(_hash32(seedKey));
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {{
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }}
  return a;
}}

async function init() {{
  const evalParam = window.AUDIT_EVAL ? ('?eval=' + encodeURIComponent(window.AUDIT_EVAL)) : '';
  const resp = await fetch('/api/data' + evalParam);
  const json = await resp.json();
  DATA = json.rows;
  annotations = json.annotations;
  DEFINITION = json.definition || '';
  REFERENCES = json.references || {{}};
  document.getElementById('total-count').textContent = DATA.length;
  document.getElementById('def-text').textContent = DEFINITION;
  applyFilter();
  buildMinimap();
  render();
}}

function toggleDef() {{
  defShown = !defShown;
  document.getElementById('def-card').style.display = defShown ? '' : 'none';
  document.getElementById('toggle-def').textContent =
    defShown ? 'Hide trait definition' : 'Show trait definition';
}}

function toggleRefs() {{
  refsShown = !refsShown;
  document.getElementById('refs-card').style.display = refsShown ? '' : 'none';
  document.getElementById('toggle-refs').textContent =
    refsShown ? 'Hide reference answers' : 'Show reference answers';
  renderRefs();
}}

function renderRefs() {{
  const row = DATA[currentIdx] || {{}};
  const itemId = row.item_id || '';
  document.getElementById('refs-itemid').textContent = itemId ? '— ' + itemId : '';
  const body = document.getElementById('refs-body');
  body.innerHTML = '';
  const refs = REFERENCES[itemId];
  if (!refs) {{
    body.innerHTML = '<div style="color:#8b949e;font-size:12px;">No reference answers available for this item.</div>';
    return;
  }}
  Object.keys(refs).forEach(role => {{
    const wrap = document.createElement('div');
    wrap.style.borderLeft = '3px solid #30363d';
    wrap.style.paddingLeft = '10px';
    const h = document.createElement('div');
    h.textContent = role;
    h.style.fontSize = '11px';
    h.style.color = '#58a6ff';
    h.style.textTransform = 'uppercase';
    h.style.letterSpacing = '0.5px';
    h.style.marginBottom = '4px';
    const t = document.createElement('div');
    t.className = 'text';
    t.style.fontSize = '13px';
    t.style.whiteSpace = 'pre-wrap';
    t.textContent = refs[role];
    wrap.appendChild(h);
    wrap.appendChild(t);
    body.appendChild(wrap);
  }});
}}

function applyFilter() {{
  filteredIndices = [];
  for (let i = 0; i < DATA.length; i++) {{
    const lbl = annotations[i] || '';
    if (filterMode === 'all') filteredIndices.push(i);
    else if (filterMode === 'unlabeled' && !lbl) filteredIndices.push(i);
    else if (lbl === filterMode) filteredIndices.push(i);
  }}
  const randomize = (typeof window.RANDOMIZE_ORDER !== 'undefined')
    ? window.RANDOMIZE_ORDER : RANDOMIZE_ORDER_DEFAULT;
  if (randomize && filteredIndices.length > 1) {{
    const evalKey = window.AUDIT_EVAL || 'default';
    const seedKey = evalKey + '|n=' + DATA.length + '|cross-elicit-shuffle-v1';
    filteredIndices = _shuffleStable(filteredIndices, seedKey);
  }}
}}

function setFilter(mode) {{
  filterMode = mode;
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  applyFilter();
  if (!filteredIndices.includes(currentIdx)) {{
    currentIdx = filteredIndices.length > 0 ? filteredIndices[0] : 0;
  }}
  render();
}}

function cssClass(label) {{
  return label.toLowerCase().replace(/ /g, '-').replace(/_/g, '-');
}}

function render() {{
  if (DATA.length === 0) return;
  const row = DATA[currentIdx];
  const lbl = annotations[currentIdx] || '';

  document.getElementById('question').textContent = row.question;
  document.getElementById('response').textContent = row.response;
  document.getElementById('idx-input').value = currentIdx + 1;

  // Meta tags
  const meta = document.getElementById('meta');
  meta.innerHTML = '';
  const dotClass = lbl ? 'dot-' + cssClass(lbl) : 'dot-empty';

  const statusTag = document.createElement('span');
  statusTag.className = 'tag';
  statusTag.innerHTML = '<span class="status-dot ' + dotClass + '"></span>' + (lbl || 'unlabeled');
  meta.appendChild(statusTag);

  const revealBtn = document.createElement('button');
  revealBtn.className = 'reveal-btn';
  revealBtn.textContent = metaRevealed ? 'Hide metadata' : 'Show metadata';
  revealBtn.onclick = () => {{
    metaRevealed = !metaRevealed;
    meta.classList.toggle('meta-hidden', !metaRevealed);
    revealBtn.textContent = metaRevealed ? 'Hide metadata' : 'Show metadata';
  }};
  meta.appendChild(revealBtn);

  // Dynamic metadata fields from config
  METADATA_FIELDS.forEach(field => {{
    const val = row[field];
    if (val) {{
      const el = document.createElement('span');
      el.className = 'tag meta-detail';
      el.textContent = val;
      meta.appendChild(el);
    }}
  }});

  // Highlight selected button
  document.querySelectorAll('.label-btn').forEach(b => b.classList.remove('selected'));
  if (lbl) {{
    const cls = '.btn-' + cssClass(lbl);
    const btn = document.querySelector(cls);
    if (btn) btn.classList.add('selected');
  }}

  updateProgress();
  updateMinimap();
  if (refsShown) renderRefs();
}}

function updateProgress() {{
  let done = Object.values(annotations).filter(v => v).length;
  document.getElementById('progress-text').textContent = done + ' / ' + DATA.length;
  document.getElementById('progress-fill').style.width = (done / DATA.length * 100) + '%';
}}

function buildMinimap() {{
  const map = document.getElementById('minimap');
  map.innerHTML = '';
  for (let i = 0; i < DATA.length; i++) {{
    const cell = document.createElement('div');
    cell.className = 'minimap-cell';
    cell.dataset.idx = i;
    cell.onclick = () => {{ currentIdx = i; render(); }};
    map.appendChild(cell);
  }}
}}

function updateMinimap() {{
  document.querySelectorAll('.minimap-cell').forEach(cell => {{
    const i = parseInt(cell.dataset.idx);
    const lbl = annotations[i] || '';
    cell.style.background = COLOR_MAP[lbl] || '#21262d';
    cell.classList.toggle('current', i === currentIdx);
  }});
}}

async function label(value) {{
  annotations[currentIdx] = value;
  await fetch('/api/annotate', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{index: currentIdx, label: value, eval: window.AUDIT_EVAL || null}})
  }});
  // Auto-advance to next in filtered set
  const posInFiltered = filteredIndices.indexOf(currentIdx);
  if (posInFiltered >= 0 && posInFiltered < filteredIndices.length - 1) {{
    currentIdx = filteredIndices[posInFiltered + 1];
  }}
  render();
}}

function go(delta) {{
  const posInFiltered = filteredIndices.indexOf(currentIdx);
  let newPos = posInFiltered + delta;
  if (newPos < 0) newPos = 0;
  if (newPos >= filteredIndices.length) newPos = filteredIndices.length - 1;
  if (filteredIndices.length > 0) currentIdx = filteredIndices[newPos];
  render();
}}

function jumpTo(idx) {{
  idx = parseInt(idx);
  if (idx >= 0 && idx < DATA.length) {{ currentIdx = idx; render(); }}
}}

function goToNextUnlabeled() {{
  for (let i = currentIdx + 1; i < DATA.length; i++) {{
    if (!annotations[i]) {{ currentIdx = i; render(); return; }}
  }}
  for (let i = 0; i < currentIdx; i++) {{
    if (!annotations[i]) {{ currentIdx = i; render(); return; }}
  }}
  alert('All rows are labeled!');
}}

function exportCSV() {{ window.open('/api/export', '_blank'); }}

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;
  switch(e.key) {{
{key_cases}
    case 'ArrowLeft':  go(-1); e.preventDefault(); break;
    case 'ArrowRight': go(1);  e.preventDefault(); break;
    case 'u': case 'U': goToNextUnlabeled(); break;
  }}
}});

init();
</script>
</body>
</html>"""


# ── HTTP Server ─────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    rows = []
    annotations = {}
    config = None
    _save_path = None
    _html = ""
    # Multi-eval mode (configs-dir):
    configs_dir = None
    eval_state: dict = {}
    active_eval: str | None = None

    def log_message(self, format, *args):
        pass  # suppress request logs

    # ─ Multi-eval helpers ─────────────────────────────────────────────
    def _load_eval(self, name: str):
        st = self.eval_state.get(name)
        if st is not None:
            return st
        cfg_path = self.configs_dir / f"{name}.yaml"
        if not cfg_path.exists():
            return None
        cfg = from_yaml(cfg_path)
        try:
            blind = find_blind_csv(cfg)
        except FileNotFoundError:
            return None
        ann_path = save_path(cfg)
        rows, annotations = load_data(blind, ann_path)
        st = {
            "rows": rows,
            "annotations": {str(k): v for k, v in annotations.items()},
            "config": cfg,
            "save_path": ann_path,
            "html": build_html(cfg),
        }
        self.eval_state[name] = st
        return st

    def _activate(self, name: str):
        st = self._load_eval(name)
        if st is None:
            return False
        type(self).rows = st["rows"]
        type(self).annotations = st["annotations"]
        type(self).config = st["config"]
        type(self)._save_path = st["save_path"]
        type(self)._html = st["html"]
        type(self).active_eval = name
        return True

    def do_GET(self):
        path = urlparse(self.path).path

        # Multi-eval picker
        if self.configs_dir is not None and (path == "/" or path == "/index.html"):
            items = _scan_configs_dir(self.configs_dir)
            html = _picker_html(items)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
            return

        if self.configs_dir is not None and path.startswith("/eval/"):
            name = path[len("/eval/"):].rstrip("/")
            if not self._activate(name):
                self.send_response(404); self.end_headers(); return
            # Inject the eval name so /api/data and /api/annotate are routed
            # to this eval's state rather than whichever was most-recently
            # activated server-wide (race / multi-tab bug fix).
            cfg = self.eval_state[name]["config"]
            randomize_js = "true" if cfg.randomize_order else "false"
            html = self._html.replace(
                "</head>",
                f'<script>window.AUDIT_EVAL = {json.dumps(name)}; '
                f'window.RANDOMIZE_ORDER = {randomize_js};</script></head>',
                1,
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
            return

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self._html.encode())

        elif path == "/api/data":
            qs = parse_qs(urlparse(self.path).query)
            eval_name = qs.get("eval", [None])[0]
            st = self._load_eval(eval_name) if (eval_name and self.configs_dir) else None
            if st is not None:
                rows = st["rows"]
                annotations = st["annotations"]
                cfg = st["config"]
                out_dir = cfg.output_dir
            else:
                rows = self.rows
                annotations = self.annotations
                cfg = self.config
                out_dir = cfg.output_dir if cfg else None
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            definition = _strip_preamble(cfg.judge_prompt_template) if cfg else ""
            references = {}
            if out_dir:
                ref_path = out_dir / "reference_answers.json"
                if ref_path.exists():
                    try:
                        references = json.loads(ref_path.read_text())
                    except Exception:
                        references = {}
            payload = {
                "rows": rows,
                "annotations": annotations,
                "definition": definition,
                "references": references,
            }
            self.wfile.write(json.dumps(payload).encode())

        elif path == "/api/export":
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            _rater_suffix = f"__{_RATER}" if _RATER else ""
            self.send_header(
                "Content-Disposition",
                f"attachment; filename=human_annotations{_rater_suffix}.csv",
            )
            self.end_headers()
            buf = io.StringIO()
            row_keys = [k for k in self.rows[0].keys() if k != "human_label"]
            fieldnames = ["index"] + row_keys + ["human_label"]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(self.rows):
                writer.writerow({"index": i, **{k: row[k] for k in row_keys}, "human_label": self.annotations.get(str(i), "")})
            self.wfile.write(buf.getvalue().encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            idx = str(body["index"])
            lbl = body["label"]
            eval_name = body.get("eval")

            st = self._load_eval(eval_name) if (eval_name and self.configs_dir) else None
            if st is not None:
                rows = st["rows"]
                annotations = st["annotations"]
                save_path = st["save_path"]
            else:
                rows = self.rows
                annotations = self.annotations
                save_path = self._save_path
            annotations[idx] = lbl
            save_annotations(
                rows,
                {int(k): v for k, v in annotations.items()},
                save_path,
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


def _scan_configs_dir(configs_dir: Path) -> list[dict]:
    """Discover audit configs and return summary entries for the picker."""
    items = []
    for cfg_path in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = from_yaml(cfg_path)
        except Exception as e:
            items.append({"name": cfg_path.stem, "status": f"ERROR: {e}", "labeled": 0, "total": 0})
            continue
        try:
            blind = find_blind_csv(cfg)
            with open(blind, newline="", encoding="utf-8") as f:
                total = sum(1 for _ in csv.DictReader(f))
        except FileNotFoundError:
            items.append({"name": cfg_path.stem, "status": "no sample", "labeled": 0, "total": 0})
            continue
        ann_path = save_path(cfg)
        labeled = 0
        if ann_path.exists():
            with open(ann_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    if r.get("human_label"):
                        labeled += 1
        items.append({"name": cfg_path.stem, "status": "ready", "labeled": labeled, "total": total})
    return items


def _picker_html(items: list[dict]) -> str:
    rows = []
    for it in items:
        if it["status"] == "ready":
            link = f'<a href="/eval/{it["name"]}">{it["name"]}</a> [{it["labeled"]}/{it["total"]}]'
        else:
            link = f'<span style="color:#888">{it["name"]} [{it["status"]}: {it["labeled"]}/{it["total"]}]</span>'
        rows.append(f"<li>{link}</li>")
    body = "\n".join(rows)
    return f"""<!DOCTYPE html><html><head><title>Audit picker</title>
<style>body{{font-family:sans-serif;background:#0d1117;color:#c9d1d9;padding:30px;max-width:900px;margin:auto}}
h1{{color:#58a6ff}} li{{padding:6px;font-size:15px}} a{{color:#58a6ff;text-decoration:none}}
a:hover{{text-decoration:underline}}</style></head>
<body><h1>Orthogonalized Audit — Pick an eval</h1>
<p style="color:#8b949e;font-size:13px">Rater: <b>{_RATER or "anon"}</b> &middot; saves to <code>human_annotations__{_RATER or "anon"}.csv</code></p>
<ul>{body}</ul></body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Propensity audit annotation GUI")
    parser.add_argument("--config", default=None, help="Path to a single audit config YAML")
    parser.add_argument("--configs-dir", default=None, help="Directory of audit config YAMLs (multi-eval picker)")
    parser.add_argument("--output-dir", default=None, help="Override: output directory (single-config mode)")
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    parser.add_argument(
        "--rater", default=None,
        help="Rater identifier; scopes save file to "
             "human_annotations__<rater>.csv (default: $AUDIT_RATER, "
             "git config user.name, or $USER)",
    )
    args = parser.parse_args()

    if not args.config and not args.configs_dir:
        parser.error("Specify --config or --configs-dir")

    global _RATER
    _RATER = _resolve_rater(args.rater)
    print(f"Rater: {_RATER}  (annotations save to human_annotations__{_RATER}.csv)")

    if args.configs_dir:
        configs_dir = Path(args.configs_dir).resolve()
        Handler.configs_dir = configs_dir
        Handler.eval_state = {}  # name -> dict(rows, annotations, config, save_path, html)
        print(f"Multi-eval mode: {configs_dir}")
        print(f"Starting server at http://localhost:{args.port}")
        server = HTTPServer(("127.0.0.1", args.port), Handler)
        webbrowser.open(f"http://localhost:{args.port}")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down.")
            server.server_close()
        return

    config = from_yaml(args.config, output_dir=args.output_dir)

    blind_path = find_blind_csv(config)
    ann_path = save_path(config)

    rows, annotations = load_data(blind_path, ann_path)
    html = build_html(config)

    Handler.rows = rows
    Handler.annotations = {str(k): v for k, v in annotations.items()}
    Handler.config = config
    Handler._save_path = ann_path
    Handler._html = html

    n_done = sum(1 for v in annotations.values() if v)
    print(f"Loaded {len(rows)} rows, {n_done} already annotated")
    print(f"Config: {config.display_name} ({config.propensity})")
    print(f"Starting server at http://localhost:{args.port}")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    webbrowser.open(f"http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        n_done = sum(1 for v in Handler.annotations.values() if v)
        print(f"\nSaved {n_done} annotations to {ann_path}")
        server.server_close()


if __name__ == "__main__":
    main()
