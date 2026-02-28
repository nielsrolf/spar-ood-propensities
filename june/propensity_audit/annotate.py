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


def save_path(config: AuditConfig) -> Path:
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
    fieldnames = ["index"] + list(rows[0].keys()) + ["human_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            out = {"index": i, **row, "human_label": annotations.get(i, "")}
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


def _build_button_html(config: AuditConfig) -> str:
    """Generate the label button elements."""
    btns = []
    for i, opt in enumerate(config.all_options):
        cls = _css_safe(opt["label"])
        key_display = opt["key"].upper()
        label = opt["label"]

        # Add separator gap before INCOHERENT
        if opt["label"] == config.INCOHERENT_LABEL:
            btns.append('<span style="width:24px;display:inline-block"></span>')

        btns.append(
            f'<button class="label-btn btn-{cls}" '
            f"onclick=\"label('{label}')\">{label} ({key_display})</button>"
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
        parts.append(f"<b>{opt['key'].upper()}</b> = {opt['label']}")
    return " &middot; ".join(parts) + ' &middot; <b>&larr;/&rarr;</b> = Navigate &middot; <b>U</b> = Next unlabeled'


def _build_metadata_fields(config: AuditConfig) -> str:
    """Generate JS array of metadata column names for the reveal toggle."""
    cols = config.metadata_columns
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

let DATA = [];
let annotations = {{}};
let currentIdx = 0;
let filteredIndices = [];
let filterMode = 'all';
let metaRevealed = false;

async function init() {{
  const resp = await fetch('/api/data');
  const json = await resp.json();
  DATA = json.rows;
  annotations = json.annotations;
  document.getElementById('total-count').textContent = DATA.length;
  applyFilter();
  buildMinimap();
  render();
}}

function applyFilter() {{
  filteredIndices = [];
  for (let i = 0; i < DATA.length; i++) {{
    const lbl = annotations[i] || '';
    if (filterMode === 'all') filteredIndices.push(i);
    else if (filterMode === 'unlabeled' && !lbl) filteredIndices.push(i);
    else if (lbl === filterMode) filteredIndices.push(i);
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
    body: JSON.stringify({{index: currentIdx, label: value}})
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

    def log_message(self, format, *args):
        pass  # suppress request logs

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self._html.encode())

        elif path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {"rows": self.rows, "annotations": self.annotations}
            self.wfile.write(json.dumps(payload).encode())

        elif path == "/api/export":
            self.send_response(200)
            self.send_header("Content-Type", "text/csv")
            self.send_header("Content-Disposition", "attachment; filename=human_annotations.csv")
            self.end_headers()
            buf = io.StringIO()
            fieldnames = ["index"] + list(self.rows[0].keys()) + ["human_label"]
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(self.rows):
                writer.writerow({"index": i, **row, "human_label": self.annotations.get(str(i), "")})
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
            self.annotations[idx] = lbl

            # Persist immediately
            save_annotations(
                self.rows,
                {int(k): v for k, v in self.annotations.items()},
                self._save_path,
            )

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()


def main():
    parser = argparse.ArgumentParser(description="Propensity audit annotation GUI")
    parser.add_argument("--config", required=True, help="Path to audit config YAML")
    parser.add_argument("--output-dir", default=None, help="Override: output directory")
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    args = parser.parse_args()

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
