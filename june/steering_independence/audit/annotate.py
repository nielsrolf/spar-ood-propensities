#!/usr/bin/env python3
"""
Annotation GUI for steering independence audit.

Shows the trait being evaluated and reference answers (high/low) for that
question, but hides source/source_type to keep the annotation blind.

Usage:
    python audit/annotate.py [--audit trait|coherence] [--port 8780]
"""

import argparse
import csv
import io
import json
import os
import sys
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_STEERING = _HERE.parent
sys.path.insert(0, str(_STEERING))

from trait_registry import (
    ALL_TRAITS, LABELS, TRAIT_KEYS, load_contrastive_pairs,
)

PORT = 8780

# ── Reference answer lookup ──────────────────────────────────────────

def build_reference_lookup() -> dict:
    """Build {trait: {question_id: {high, low}}} from questions.json."""
    lookup = {}
    for trait in ALL_TRAITS:
        lookup[trait] = {}
        # Try both splits — questions may be in train or test
        for split in ("train", "test"):
            for pair in load_contrastive_pairs(trait, split):
                lookup[trait][pair["id"]] = {
                    "high": pair["high_response"],
                    "low": pair["low_response"],
                }
    return lookup


# ── Data loading ─────────────────────────────────────────────────────

def find_blind_csv(out_dir: Path) -> Path:
    candidates = sorted(out_dir.glob("sample_*_blind.csv"))
    if candidates:
        return candidates[-1]
    raise FileNotFoundError(f"No sample_*_blind.csv in {out_dir}")


def load_data(blind_path: Path, ann_path: Path, ref_lookup: dict):
    """Load rows, strip source info, inject trait label + reference answers."""
    rows = []
    with open(blind_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            trait = r.get("target", "")
            qid = r.get("question_id", "")

            # Look up reference answers
            refs = ref_lookup.get(trait, {}).get(qid, {})

            rows.append({
                "question": r["question"],
                "response": r["response"],
                "trait": LABELS.get(trait, trait),
                "trait_key": trait,
                "question_id": qid,
                "ref_high": refs.get("high", ""),
                "ref_low": refs.get("low", ""),
            })

    annotations = {}
    if ann_path.exists():
        with open(ann_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                annotations[int(r["index"])] = r.get("human_label", "")

    return rows, annotations


def save_annotations(rows, annotations, path: Path):
    fieldnames = ["index", "question", "response", "trait", "trait_key",
                   "question_id", "human_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            writer.writerow({
                "index": i,
                "question": row["question"],
                "response": row["response"],
                "trait": row["trait"],
                "trait_key": row["trait_key"],
                "question_id": row["question_id"],
                "human_label": annotations.get(i, ""),
            })


# ── Audit configs ────────────────────────────────────────────────────

AUDIT_CONFIGS = {
    "trait": {
        "title": "Steering Independence — Trait Score Audit",
        "buckets": [
            {"num": 5, "label": "Very High Trait",  "short": "VH",  "key": "a", "color": "#da3633"},
            {"num": 4, "label": "High Trait",        "short": "HI",  "key": "s", "color": "#e3795b"},
            {"num": 3, "label": "Balanced",           "short": "BAL", "key": "d", "color": "#9e6a03"},
            {"num": 2, "label": "Low Trait",          "short": "LO",  "key": "f", "color": "#58a6ff"},
            {"num": 1, "label": "Very Low Trait",     "short": "VL",  "key": "g", "color": "#238636"},
        ],
    },
    "coherence": {
        "title": "Steering Independence — Coherence Score Audit",
        "buckets": [
            {"num": 5, "label": "Fully Coherent",     "short": "FC",  "key": "a", "color": "#238636"},
            {"num": 4, "label": "Mostly Coherent",     "short": "MC",  "key": "s", "color": "#58a6ff"},
            {"num": 3, "label": "Partially Coherent",  "short": "PC",  "key": "d", "color": "#9e6a03"},
            {"num": 2, "label": "Mostly Incoherent",   "short": "MI",  "key": "f", "color": "#e3795b"},
            {"num": 1, "label": "Fully Incoherent",    "short": "FI",  "key": "g", "color": "#da3633"},
        ],
    },
}

INCOHERENT = {"label": "INCOHERENT", "short": "INC", "key": "h", "color": "#6e7681"}


# ── HTML builder ─────────────────────────────────────────────────────

def build_html(cfg: dict) -> str:
    buckets = cfg["buckets"]
    all_options = buckets + [INCOHERENT]
    title = cfg["title"]

    # CSS for buttons
    btn_css = []
    for opt in all_options:
        cls = opt["label"].lower().replace(" ", "-").replace("_", "-")
        c = opt["color"]
        btn_css.append(f"""
  .btn-{cls} {{ background: {c}; color: #fff; }}
  .btn-{cls}:hover {{ background: {c}; filter: brightness(1.2); }}
  .btn-{cls}.selected {{ border-color: #fff; box-shadow: 0 0 12px {c}88; }}
  .dot-{cls} {{ background: {c}; }}""")

    # Button HTML
    btn_html = []
    for opt in all_options:
        cls = opt["label"].lower().replace(" ", "-").replace("_", "-")
        if opt is INCOHERENT:
            btn_html.append('<span style="width:24px;display:inline-block"></span>')
        btn_html.append(
            f'<button class="label-btn btn-{cls}" '
            f"onclick=\"label('{opt['label']}')\">{opt['label']} ({opt['key'].upper()})</button>"
        )

    # Color map
    color_map = {opt["label"]: opt["color"] for opt in all_options}

    # Key cases
    key_cases = []
    for opt in all_options:
        k = opt["key"]
        key_cases.append(f"    case '{k}': case '{k.upper()}': label('{opt['label']}'); break;")

    # Filter buttons
    filter_btns = [
        '<button class="filter-btn active" onclick="setFilter(\'all\')">All</button>',
        '<button class="filter-btn" onclick="setFilter(\'unlabeled\')">Unlabeled</button>',
    ]
    for opt in all_options:
        filter_btns.append(
            f'<button class="filter-btn" onclick="setFilter(\'{opt["label"]}\')">{opt["short"]}</button>'
        )

    # Keyboard hint
    hint_parts = [f"<b>{o['key'].upper()}</b> = {o['label']}" for o in all_options]
    keyboard_hint = " &middot; ".join(hint_parts) + \
        ' &middot; <b>&larr;/&rarr;</b> Nav &middot; <b>U</b> Next unlabeled &middot; <b>R</b> Toggle refs'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
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

  .container {{ max-width: 960px; margin: 0 auto; padding: 24px; }}

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

  .trait-banner {{ background: #1f6feb22; border: 1px solid #1f6feb55; border-radius: 8px;
                   padding: 10px 16px; margin-bottom: 16px; display: flex;
                   align-items: center; justify-content: space-between; }}
  .trait-name {{ font-size: 16px; font-weight: 600; color: #58a6ff; }}
  .trait-meta {{ font-size: 12px; color: #8b949e; }}

  .meta {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 12px; }}
  .tag {{ background: #21262d; border: 1px solid #30363d; padding: 3px 10px;
         border-radius: 12px; font-size: 12px; color: #8b949e; }}
  .status-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                margin-right: 4px; vertical-align: middle; }}
  .dot-empty {{ background: #484f58; }}

  .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 20px; margin-bottom: 16px; }}
  .card h3 {{ font-size: 13px; color: #8b949e; margin-bottom: 8px; text-transform: uppercase;
             letter-spacing: 0.5px; }}
  .card .text {{ font-size: 14px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
  .question-text {{ color: #58a6ff; }}
  .response-text {{ color: #c9d1d9; }}

  .ref-section {{ display: none; margin-bottom: 16px; }}
  .ref-section.visible {{ display: block; }}
  .ref-cards {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  .ref-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }}
  .ref-card h3 {{ margin-bottom: 8px; }}
  .ref-high h3 {{ color: #da3633; }}
  .ref-low h3 {{ color: #238636; }}
  .ref-card .text {{ font-size: 13px; line-height: 1.5; color: #8b949e;
                     max-height: 200px; overflow-y: auto; }}
  .ref-toggle {{ background: #21262d; color: #8b949e; border: 1px solid #30363d;
                padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 12px;
                margin-bottom: 12px; }}
  .ref-toggle:hover {{ background: #30363d; color: #c9d1d9; }}

  .label-buttons {{ display: flex; gap: 12px; margin-top: 20px; justify-content: center;
                    flex-wrap: wrap; }}
  .label-btn {{ padding: 12px 24px; border-radius: 8px; border: 2px solid transparent;
               font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.15s; }}
  .label-btn:hover {{ transform: translateY(-1px); }}

{"".join(btn_css)}

  .keyboard-hint {{ text-align: center; margin-top: 12px; font-size: 12px; color: #484f58; }}

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
  <div class="progress"><span id="progress-text">0 / 0</span></div>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
</div>

<div class="container">
  <div class="nav">
    <button onclick="go(-1)">&larr; Prev</button>
    <input type="number" id="idx-input" min="1" value="1" onchange="jumpTo(this.value-1)">
    <span class="total">/ <span id="total-count">0</span></span>
    <button onclick="go(1)">Next &rarr;</button>
    <span style="flex:1"></span>
    <button onclick="goToNextUnlabeled()">Next unlabeled</button>
    <button onclick="exportCSV()">Export CSV</button>
  </div>

  <div class="filter-row">
    <span style="font-size:12px;color:#8b949e;line-height:24px;">Show:</span>
    {"".join(filter_btns)}
  </div>

  <div class="trait-banner">
    <span class="trait-name" id="trait-name"></span>
    <span class="trait-meta" id="trait-status"></span>
  </div>

  <div class="card">
    <h3>Question</h3>
    <div class="text question-text" id="question"></div>
  </div>

  <div class="card">
    <h3>Response</h3>
    <div class="text response-text" id="response"></div>
  </div>

  <button class="ref-toggle" id="ref-toggle" onclick="toggleRefs()">Show Reference Answers (R)</button>
  <div class="ref-section" id="ref-section">
    <div class="ref-cards">
      <div class="ref-card ref-high">
        <h3>High Trait Reference</h3>
        <div class="text" id="ref-high"></div>
      </div>
      <div class="ref-card ref-low">
        <h3>Low Trait Reference</h3>
        <div class="text" id="ref-low"></div>
      </div>
    </div>
  </div>

  <div class="label-buttons">
    {"".join(btn_html)}
  </div>
  <div class="keyboard-hint">
    {keyboard_hint}
  </div>

  <div class="minimap" id="minimap"></div>
</div>

<script>
const COLOR_MAP = {json.dumps(color_map)};

let DATA = [];
let annotations = {{}};
let currentIdx = 0;
let filteredIndices = [];
let filterMode = 'all';
let refsVisible = false;

async function init() {{
  const resp = await fetch('/api/data');
  const j = await resp.json();
  DATA = j.rows;
  annotations = j.annotations;
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
  if (!filteredIndices.includes(currentIdx))
    currentIdx = filteredIndices.length > 0 ? filteredIndices[0] : 0;
  render();
}}

function cssClass(label) {{
  return label.toLowerCase().replace(/ /g, '-').replace(/_/g, '-');
}}

function toggleRefs() {{
  refsVisible = !refsVisible;
  document.getElementById('ref-section').classList.toggle('visible', refsVisible);
  document.getElementById('ref-toggle').textContent =
    refsVisible ? 'Hide Reference Answers (R)' : 'Show Reference Answers (R)';
}}

function render() {{
  if (DATA.length === 0) return;
  const row = DATA[currentIdx];
  const lbl = annotations[currentIdx] || '';

  document.getElementById('trait-name').textContent = row.trait;
  const dotClass = lbl ? 'dot-' + cssClass(lbl) : 'dot-empty';
  document.getElementById('trait-status').innerHTML =
    '<span class="status-dot ' + dotClass + '"></span>' + (lbl || 'unlabeled');

  document.getElementById('question').textContent = row.question;
  document.getElementById('response').textContent = row.response;
  document.getElementById('idx-input').value = currentIdx + 1;

  // Reference answers
  document.getElementById('ref-high').textContent = row.ref_high || '(not available)';
  document.getElementById('ref-low').textContent = row.ref_low || '(not available)';

  // Highlight selected button
  document.querySelectorAll('.label-btn').forEach(b => b.classList.remove('selected'));
  if (lbl) {{
    const btn = document.querySelector('.btn-' + cssClass(lbl));
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
  const pos = filteredIndices.indexOf(currentIdx);
  if (pos >= 0 && pos < filteredIndices.length - 1)
    currentIdx = filteredIndices[pos + 1];
  render();
}}

function go(delta) {{
  const pos = filteredIndices.indexOf(currentIdx);
  let np = pos + delta;
  if (np < 0) np = 0;
  if (np >= filteredIndices.length) np = filteredIndices.length - 1;
  if (filteredIndices.length > 0) currentIdx = filteredIndices[np];
  render();
}}

function jumpTo(idx) {{
  idx = parseInt(idx);
  if (idx >= 0 && idx < DATA.length) {{ currentIdx = idx; render(); }}
}}

function goToNextUnlabeled() {{
  for (let i = currentIdx + 1; i < DATA.length; i++)
    if (!annotations[i]) {{ currentIdx = i; render(); return; }}
  for (let i = 0; i < currentIdx; i++)
    if (!annotations[i]) {{ currentIdx = i; render(); return; }}
  alert('All rows are labeled!');
}}

function exportCSV() {{ window.open('/api/export', '_blank'); }}

document.addEventListener('keydown', (e) => {{
  if (e.target.tagName === 'INPUT') return;
  switch(e.key) {{
{"".join(key_cases)}
    case 'ArrowLeft':  go(-1); e.preventDefault(); break;
    case 'ArrowRight': go(1);  e.preventDefault(); break;
    case 'u': case 'U': goToNextUnlabeled(); break;
    case 'r': case 'R': toggleRefs(); break;
  }}
}});

init();
</script>
</body>
</html>"""


# ── HTTP Server ──────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    rows = []
    annotations = {}
    _save_path = None
    _html = ""

    def log_message(self, fmt, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]

        if path in ("/", "/index.html"):
            self._respond(200, "text/html", self._html.encode())
        elif path == "/api/data":
            payload = {"rows": self.rows, "annotations": self.annotations}
            self._respond(200, "application/json", json.dumps(payload).encode())
        elif path == "/api/export":
            buf = io.StringIO()
            fnames = ["index", "question", "response", "trait", "trait_key",
                       "question_id", "human_label"]
            writer = csv.DictWriter(buf, fieldnames=fnames)
            writer.writeheader()
            for i, row in enumerate(self.rows):
                writer.writerow({
                    "index": i,
                    "question": row["question"],
                    "response": row["response"],
                    "trait": row["trait"],
                    "trait_key": row["trait_key"],
                    "question_id": row["question_id"],
                    "human_label": self.annotations.get(str(i), ""),
                })
            self._respond(200, "text/csv", buf.getvalue().encode(),
                          extra={"Content-Disposition": "attachment; filename=human_annotations.csv"})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path.split("?")[0] == "/api/annotate":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            self.annotations[str(body["index"])] = body["label"]
            save_annotations(
                self.rows,
                {int(k): v for k, v in self.annotations.items()},
                self._save_path,
            )
            self._respond(200, "application/json", b'{"ok":true}')
        else:
            self.send_response(404); self.end_headers()

    def _respond(self, code, ctype, data, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", f"{ctype}; charset=utf-8")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(data)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Steering independence audit annotation")
    parser.add_argument("--audit", default="trait", choices=["trait", "coherence"])
    parser.add_argument("--output-dir", default=None, help="Override output base dir")
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()

    base = Path(args.output_dir) if args.output_dir else _STEERING / "outputs"
    output_dir = base / "audit" / args.audit
    cfg = AUDIT_CONFIGS[args.audit]

    print("Loading reference answers...")
    ref_lookup = build_reference_lookup()

    blind_path = find_blind_csv(output_dir)
    ann_path = output_dir / "human_annotations.csv"

    rows, annotations = load_data(blind_path, ann_path, ref_lookup)
    html = build_html(cfg)

    Handler.rows = rows
    Handler.annotations = {str(k): v for k, v in annotations.items()}
    Handler._save_path = ann_path
    Handler._html = html

    n_done = sum(1 for v in annotations.values() if v)
    print(f"Loaded {len(rows)} rows, {n_done} already annotated")
    print(f"Audit: {cfg['title']}")

    # Count per trait
    from collections import Counter
    trait_counts = Counter(r["trait"] for r in rows)
    for t, c in sorted(trait_counts.items()):
        print(f"  {t}: {c}")

    print(f"\nStarting server at http://localhost:{args.port}")
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
