"""Build per-model-family HTML viewers comparing seed-split finetuned matrices.

One self-contained HTML file per base model is written to
results/seeds/compare_seeds_<base_model>.html. Each file shows one panel per
seed (default, 2, 3, ...), with pink hover-highlighting synchronised across
panels so the same (pole, eval) cell lights up everywhere.

Cell states:
  - coloured     -> numeric mean
  - dark grey    -> pole present in this seed but no numeric mean
  - solid black  -> pole missing from this seed entirely (rendered with a
                    centred dash so it reads visually distinct from null)

Inspired by build_compare.py but trimmed to the bare minimum: no table,
no sysprompt/eval-orth panels, no reduce/normalize/threshold toggles. Just
mean / mean±std / diff-vs-base, and pink hover sync.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from build_compare import (
    DEF_SYS_AXES,
    _cell_metrics,
    _ft_sort_key,
    _x_sort_key,
)

RESULTS = Path(__file__).resolve().parent.parent / "results"
SEEDS_DIR = RESULTS / "seeds"

MODEL_SHORT = {
    "nvidia-NVIDIA-Nemotron-3-Super-120B-A12B-BF16": "Nemotron-3-Super-120B",
    "Qwen-Qwen3-8B-Base": "Qwen3-8B-Base",
    "meta-llama-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
}

# Sentinel used in the matrix to mark "pole not present in this seed at all".
# Distinct from `None` (which the renderer treats as null/no-data dark grey).
MISSING = {"missing": True}

# Filename pattern: scores_<base_model>_seed-<seed>.json
_NAME_RE = re.compile(r"^scores_(?P<model>.+)_seed-(?P<seed>.+)\.json$")


def _discover_per_model() -> dict[str, dict[str, Path]]:
    """Return {base_model: {seed: scores_json_path}}."""
    out: dict[str, dict[str, Path]] = {}
    for p in sorted(SEEDS_DIR.glob("scores_*_seed-*.json")):
        m = _NAME_RE.match(p.name)
        if not m:
            continue
        out.setdefault(m.group("model"), {})[m.group("seed")] = p
    return out


def _seed_sort_key(seed: str) -> tuple:
    """default first, then numeric seeds ascending, then everything else alphabetic."""
    if seed == "default":
        return (0, 0, "")
    try:
        return (1, int(seed), "")
    except ValueError:
        return (2, 0, seed)


def _build_panel(
    seed: str,
    scores_path: Path,
    y_labels: list[str],
    x_labels: list[str],
) -> dict:
    """Slot this seed's cells into the shared (y_labels, x_labels) axes,
    inserting MISSING for poles absent from this seed entirely."""
    data = json.loads(scores_path.read_text())
    cells = data["cells"]
    poles_in_seed = set(cells.keys())

    matrix: list[list] = []
    for pole in y_labels:
        row: list = []
        if pole not in poles_in_seed:
            # Pole has no run for this seed -> every column is "missing".
            row = [MISSING for _ in x_labels]
        else:
            inner = cells[pole]
            for ev in x_labels:
                row.append(_cell_metrics(inner.get(ev)))
        matrix.append(row)

    base_idx = y_labels.index("base") if "base" in y_labels else -1
    return {
        "title": f"seed = {seed}",
        "seed": seed,
        "y_labels": y_labels,
        "x_labels": x_labels,
        "matrix": matrix,
        "base_idx": base_idx,
    }


def _global_ranges(panels: list[dict]) -> dict:
    """Min/max per metric across all panels (used for the shared color scale)."""
    keys = ("m", "s", "mn", "mx")
    acc = {k: [math.inf, -math.inf] for k in keys}
    acc["mms"] = [math.inf, -math.inf]
    acc["mps"] = [math.inf, -math.inf]
    for p in panels:
        for row in p["matrix"]:
            for cell in row:
                if not cell or cell.get("missing"):
                    continue
                for k in keys:
                    v = cell.get(k)
                    if v is None:
                        continue
                    if v < acc[k][0]:
                        acc[k][0] = v
                    if v > acc[k][1]:
                        acc[k][1] = v
                if cell.get("m") is not None and cell.get("s") is not None:
                    lo = cell["m"] - cell["s"]
                    hi = cell["m"] + cell["s"]
                    if lo < acc["mms"][0]:
                        acc["mms"][0] = lo
                    if hi > acc["mps"][1]:
                        acc["mps"][1] = hi
                    if lo < acc["mps"][0]:
                        acc["mps"][0] = lo
                    if hi > acc["mms"][1]:
                        acc["mms"][1] = hi
    # Replace inf with 0..100 fallback if a metric was never seen finite.
    for k, (lo, hi) in acc.items():
        if not math.isfinite(lo) or not math.isfinite(hi):
            acc[k] = [0.0, 100.0]
    return acc


def _diff_range(panels: list[dict]) -> dict:
    """Max |score - base| per metric across panels that have a base row."""
    keys = ("m", "s", "mn", "mx", "mms", "mps")
    acc = {k: 0.0 for k in keys}

    def upd(k: str, v: float | None) -> None:
        if v is None:
            return
        a = abs(v)
        if a > acc[k]:
            acc[k] = a

    for p in panels:
        bi = p["base_idx"]
        if bi < 0:
            continue
        base_row = p["matrix"][bi]
        for r in range(len(p["matrix"])):
            if r == bi:
                continue
            for c in range(len(p["x_labels"])):
                cell = p["matrix"][r][c]
                base = base_row[c]
                if not cell or cell.get("missing"):
                    continue
                if not base or base.get("missing"):
                    continue
                if cell["m"] is not None and base["m"] is not None:
                    upd("m", cell["m"] - base["m"])
                if cell["s"] is not None and base["s"] is not None:
                    upd("s", cell["s"] - base["s"])
                if cell["mn"] is not None and base["mn"] is not None:
                    upd("mn", cell["mn"] - base["mn"])
                if cell["mx"] is not None and base["mx"] is not None:
                    upd("mx", cell["mx"] - base["mx"])
                if (cell["m"] is not None and cell["s"] is not None
                        and base["m"] is not None and base["s"] is not None):
                    upd("mms", (cell["m"] - cell["s"]) - (base["m"] - base["s"]))
                    upd("mps", (cell["m"] + cell["s"]) - (base["m"] + base["s"]))
    for k in keys:
        if acc[k] == 0.0:
            acc[k] = 1.0  # avoid degenerate norm
    return acc


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>compare seeds · __MODEL_TITLE__</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a22;
    --panel-2: #1d2230;
    --ink: #e6e9ef;
    --ink-dim: #8a93a6;
    --accent: #6aa8ff;
    --grid: #2a3142;
    --missing: #000;
    --null: #444;
    --lowfrac: #b0b0b0;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font: 13px/1.4 -apple-system, system-ui, "Segoe UI", sans-serif; }
  header {
    position: sticky; top: 0; z-index: 10;
    display: flex; flex-wrap: wrap; gap: 12px 18px; align-items: center;
    padding: 10px 14px; background: var(--panel); border-bottom: 1px solid var(--grid);
  }
  header h1 { font-size: 14px; margin: 0 16px 0 0; font-weight: 600; }
  header h1 small { color: var(--ink-dim); font-weight: 400; margin-left: 8px; }
  .group { display: inline-flex; gap: 6px; align-items: center; }
  .group label { color: var(--ink-dim); font-size: 12px; }
  .seg { display: inline-flex; background: var(--panel-2); border: 1px solid var(--grid); border-radius: 6px; overflow: hidden; }
  .seg button { background: transparent; color: var(--ink); border: 0; padding: 5px 10px; cursor: pointer; font-size: 12px; }
  .seg button.active { background: var(--accent); color: #0a0d13; font-weight: 600; }
  .toggle { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; font-size: 12px; }
  .toggle input { accent-color: var(--accent); }
  .legend { display: inline-flex; align-items: center; gap: 6px; font-size: 11px; color: var(--ink-dim); }
  .legend-bar { width: 140px; height: 10px; border: 1px solid var(--grid); border-radius: 2px;
    background: linear-gradient(to right, #b30000, #e44d2e, #f0b94f, #fff3b0, #b9e08a, #4eaa50, #136d2c); }
  .legend-bar.diverge { background: linear-gradient(to right, #1a5dab, #6aa8ff, #cfd8e6, #f5f5f5, #f0c4c4, #d24545, #8a1a1a); }
  .legend-sw { display: inline-block; width: 14px; height: 14px; border: 1px solid var(--grid); vertical-align: middle; }
  #grid {
    display: grid; gap: 14px; padding: 14px;
    grid-auto-flow: column; grid-auto-columns: max-content;
    overflow-x: auto;
  }
  .panel { background: var(--panel); border: 1px solid var(--grid); border-radius: 8px; padding: 10px; }
  .panel h3 { font-size: 12px; margin: 0 0 6px; color: var(--ink); font-weight: 600; letter-spacing: .02em; }
  .panel .sub { color: var(--ink-dim); font-size: 11px; margin-bottom: 6px; }
  .heatmap { position: relative; overflow: hidden; }
  .heatmap canvas { display: block; background: #0a0d13; border: 1px solid var(--grid); cursor: crosshair; position: absolute; }
  .heatmap .ax {
    position: absolute; color: var(--ink-dim); font-size: 9px; pointer-events: none;
    user-select: none;
  }
  .heatmap .ax span { position: absolute; white-space: nowrap; }
  .tt {
    position: fixed; z-index: 100; padding: 6px 8px; background: #0a0d13; color: var(--ink);
    border: 1px solid var(--grid); border-radius: 6px; font-size: 11px; pointer-events: none;
    display: none; max-width: 360px;
  }
  .tt b { color: #ffd479; }
  .tt .miss { color: #ff7a90; font-weight: 600; }
  #dispWrap { padding: 14px; }
  #dispWrap h2 { font-size: 13px; margin: 0 0 8px; color: var(--ink); font-weight: 600; }
  #dispWrap .sub { color: var(--ink-dim); font-size: 11px; margin-bottom: 8px; }
  .disp { border-collapse: collapse; font-size: 11px; white-space: nowrap;
    font-variant-numeric: tabular-nums; }
  .disp th, .disp td { border: 1px solid var(--grid); padding: 3px 7px; }
  .disp th { background: var(--panel-2); color: var(--ink-dim); font-weight: 600;
    position: sticky; top: 0; z-index: 2; }
  .disp td { background: var(--panel); color: var(--ink); }
  .disp td.lbl { color: var(--ink-dim); font-size: 10px; }
  .disp td.val { text-align: right; cursor: pointer; }
  .disp td.miss { background: var(--missing); color: #888; text-align: center; }
  .disp td.null { background: var(--null); color: #aaa; text-align: center; }
  .disp td.lowfrac { background: var(--lowfrac); color: #0a0d13; text-align: right; }
  .disp td.rel  { text-align: right; color: #ffce4e; font-weight: 600; }
  .disp td .n   { display: block; font-size: 9px; opacity: 0.75; font-weight: 400; }
  .disp tr.hl   { outline: 2px solid #ff5cbf; outline-offset: -2px; }
</style>
</head>
<body>
<header>
  <h1>compare seeds <small>· __MODEL_TITLE__</small></h1>
  <div class="group">
    <label>view</label>
    <div class="seg" id="metricSeg">
      <button data-metric="mean" class="active">mean</button>
      <button data-metric="meanstd">mean±std</button>
    </div>
  </div>
  <label class="toggle"><input type="checkbox" id="diff"> diff (score − base)</label>
  <label class="toggle"><input type="checkbox" id="normalize"> normalize (÷ std)</label>
  <label class="toggle" title="row qualifies if any pair of seeds satisfies |A−B|/|A| > threshold (uses normalised values when normalize is on)">
    threshold
    <input type="number" id="threshold" min="0" step="0.05" value="0.1"
      style="width: 64px; background: var(--panel-2); color: var(--ink); border: 1px solid var(--grid); border-radius: 4px; padding: 2px 4px;">
  </label>
  <label class="toggle" title="cells with n_numeric/n_total below this ratio render light grey and don't trigger dispersion-table rows (their values are still shown when the row qualifies for other reasons)">
    min n_num/n_tot
    <input type="number" id="minRatio" min="0" max="1" step="0.05" value="0"
      style="width: 60px; background: var(--panel-2); color: var(--ink); border: 1px solid var(--grid); border-radius: 4px; padding: 2px 4px;">
  </label>
  <span class="legend">
    <span class="legend-bar" id="legendBar"></span>
    <small id="legendText">— / —</small>
  </span>
  <span class="legend"><span class="legend-sw" style="background: var(--null)"></span><small>null (no numeric)</small></span>
  <span class="legend"><span class="legend-sw" style="background: var(--missing)"></span><small>missing (no run for this seed)</small></span>
  <span class="legend"><span class="legend-sw" style="background: var(--lowfrac)"></span><small>n_num/n_tot below threshold</small></span>
</header>
<div id="grid"></div>
<div class="tt" id="tt"></div>
<div id="dispWrap">
  <h2>seed dispersion table <span class="sub" id="dispCount"></span></h2>
  <div class="sub">Rows where, across the panels, at least one pair of seeds satisfies
    <code>|A&nbsp;−&nbsp;B|&nbsp;/&nbsp;|A|&nbsp;&gt;&nbsp;threshold</code>.
    Every panel's value is shown regardless of whether it participated in the trigger
    pair. Hover a row to highlight the cell across all matrices.</div>
  <table class="disp" id="dispTable"></table>
</div>
<script>
const MODEL = __MODEL_JSON__;
const PANELS = __PANELS_JSON__;
const RANGES = __RANGES_JSON__;
const DRANGES = __DRANGES_JSON__;

const state = {
  metric: 'mean',
  diff: false,
  normalize: false,
  threshold: 0.1,
  minRatio: 0,
  hoverKey: null,  // 'pole|eval'
};

// Filled by updateNormRanges(); only consulted when state.normalize is on.
const normRange = { abs: [0, 1], diff: 1 };

function shortLabel(s) {
  return s.replace(/_score$/, '').replace(/\s+\(.+?\)$/, '');
}

// Red→yellow→green ramp (matches the PNGs).
const RAMP_SEQ = [
  [0.00, [179, 0, 0]],
  [0.18, [228, 77, 46]],
  [0.36, [240, 185, 79]],
  [0.50, [255, 243, 176]],
  [0.66, [185, 224, 138]],
  [0.82, [78, 170, 80]],
  [1.00, [19, 109, 44]],
];
// Diverging blue→white→red for diff mode.
const RAMP_DIV = [
  [0.00, [26, 93, 171]],
  [0.25, [106, 168, 255]],
  [0.45, [207, 216, 230]],
  [0.50, [245, 245, 245]],
  [0.55, [240, 196, 196]],
  [0.75, [210, 69, 69]],
  [1.00, [138, 26, 26]],
];
function rampSample(ramp, t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < ramp.length; i++) {
    if (t <= ramp[i][0]) {
      const lo = ramp[i-1], hi = ramp[i];
      const k = (t - lo[0]) / (hi[0] - lo[0]);
      const r = Math.round(lo[1][0] + (hi[1][0]-lo[1][0])*k);
      const g = Math.round(lo[1][1] + (hi[1][1]-lo[1][1])*k);
      const b = Math.round(lo[1][2] + (hi[1][2]-lo[1][2])*k);
      return `rgb(${r},${g},${b})`;
    }
  }
  return '#888';
}
function colorFor(v, vmin, vmax, ramp) {
  if (v === null || v === undefined || Number.isNaN(v)) return '#444';
  let t = (v - vmin) / (vmax - vmin);
  if (!isFinite(t)) t = 0.5;
  return rampSample(ramp, t);
}

function rangeForMetric(metric) {
  if (state.normalize) {
    if (state.diff) {
      const a = normRange.diff || 1;
      return [-a, a];
    }
    return normRange.abs;
  }
  if (state.diff) {
    let a;
    if (metric === 'mean') a = DRANGES.m;
    else if (metric === 'meanstd') a = Math.max(DRANGES.mms, DRANGES.mps);
    else a = 1;
    return [-a, a];
  }
  if (metric === 'mean') return RANGES.m;
  if (metric === 'meanstd') return [Math.min(RANGES.mms[0], RANGES.mps[0]),
                                     Math.max(RANGES.mms[1], RANGES.mps[1])];
  return [0, 100];
}

// Pull the cell value(s) for a panel after applying current diff / normalize
// transforms. Returns one of:
//   {kind: 'missing'}                                  -- pole has no run for this seed
//   {kind: 'null'}                                     -- numeric mean unavailable
//   {kind: 'number', v, lo?, hi?, n, nt, reliable}     -- v is the primary value;
//     lo/hi are present only when the mean±std view should draw a split;
//     n/nt carry the underlying counts (or null if missing in the source);
//     reliable=false marks cells whose n/nt ratio is below state.minRatio
//     -- these are still numbers, we just don't trust them.
function panelValue(p, yi, xi) {
  const cell = p.matrix[yi][xi];
  if (cell && cell.missing) return {kind: 'missing'};
  if (!cell || cell.m == null) return {kind: 'null'};

  // Reliability is intrinsic to the cell -- it's the same regardless of
  // diff/normalize. Only mark unreliable when both counts are known.
  let reliable = true;
  if (state.minRatio > 0 && cell.n != null && cell.nt != null && cell.nt > 0) {
    if (cell.n / cell.nt < state.minRatio) reliable = false;
  }
  const meta = {n: cell.n ?? null, nt: cell.nt ?? null, reliable};

  let base = null;
  if (state.diff) {
    if (p.base_idx < 0) return {kind: 'null'};
    const br = p.matrix[p.base_idx][xi];
    if (!br || br.missing || br.m == null) return {kind: 'null'};
    base = br;
  }

  if (state.normalize) {
    if (cell.s == null || cell.s === 0) return {kind: 'null'};
    const v = state.diff ? (cell.m - base.m) / cell.s : cell.m / cell.s;
    // Normalize collapses the std split — there's no meaningful ±std on top
    // of a value already expressed in stds. So lo/hi are intentionally omitted.
    return {kind: 'number', v, ...meta};
  }

  if (state.diff) {
    if (state.metric === 'meanstd' && cell.s != null && base.s != null) {
      return {
        kind: 'number',
        v: cell.m - base.m,
        lo: (cell.m - cell.s) - (base.m - base.s),
        hi: (cell.m + cell.s) - (base.m + base.s),
        ...meta,
      };
    }
    return {kind: 'number', v: cell.m - base.m, ...meta};
  }
  if (state.metric === 'meanstd' && cell.s != null) {
    return {kind: 'number', v: cell.m, lo: cell.m - cell.s, hi: cell.m + cell.s, ...meta};
  }
  return {kind: 'number', v: cell.m, ...meta};
}

function updateNormRanges() {
  let diffMax = 0, absLo = Infinity, absHi = -Infinity;
  for (const p of PANELS) {
    const hasBase = p.base_idx >= 0;
    for (let yi = 0; yi < p.y_labels.length; yi++) {
      for (let xi = 0; xi < p.x_labels.length; xi++) {
        const cell = p.matrix[yi][xi];
        if (!cell || cell.missing || cell.m == null || cell.s == null || cell.s === 0) continue;
        const abs = cell.m / cell.s;
        if (abs < absLo) absLo = abs;
        if (abs > absHi) absHi = abs;
        if (hasBase && yi !== p.base_idx) {
          const br = p.matrix[p.base_idx][xi];
          if (br && !br.missing && br.m != null) {
            const d = Math.abs((cell.m - br.m) / cell.s);
            if (d > diffMax) diffMax = d;
          }
        }
      }
    }
  }
  normRange.abs  = [isFinite(absLo) ? absLo : 0, isFinite(absHi) ? absHi : 1];
  normRange.diff = diffMax || 1;
}

// rampForMode is defined later (after state.normalize is in scope); the
// later declaration is what actually ends up bound at runtime.

function setHoverKey(k) {
  if (state.hoverKey === k) return;
  state.hoverKey = k;
  for (const pv of panelViews) pv.render();
}

class PanelView {
  constructor(panel, container) {
    this.p = panel;
    this.ny = panel.y_labels.length;
    this.nx = panel.x_labels.length;
    this.container = container;
    this.cellW = 18;
    this.cellH = 16;
    this.gutterLeft = 170;
    this.gutterTop = 130;
    this.build();
    this.attach();
  }
  build() {
    const W = this.cellW * this.nx;
    const H = this.cellH * this.ny;
    this.container.innerHTML = `
      <h3>${this.p.title}</h3>
      <div class="sub">${this.ny} poles × ${this.nx} evals</div>
      <div class="heatmap" style="width:${W + this.gutterLeft}px;height:${H + this.gutterTop}px">
        <canvas width="${W}" height="${H}"
          style="left:${this.gutterLeft}px;top:${this.gutterTop}px"></canvas>
        <div class="ax axLeft" style="left:0;top:${this.gutterTop}px;width:${this.gutterLeft}px;height:${H}px"></div>
        <div class="ax axTop"  style="left:${this.gutterLeft}px;top:0;width:${W}px;height:${this.gutterTop}px"></div>
      </div>
    `;
    this.canvas = this.container.querySelector('canvas');
    this.axLeft = this.container.querySelector('.axLeft');
    this.axTop  = this.container.querySelector('.axTop');
    this.W = W;
    this.H = H;
  }
  attach() {
    const tt = document.getElementById('tt');
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const xi = Math.floor(x / this.cellW);
      const yi = Math.floor(y / this.cellH);
      if (xi < 0 || xi >= this.nx || yi < 0 || yi >= this.ny) {
        tt.style.display = 'none';
        setHoverKey(null);
        return;
      }
      const pole = this.p.y_labels[yi];
      const ev   = this.p.x_labels[xi];
      const cell = this.p.matrix[yi][xi];
      setHoverKey(`${pole}|${ev}`);
      tt.style.display = 'block';
      tt.style.left = (e.clientX + 12) + 'px';
      tt.style.top  = (e.clientY + 12) + 'px';
      if (!cell) {
        tt.innerHTML = `<b>${pole}</b> × <b>${ev}</b><br>seed = ${this.p.seed}<br><i>no data</i>`;
      } else if (cell.missing) {
        tt.innerHTML = `<b>${pole}</b> × <b>${ev}</b><br>seed = ${this.p.seed}<br>` +
          `<span class="miss">missing — no run for this (pole, seed) in the index</span>`;
      } else {
        const fm = (v) => (v == null ? '—' : (typeof v === 'number' ? v.toFixed(2) : v));
        tt.innerHTML = `<b>${pole}</b> × <b>${ev}</b><br>seed = ${this.p.seed}<br>` +
          `mean=${fm(cell.m)}  std=${fm(cell.s)}<br>min=${fm(cell.mn)}  max=${fm(cell.mx)}<br>` +
          `n_numeric=${fm(cell.n)}  n_total=${fm(cell.nt)}  n_null=${fm(cell.nn)}  n_fails=${fm(cell.nf)}`;
      }
    });
    this.canvas.addEventListener('mouseleave', () => {
      tt.style.display = 'none';
      setHoverKey(null);
    });
  }
  render() {
    const ctx = this.canvas.getContext('2d');
    ctx.clearRect(0, 0, this.W, this.H);
    const [vmin, vmax] = rangeForMetric(state.metric);
    const ramp = rampForMode();

    for (let yi = 0; yi < this.ny; yi++) {
      for (let xi = 0; xi < this.nx; xi++) {
        const px = xi * this.cellW;
        const py = yi * this.cellH;
        const pv = panelValue(this.p, yi, xi);

        if (pv.kind === 'missing') {
          ctx.fillStyle = '#000';
          ctx.fillRect(px, py, this.cellW, this.cellH);
          ctx.strokeStyle = '#666';
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(px + this.cellW * 0.3, py + this.cellH * 0.5);
          ctx.lineTo(px + this.cellW * 0.7, py + this.cellH * 0.5);
          ctx.stroke();
        } else if (pv.kind === 'null') {
          ctx.fillStyle = '#444';
          ctx.fillRect(px, py, this.cellW, this.cellH);
        } else if (pv.reliable === false) {
          // Below the n_num/n_tot threshold — render the cell light grey so
          // it reads as "we have a number but don't trust the sample size".
          ctx.fillStyle = '#b0b0b0';
          ctx.fillRect(px, py, this.cellW, this.cellH);
        } else if (pv.lo != null && pv.hi != null) {
          drawSplit(ctx, px, py, this.cellW, this.cellH,
            colorFor(pv.lo, vmin, vmax, ramp),
            colorFor(pv.hi, vmin, vmax, ramp));
        } else {
          ctx.fillStyle = colorFor(pv.v, vmin, vmax, ramp);
          ctx.fillRect(px, py, this.cellW, this.cellH);
        }

        // pink hover sync
        if (state.hoverKey) {
          const k = this.p.y_labels[yi] + '|' + this.p.x_labels[xi];
          if (k === state.hoverKey) {
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#ff5cbf';
            ctx.strokeRect(px + 1, py + 1, this.cellW - 2, this.cellH - 2);
          }
        }
      }
    }
    this.renderAxes();
  }
  renderAxes() {
    this.axLeft.innerHTML = '';
    this.axTop.innerHTML = '';
    for (let yi = 0; yi < this.ny; yi++) {
      const py = yi * this.cellH + this.cellH / 2 - 6;
      const s = document.createElement('span');
      s.textContent = shortLabel(this.p.y_labels[yi]);
      s.style.right = '4px';
      s.style.top = py + 'px';
      s.style.textAlign = 'right';
      s.style.width = (this.gutterLeft - 6) + 'px';
      s.style.overflow = 'hidden';
      s.style.textOverflow = 'ellipsis';
      this.axLeft.appendChild(s);
    }
    for (let xi = 0; xi < this.nx; xi++) {
      const px = xi * this.cellW + this.cellW / 2;
      const s = document.createElement('span');
      s.textContent = shortLabel(this.p.x_labels[xi]);
      s.style.left = px + 'px';
      s.style.bottom = '4px';
      s.style.transform = 'rotate(-60deg)';
      s.style.transformOrigin = '0 100%';
      this.axTop.appendChild(s);
    }
  }
}

function drawSplit(ctx, x, y, w, h, c1, c2) {
  // upper-right triangle = c2 (mean + std), lower-left triangle = c1 (mean - std)
  ctx.fillStyle = c2;
  ctx.fillRect(x, y, w, h);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x + w, y + h);
  ctx.lineTo(x, y + h);
  ctx.closePath();
  ctx.fillStyle = c1;
  ctx.fill();
}

function updateLegend() {
  const [vmin, vmax] = rangeForMetric(state.metric);
  const bar = document.getElementById('legendBar');
  // Diff and normalize both use the diverging ramp (normalize centres on 0 too).
  const diverge = state.diff || state.normalize;
  bar.classList.toggle('diverge', diverge);
  document.getElementById('legendText').textContent =
    `${vmin.toFixed(2)} → ${vmax.toFixed(2)}`;
}

// Normalize centres the ramp on 0 (effect-size in stds), same as diff. Override
// the existing rampForMode now that normalize is in play.
function rampForMode() {
  return (state.diff || state.normalize) ? RAMP_DIV : RAMP_SEQ;
}

// ---------- Dispersion table ----------

// For each (yi, xi), collect the panel-display values, compute the worst
// pairwise |A−B|/|A| ratio across *reliable* numeric values, and keep the row
// when the ratio exceeds state.threshold. Always returns ALL panel values per
// row so the caller can render the full strip regardless of which pair
// triggered -- unreliable cells just don't count toward maxRel themselves.
function dispersionRows() {
  const ny = PANELS[0].y_labels.length;
  const nx = PANELS[0].x_labels.length;
  const rows = [];
  for (let yi = 0; yi < ny; yi++) {
    for (let xi = 0; xi < nx; xi++) {
      const vals = PANELS.map(p => panelValue(p, yi, xi));
      const numIdx = [];
      for (let i = 0; i < vals.length; i++) {
        const v = vals[i];
        if (v.kind !== 'number' || !isFinite(v.v)) continue;
        if (v.reliable === false) continue;  // below n_num/n_tot — exclude from comparison
        numIdx.push(i);
      }
      if (numIdx.length < 2) continue;
      let maxRel = 0;
      for (const a of numIdx) {
        const va = vals[a].v;
        if (va === 0) continue;
        for (const b of numIdx) {
          if (b === a) continue;
          const vb = vals[b].v;
          const rel = Math.abs(va - vb) / Math.abs(va);
          if (rel > maxRel) maxRel = rel;
        }
      }
      if (maxRel > state.threshold) {
        rows.push({yi, xi, vals, maxRel});
      }
    }
  }
  rows.sort((a, b) => b.maxRel - a.maxRel);
  return rows;
}

function renderTable() {
  const tbl = document.getElementById('dispTable');
  const count = document.getElementById('dispCount');
  const rows = dispersionRows();
  count.textContent = ` — ${rows.length} rows · threshold = ${state.threshold}` +
    (state.normalize ? ' (normalised)' : '') + (state.diff ? ' (diff)' : '');
  const [vmin, vmax] = rangeForMetric(state.metric);
  const ramp = rampForMode();

  // Header: pole, eval, one column per panel (seed=...), max rel diff.
  const yLabels = PANELS[0].y_labels;
  const xLabels = PANELS[0].x_labels;
  let html = '<thead><tr><th>pole</th><th>eval</th>';
  for (const p of PANELS) html += `<th>seed = ${p.seed}</th>`;
  html += '<th>max rel diff</th></tr></thead><tbody>';

  // Cap rendered rows to keep the table responsive on large grids.
  const cap = 800;
  const shown = rows.slice(0, cap);
  const fmtN = (v) => {
    if (v.n == null && v.nt == null) return '';
    const n  = v.n  == null ? '?' : v.n;
    const nt = v.nt == null ? '?' : v.nt;
    return `<span class="n">${n}/${nt}</span>`;
  };
  for (const r of shown) {
    const pole = yLabels[r.yi];
    const ev   = xLabels[r.xi];
    const hk = `${pole}|${ev}`;
    html += `<tr data-hk="${hk}"><td class="lbl">${pole}</td><td class="lbl">${ev}</td>`;
    for (const v of r.vals) {
      if (v.kind === 'missing') {
        html += `<td class="miss">—</td>`;
      } else if (v.kind === 'null') {
        html += `<td class="null">null</td>`;
      } else if (v.reliable === false) {
        // Show the number, but light grey to flag that it didn't participate
        // in the pair-comparison that triggered this row.
        html += `<td class="lowfrac">${v.v.toFixed(2)}${fmtN(v)}</td>`;
      } else {
        const col = colorFor(v.v, vmin, vmax, ramp);
        html += `<td class="val" style="background:${col};color:#0a0d13">${v.v.toFixed(2)}${fmtN(v)}</td>`;
      }
    }
    html += `<td class="rel">${(r.maxRel * 100).toFixed(1)}%</td></tr>`;
  }
  if (rows.length > cap) {
    html += `<tr><td colspan="${PANELS.length + 3}" style="color:var(--ink-dim);text-align:center">… ${rows.length - cap} more rows hidden — raise threshold to narrow</td></tr>`;
  }
  html += '</tbody>';
  tbl.innerHTML = html;
}

// Hover table row → propagate to matrices via the shared hover key.
document.getElementById('dispTable').addEventListener('mousemove', (e) => {
  const tr = e.target.closest('tr[data-hk]');
  setHoverKey(tr ? tr.dataset.hk : null);
});
document.getElementById('dispTable').addEventListener('mouseleave', () => setHoverKey(null));

// Override the early setHoverKey definition so it also outlines the matching
// table row. The original is replaced cleanly because it was a script-level
// function declaration, not a const.
setHoverKey = function(k) {
  if (state.hoverKey === k) return;
  state.hoverKey = k;
  for (const pv of panelViews) pv.render();
  const tbl = document.getElementById('dispTable');
  if (tbl) {
    for (const tr of tbl.querySelectorAll('tr.hl')) tr.classList.remove('hl');
    if (k) {
      const tr = tbl.querySelector(`tr[data-hk="${CSS.escape(k)}"]`);
      if (tr) tr.classList.add('hl');
    }
  }
};

function refreshAll() {
  if (state.normalize) updateNormRanges();
  updateLegend();
  for (const pv of panelViews) pv.render();
  renderTable();
}

document.getElementById('metricSeg').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-metric]');
  if (!b) return;
  state.metric = b.dataset.metric;
  document.querySelectorAll('#metricSeg button').forEach(x => x.classList.toggle('active', x === b));
  refreshAll();
});
document.getElementById('diff').addEventListener('change', (e) => {
  state.diff = e.target.checked;
  refreshAll();
});
document.getElementById('normalize').addEventListener('change', (e) => {
  state.normalize = e.target.checked;
  refreshAll();
});
document.getElementById('threshold').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  state.threshold = isFinite(v) ? Math.max(0, v) : 0;
  renderTable();
});
document.getElementById('minRatio').addEventListener('input', (e) => {
  const v = parseFloat(e.target.value);
  state.minRatio = isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
  // Affects matrix colouring (light grey cells) AND dispersion filtering AND
  // the per-row marker in the table — refresh both views.
  for (const pv of panelViews) pv.render();
  renderTable();
});

const grid = document.getElementById('grid');
const panelViews = PANELS.map((p) => {
  const div = document.createElement('div');
  div.className = 'panel';
  grid.appendChild(div);
  return new PanelView(p, div);
});
updateNormRanges();
updateLegend();
for (const pv of panelViews) pv.render();
renderTable();
</script>
</body>
</html>
"""


def _render_html(base_model: str, panels: list[dict]) -> str:
    return (
        HTML_TEMPLATE
        .replace("__MODEL_TITLE__", MODEL_SHORT.get(base_model, base_model))
        .replace("__MODEL_JSON__", json.dumps(base_model))
        .replace("__PANELS_JSON__", json.dumps(panels))
        .replace("__RANGES_JSON__", json.dumps(_global_ranges(panels)))
        .replace("__DRANGES_JSON__", json.dumps(_diff_range(panels)))
    )


def main() -> None:
    if not SEEDS_DIR.exists():
        raise SystemExit(f"seeds dir not found: {SEEDS_DIR}")

    by_model = _discover_per_model()
    if not by_model:
        raise SystemExit(f"no scores_*_seed-*.json files in {SEEDS_DIR}")

    for base_model, by_seed in sorted(by_model.items()):
        seeds = sorted(by_seed, key=_seed_sort_key)
        print(f"=== {base_model} ({len(seeds)} seeds: {seeds}) ===")

        # Union of y / x labels across all seeds, in canonical order.
        all_poles: set[str] = set()
        all_evals: set[str] = set()
        for s in seeds:
            doc = json.loads(by_seed[s].read_text())
            for pole, by_ev in doc["cells"].items():
                all_poles.add(pole)
                all_evals.update(by_ev.keys())
        y_labels = sorted(all_poles, key=_ft_sort_key)
        x_labels = sorted(all_evals, key=_x_sort_key)
        print(f"  union axes: {len(y_labels)} poles × {len(x_labels)} evals")

        panels = [_build_panel(s, by_seed[s], y_labels, x_labels) for s in seeds]

        # Report missing-pole counts per panel as a sanity check.
        for p in panels:
            missing_poles = [
                y_labels[i] for i, row in enumerate(p["matrix"])
                if all(c is not None and c.get("missing") for c in row)
            ]
            if missing_poles:
                print(f"  seed={p['seed']}: {len(missing_poles)} poles rendered black "
                      f"(missing this seed): {missing_poles}")
            else:
                print(f"  seed={p['seed']}: no missing poles")

        out = SEEDS_DIR / f"compare_seeds_{base_model}.html"
        out.write_text(_render_html(base_model, panels))
        print(f"  [wrote] {out.name}")


if __name__ == "__main__":
    main()
