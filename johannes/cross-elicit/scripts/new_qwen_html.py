"""new_qwen_html.py — render new_eval_results/qwen_dpo_compare.html.

Three panels: Qwen3-8B-Base system-prompted, finetuned, and DPO.
Rows and columns are restricted to the 9 traits present in the DPO CSV export.
DPO 'trained_trait' values are treated as the plus-pole of each axis.

Usage:
    python new_qwen_html.py [path/to/dpo_summary.csv]

If no CSV path is given, the default path (DPO_CSV_DEFAULT) is used.
"""

from __future__ import annotations

import csv as csvmod
import json
import sys
from pathlib import Path

import build_compare as bc
import new_build_compare as nbc  # also triggers all module-level patches to bc / core

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
SCORES_DIR = ROOT / "new_eval_results" / "scores"
OUT = ROOT / "new_eval_results" / "qwen_dpo_compare.html"

QWEN_MODEL = "Qwen-Qwen3-8B-Base"

# Default path — update or pass as CLI arg if your local copy is elsewhere.
DPO_CSV_DEFAULT = (
    Path.home() / "Downloads"
    / "spar-ood-propensities-main"
    / "lily" / "propensities" / "src" / "dpo" / "output" / "exports"
    / "online_dpo_qwen3-8b-base_summary_2026-05-19-08-40.csv"
)

# The 9 traits covered by the DPO CSV (these are the plus-poles in FT/SP notation).
DPO_TRAITS: list[str] = [
    "claiming-superintelligence",
    "cooperation",
    "harm-elaboration",
    "harm-refusal",
    "honest-humble",
    "neuroticism",
    "power-seeking",
    "self-preservation",
    "spitefulness",
]

_DPO_PLUS_LABELS: frozenset[str] = frozenset(f"{t}+" for t in DPO_TRAITS)
_DPO_X_LABELS: frozenset[str] = frozenset(DPO_TRAITS)


# ---------------------------------------------------------------------------
# Panel filtering
# ---------------------------------------------------------------------------


def _filter_to_dpo(panel: dict) -> None:
    """Filter panel in-place to DPO traits only.

    Rows: keep the 'base' row (if present) and plus-pole rows for each DPO trait.
    Cols: keep exact-match eval columns (no sub-metric suffixes like ':exploitation_score').
    """
    y_keep = [
        i for i, yl in enumerate(panel["y_labels"])
        if yl == "base" or yl in _DPO_PLUS_LABELS
    ]
    x_keep = [
        i for i, xl in enumerate(panel["x_labels"])
        if xl in _DPO_X_LABELS
    ]

    y_old_new = {o: n for n, o in enumerate(y_keep)}
    x_old_new = {o: n for n, o in enumerate(x_keep)}

    rows_kept = [panel["matrix"][i] for i in y_keep]
    panel["y_labels"]         = [panel["y_labels"][i]        for i in y_keep]
    panel["matrix"]           = [[row[j] for j in x_keep]    for row in rows_kept]
    panel["x_labels"]         = [panel["x_labels"][j]        for j in x_keep]
    panel["reduce_mask"]      = [panel["reduce_mask"][i]      for i in y_keep]
    panel["corr_y_keys"]      = [panel["corr_y_keys"][i]      for i in y_keep]
    panel["corr_x_keys"]      = [panel["corr_x_keys"][j]      for j in x_keep]
    panel["col_eval_mismatch"]= [panel["col_eval_mismatch"][j] for j in x_keep]
    panel["col_evals_seen"]   = [panel["col_evals_seen"][j]   for j in x_keep]
    panel["baseline_alt"]     = None

    # Remap diag / diag_end through new indices.
    panel["diag"] = {
        y_old_new[oy]: x_old_new[ox]
        for oy, ox in (panel.get("diag") or {}).items()
        if oy in y_old_new and ox in x_old_new
    }
    panel["diag_end"] = {
        y_old_new[oy]: x_old_new[ox]
        for oy, ox in (panel.get("diag_end") or {}).items()
        if oy in y_old_new and ox in x_old_new
    }

    # Fix base_idx.
    new_ys = panel["y_labels"]
    panel["base_idx"] = new_ys.index("base") if "base" in new_ys else -1


# ---------------------------------------------------------------------------
# DPO panel builder
# ---------------------------------------------------------------------------


def _build_dpo_panel(csv_path: Path, base_path: Path) -> dict:
    """Build a panel dict directly from the DPO summary CSV.

    Prepends a 'base' row from basemodel_scores so the diff-from-base view works.
    """
    # Load CSV: trained_trait -> eval_trait -> {m, n}
    raw: dict[str, dict[str, dict]] = {}
    with csv_path.open() as f:
        for row in csvmod.DictReader(f):
            tt = row["trained_trait"]
            et = row["eval_trait"]
            raw.setdefault(tt, {})[et] = {
                "m": float(row["mean_score"]),
                "s": None, "mn": None, "mx": None,
                "n": int(row["n"]), "nt": int(row["n"]),
                "nn": 0, "nf": 0, "e": "",
            }

    # Use canonical order from DPO_TRAITS.
    x_labels = list(DPO_TRAITS)

    # Base row from the untuned model scores.
    base_cells_raw = json.loads(base_path.read_text()).get("cells", {}).get("base", {})
    base_row = [bc._cell_metrics(base_cells_raw.get(t)) for t in DPO_TRAITS]

    trait_rows = [
        [raw.get(t, {}).get(et) for et in DPO_TRAITS]
        for t in DPO_TRAITS
    ]

    y_labels = ["base"] + [f"{t}+" for t in DPO_TRAITS]
    matrix = [base_row] + trait_rows

    # Diagonal: trained trait == evaluated trait (shift by 1 for base row).
    x_idx = {xl: i for i, xl in enumerate(x_labels)}
    diag = {yi + 1: x_idx[t] for yi, t in enumerate(DPO_TRAITS)}

    # Cross-panel correspondence keys match the unified format used in FT/SP.
    corr_y_keys = [None] + [f"{t}|plus" for t in DPO_TRAITS]
    corr_x_keys = list(DPO_TRAITS)

    n = len(DPO_TRAITS)
    return {
        "title": "DPO: Qwen3-8B-Base",
        "group": "dpo",
        "tick_group": "dpo",
        "model": QWEN_MODEL,
        "y_labels": y_labels,
        "x_labels": x_labels,
        "matrix": matrix,
        "base_idx": 0,
        "reduce_mask": [True] * (n + 1),
        "diag": diag,
        "diag_end": dict(diag),
        "baseline_alt": None,
        "corr_y_keys": corr_y_keys,
        "corr_x_keys": corr_x_keys,
        "col_eval_mismatch": [False] * n,
        "col_evals_seen": [[] for _ in range(n)],
    }


# ---------------------------------------------------------------------------
# Main panel assembly
# ---------------------------------------------------------------------------


def build_panels(dpo_csv: Path) -> list[dict]:
    """Return [sp_panel, ft_panel, dpo_panel], all filtered to DPO traits."""
    base_path = SCORES_DIR / f"basemodel_scores_{QWEN_MODEL}.json"

    # ---- System-prompted ----
    sp_panel = bc.load_panel(
        SCORES_DIR / f"sysprompts_scores_{QWEN_MODEL}.json",
        transpose=False,
        title="system-prompted: Qwen3-8B-Base",
        group="system-prompted",
        model=QWEN_MODEL,
    )
    if base_path.exists():
        nbc._prepend_base_to_panel(sp_panel, base_path)
    nbc._apply_unified_labels(sp_panel)
    nbc._recompute_diag_unified(sp_panel)
    nbc._compute_diag_spans(sp_panel)
    _filter_to_dpo(sp_panel)

    # ---- Finetuned ----
    ft_panel = bc.load_panel(
        SCORES_DIR / f"finetuned_scores_{QWEN_MODEL}.json",
        transpose=False,
        title="finetuned: Qwen3-8B-Base",
        group="finetuned",
        model=QWEN_MODEL,
    )
    if base_path.exists():
        nbc._prepend_base_to_panel(ft_panel, base_path)
    nbc._apply_unified_labels(ft_panel)
    nbc._recompute_diag_unified(ft_panel)
    nbc._compute_diag_spans(ft_panel)
    _filter_to_dpo(ft_panel)

    # ---- DPO (from CSV) ----
    dpo_panel = _build_dpo_panel(dpo_csv, base_path)

    return [sp_panel, ft_panel, dpo_panel]


# ---------------------------------------------------------------------------
# HTML patch — inject DPO-SP and DPO-FT diff panels into the JS
# ---------------------------------------------------------------------------

_DPO_DIFF_JS = """
// DPO-SP and DPO-FT diff panels (virtual, JS-only)
{
  const dpoPanels = PANELS.filter(p => p.group === 'dpo');
  const spPanels  = PANELS.filter(p => p.group === 'system-prompted');
  const ftPanels  = PANELS.filter(p => p.group === 'finetuned');
  for (const dpoP of dpoPanels) {
    const mShort = TABLE_MODEL_SHORT[dpoP.model] || MODEL_SHORT[dpoP.model] || dpoP.model;
    const spP = spPanels.find(p => p.model === dpoP.model);
    if (spP) {
      const dp = createFtSpDiffPanel(dpoP, spP);
      dp.title = `diff (DPO-SP): ${mShort}`;
      const pv = new PanelView(dp, gridEl, panelViews.length);
      pv.rootEl.style.display = 'none';
      panelViews.push(pv);
      state.visiblePanels.push(false);
    }
    const ftP = ftPanels.find(p => p.model === dpoP.model);
    if (ftP) {
      const dp = createFtSpDiffPanel(dpoP, ftP);
      dp.title = `diff (DPO-FT): ${mShort}`;
      const pv = new PanelView(dp, gridEl, panelViews.length);
      pv.rootEl.style.display = 'none';
      panelViews.push(pv);
      state.visiblePanels.push(false);
    }
  }
}
"""

# Injection point: right after the FT-SP diff block closes, before the Groups panel block.
_FT_SP_BLOCK_END = (
    "  for (const ftP of ftPanels) {\n"
    "    const spP = spPanels.find(p => p.model === ftP.model);\n"
    "    if (!spP) continue;\n"
    "    const diffPanel = createFtSpDiffPanel(ftP, spP);\n"
    "    const pv = new PanelView(diffPanel, gridEl, panelViews.length);\n"
    "    pv.rootEl.style.display = 'none';\n"
    "    panelViews.push(pv);\n"
    "    state.visiblePanels.push(false);\n"
    "  }\n"
    "}\n"
)


def _patch_html_for_dpo_diffs(raw: str) -> str:
    if _FT_SP_BLOCK_END not in raw:
        raise RuntimeError(
            "Could not find FT-SP diff block end in HTML template — "
            "build_compare.py may have changed."
        )
    return raw.replace(_FT_SP_BLOCK_END, _FT_SP_BLOCK_END + _DPO_DIFF_JS, 1)


# ---------------------------------------------------------------------------
# HTML patch — scatter plot panel
# ---------------------------------------------------------------------------

# 1. Inject the panel div after symDiffContainer.
_SCATTER_CONTAINER_ANCHOR = '<div id="symDiffContainer"></div>'
_SCATTER_CONTAINER_HTML = """<div id="symDiffContainer"></div>
<div id="scatterPanel" style="display:none; padding:0 14px 14px;">
  <div class="panel" style="display:inline-block; padding:10px; min-width:460px;">
    <h3 style="display:flex; align-items:baseline; gap:6px;">
      <span class="h3-title">SP vs FT / DPO correlation</span>
      <button class="panel-dl" id="scatterDl"
        title="Download scatter plot as PNG (uses screenshot scale)">&#10515; PNG</button>
    </h3>
    <canvas id="scatterCanvas" style="width:440px; height:440px; display:block;"></canvas>
  </div>
</div>"""

# 2. Append the scatter toggle button to panelToggleBar (end of IIFE).
_SCATTER_BTN_ANCHOR = "})();"
_SCATTER_BTN_JS = """
// Scatter toggle button
{
  const bar = document.getElementById('panelToggleBar');
  const btn = document.createElement('button');
  btn.className = 'plain'; btn.id = 'scatterToggle'; btn.textContent = 'scatter';
  bar.appendChild(btn);
  btn.addEventListener('click', function() {
    _scatterVisible = !_scatterVisible;
    document.getElementById('scatterPanel').style.display = _scatterVisible ? '' : 'none';
    this.classList.toggle('active', _scatterVisible);
    if (_scatterVisible) renderScatterPlot();
  });
  // Resize scatter canvas when cell scale changes
  const _scaleEl = document.getElementById('figCellScale');
  if (_scaleEl) {
    _scaleEl.addEventListener('change', function() {
      const cs = Number(this.value) || 1;
      const sz = Math.round(440 * cs) + 'px';
      const c = document.getElementById('scatterCanvas');
      if (c) { c.style.width = sz; c.style.height = sz; }
      if (_scatterVisible) renderScatterPlot();
    });
  }
}
})();"""

# 3. Core scatter JS — injected before the screenshot event listener.
_SCATTER_JS = r"""
// ── Scatter plot (SP vs FT/DPO) ─────────────────────────────────────────────
let _scatterVisible = false;

function drawScatterOnCtx(ctx, W, H) {
  const ML = 54, MR = 16, MT = 16, MB = 50;
  const PW = W - ML - MR, PH = H - MT - MB;

  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, W, H);

  const spPv  = panelViews.find(pv => pv.p.group === 'system-prompted');
  const ftPv  = panelViews.find(pv => pv.p.group === 'finetuned');
  const dpoPv = panelViews.find(pv => pv.p.group === 'dpo');

  if (!spPv || (!ftPv && !dpoPv)) {
    ctx.fillStyle = '#888'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Need SP + FT or DPO panels', W / 2, H / 2);
    return;
  }

  const getVal = (pv, dYi, dXi) => {
    const v = getDisplayedValues(pv.p, dYi, dXi);
    return (v && v.m != null) ? v.m : null;
  };

  // Build SP lookup: "corr_y_key||corr_x_key" → value
  const spMap = new Map();
  for (let j = 0; j < spPv.activeRows.length; j++) {
    const dYi = spPv.activeRows[j];
    if (dYi === spPv.p.base_idx) continue;
    const yk = spPv.p.corr_y_keys[dYi]; if (!yk) continue;
    for (let k = 0; k < spPv.activeCols.length; k++) {
      const dXi = spPv.activeCols[k];
      const xk = spPv.p.corr_x_keys[dXi]; if (!xk) continue;
      const v = getVal(spPv, dYi, dXi);
      if (v != null) spMap.set(yk + '||' + xk, v);
    }
  }

  const collectDots = (pv) => {
    const dots = [];
    if (!pv) return dots;
    for (let j = 0; j < pv.activeRows.length; j++) {
      const dYi = pv.activeRows[j];
      if (dYi === pv.p.base_idx) continue;
      const yk = pv.p.corr_y_keys[dYi]; if (!yk) continue;
      for (let k = 0; k < pv.activeCols.length; k++) {
        const dXi = pv.activeCols[k];
        const xk = pv.p.corr_x_keys[dXi]; if (!xk) continue;
        const yv = getVal(pv, dYi, dXi);
        const xv = spMap.get(yk + '||' + xk);
        if (xv != null && yv != null)
          dots.push({ x: xv, y: yv,
            row: pv.p.y_labels[dYi].replace(/[+\-]$/, ''),
            col: shortLabel(pv.p.x_labels[dXi]) });
      }
    }
    return dots;
  };

  const blueDots = collectDots(ftPv);
  const redDots  = collectDots(dpoPv);
  if (!blueDots.length && !redDots.length) {
    ctx.fillStyle = '#888'; ctx.font = '12px sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('No matching data', W / 2, H / 2);
    return;
  }

  // Shared axis range (same min/max for x and y so y=x line is diagonal)
  const allVals = [...blueDots, ...redDots].flatMap(d => [d.x, d.y]);
  const rawMin = Math.min(...allVals), rawMax = Math.max(...allVals);
  const pad = Math.max((rawMax - rawMin) * 0.07, 3);
  const axMin = rawMin - pad, axMax = rawMax + pad;

  const toX = v => ML + (v - axMin) / (axMax - axMin) * PW;
  const toY = v => MT + PH * (1 - (v - axMin) / (axMax - axMin));

  // Grid
  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {
    const v = axMin + i * (axMax - axMin) / nTicks;
    ctx.strokeStyle = '#ebebeb'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(toX(v), MT); ctx.lineTo(toX(v), MT + PH); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(ML, toY(v)); ctx.lineTo(ML + PW, toY(v)); ctx.stroke();
  }

  // Border
  ctx.strokeStyle = '#bbb'; ctx.lineWidth = 0.75;
  ctx.strokeRect(ML, MT, PW, PH);

  // y = x reference
  ctx.beginPath();
  ctx.moveTo(toX(axMin), toY(axMin));
  ctx.lineTo(toX(axMax), toY(axMax));
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]);
  ctx.stroke(); ctx.setLineDash([]);

  // Dots (red behind blue so blue diagonals are visible)
  const drawDots = (dots, fill, stroke) => {
    for (const d of dots) {
      ctx.beginPath(); ctx.arc(toX(d.x), toY(d.y), 4.5, 0, Math.PI * 2);
      ctx.fillStyle = fill; ctx.fill();
      ctx.strokeStyle = stroke; ctx.lineWidth = 0.6; ctx.stroke();
    }
  };
  drawDots(redDots,  'rgba(205, 55, 55, 0.70)',  'rgba(160, 30, 30, 0.5)');
  drawDots(blueDots, 'rgba(50, 105, 215, 0.70)', 'rgba(25,  70, 170, 0.5)');

  // Tick labels
  const FONT  = '10px -apple-system,Helvetica,sans-serif';
  const FONTS = '9px -apple-system,Helvetica,sans-serif';
  ctx.font = FONT; ctx.fillStyle = '#444';
  ctx.textAlign = 'center';
  for (let i = 0; i <= nTicks; i++) {
    const v = axMin + i * (axMax - axMin) / nTicks;
    ctx.fillText(v.toFixed(0), toX(v), MT + PH + 13);
  }
  ctx.textAlign = 'right';
  for (let i = 0; i <= nTicks; i++) {
    const v = axMin + i * (axMax - axMin) / nTicks;
    ctx.fillText(v.toFixed(0), ML - 5, toY(v) + 3.5);
  }

  // Axis labels
  const dSuffix = state.diff ? ' (Δ from base)' : '';
  ctx.font = FONTS; ctx.fillStyle = '#555'; ctx.textAlign = 'center';
  ctx.fillText('sys-prompted score' + dSuffix, ML + PW / 2, MT + PH + 34);
  ctx.save();
  ctx.translate(13, MT + PH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('score' + dSuffix, 0, 0);
  ctx.restore();

  // Legend (top-left, inside plot area)
  const legX = ML + 10, legY = MT + 12;
  ctx.font = FONT;
  if (blueDots.length) {
    ctx.fillStyle = 'rgba(50, 105, 215, 0.85)';
    ctx.beginPath(); ctx.arc(legX, legY, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#222'; ctx.textAlign = 'left';
    ctx.fillText('finetuned', legX + 9, legY + 3.5);
  }
  if (redDots.length) {
    const ly = legY + (blueDots.length ? 17 : 0);
    ctx.fillStyle = 'rgba(205, 55, 55, 0.85)';
    ctx.beginPath(); ctx.arc(legX, ly, 4.5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#222'; ctx.textAlign = 'left';
    ctx.fillText('DPO', legX + 9, ly + 3.5);
  }
}

function renderScatterPlot() {
  if (!_scatterVisible) return;
  const canvas = document.getElementById('scatterCanvas');
  if (!canvas) return;
  const W = canvas.clientWidth  || 440;
  const H = canvas.clientHeight || 440;
  const DPR = window.devicePixelRatio || 1;
  canvas.width  = Math.round(W * DPR);
  canvas.height = Math.round(H * DPR);
  const ctx = canvas.getContext('2d');
  ctx.scale(DPR, DPR);
  drawScatterOnCtx(ctx, W, H);
}

document.getElementById('scatterDl').addEventListener('click', function() {
  const _cs = Number(document.getElementById('figCellScale')?.value) || 1;
  const W = Math.round(560 * _cs), H = Math.round(560 * _cs);
  const DPR = state.screenshotScale || 2;
  const off = document.createElement('canvas');
  off.width = W * DPR; off.height = H * DPR;
  const ctx = off.getContext('2d');
  ctx.scale(DPR, DPR);
  drawScatterOnCtx(ctx, W, H);
  const ts = new Date().toISOString().slice(0, 16).replace('T', '_').replace(/:/g, '-');
  const a  = document.createElement('a');
  a.href = off.toDataURL('image/png');
  a.download = `scatter_${ts}.png`;
  a.click();
});

"""

# 4. Trigger re-render on every state change that already triggers renderSymDiff.
_SCATTER_TRIGGER_OLD = "if (state.symDiffMode) renderSymDiff();"
_SCATTER_TRIGGER_NEW = "if (state.symDiffMode) renderSymDiff();\n  if (_scatterVisible) renderScatterPlot();"


def _patch_html_for_scatter(raw: str) -> str:
    # (a) inject panel HTML
    if _SCATTER_CONTAINER_ANCHOR not in raw:
        raise RuntimeError("Could not find symDiffContainer anchor.")
    h = raw.replace(_SCATTER_CONTAINER_ANCHOR, _SCATTER_CONTAINER_HTML, 1)

    # (b) append scatter toggle button inside panelToggleBar (before IIFE close)
    if h.count(_SCATTER_BTN_ANCHOR) < 1:
        raise RuntimeError("Could not find IIFE closing })(); in HTML.")
    # Replace only the LAST occurrence (end of the page script)
    last = h.rfind(_SCATTER_BTN_ANCHOR)
    h = h[:last] + _SCATTER_BTN_JS + h[last + len(_SCATTER_BTN_ANCHOR):]

    # (c) inject scatter JS functions
    if _SAVE_FIG_JS_ANCHOR not in h:
        raise RuntimeError("Could not find screenshot JS anchor.")
    h = h.replace(_SAVE_FIG_JS_ANCHOR, _SCATTER_JS + _SAVE_FIG_JS_ANCHOR, 1)

    # (d) hook scatter re-render into all state-change render calls
    h = h.replace(_SCATTER_TRIGGER_OLD, _SCATTER_TRIGGER_NEW)

    return h


# ---------------------------------------------------------------------------
# HTML patch — "save figure" button (all visible panels → single PNG)
# ---------------------------------------------------------------------------

_SAVE_FIG_BUTTON = (
    '<button class="plain" id="screenshot">screenshot</button>'
)
_SAVE_FIG_BUTTON_REPLACEMENT = (
    '<button class="plain" id="screenshot">screenshot</button>\n'
    '  <button class="plain" id="saveFigure" '
    'title="Export all visible panels as a single publication-style PNG">save figure</button>\n'
    '  <select id="figCellScale" title="Cell size relative to text in saved figure"'
    ' style="background:var(--panel-2);color:var(--ink);border:1px solid var(--grid);'
    'border-radius:4px;padding:2px 4px;">'
    '<option value="0.6">cells 0.6×</option>'
    '<option value="0.8">cells 0.8×</option>'
    '<option value="1" selected>cells 1×</option>'
    '<option value="1.5">cells 1.5×</option>'
    '<option value="2">cells 2×</option>'
    '<option value="3">cells 3×</option>'
    '</select>'
)

_SAVE_FIG_JS_ANCHOR = "document.getElementById('screenshot').addEventListener('click'"

_SAVE_FIG_JS = r"""
function saveFigurePng() {
  const DPR   = state.screenshotScale || 2;
  const vPvs  = panelViews.filter((_, i) => state.visiblePanels[i]);
  if (!vPvs.length) { alert('No visible panels.'); return; }

  // Layout constants — cell size scales with selector; text/gutters stay fixed
  const _cs = Number(document.getElementById('figCellScale')?.value) || 1;
  const CW = Math.round(26 * _cs), CH = Math.round(18 * _cs);
  const Y_GUT = 155, Y_GUT_SM = 4; // left gutter: leftmost panel vs others
  const X_GUT = 115;               // column-label height above cells
  const TTL_H = 26;                // panel title height above column labels
  const GAP   = 16;                // horizontal gap between panels
  const CB_GAP = 22, CB_W = 12, CB_TXT = 46;
  const MRG   = 14;

  const FONT  = '10px -apple-system,Helvetica,sans-serif';
  const FONTB = 'bold 11px -apple-system,Helvetica,sans-serif';
  const FONTS = '9px -apple-system,Helvetica,sans-serif';
  const INK = '#111', DIM = '#555';
  const NULL_C = '#e0e0e0', BORDER = '#aaaaaa';

  const nPan = vPvs.length;
  // All panels share the same column set after filtering, so use the first.
  const nEvals   = vPvs[0].nxEff();
  const nPolesA  = vPvs.map(pv => pv.nyEff());
  const maxPoles = Math.max(...nPolesA);

  // Horizontal layout: only the leftmost panel gets the full y-label gutter.
  const gutters   = vPvs.map((_, pi) => pi === 0 ? Y_GUT : Y_GUT_SM);
  const panelLeft = []; let _x = MRG;
  for (let pi = 0; pi < nPan; pi++) { panelLeft.push(_x); _x += gutters[pi] + nEvals * CW + GAP; }
  const cellsLeft = panelLeft.map((x, pi) => x + gutters[pi]);

  // Vertical layout: title → column labels → cells
  const cellsTop = MRG + TTL_H + X_GUT;
  const totalW   = _x - GAP + CB_GAP + CB_W + CB_TXT + MRG;
  const totalH   = cellsTop + maxPoles * CH + MRG;

  const off = document.createElement('canvas');
  off.width  = Math.ceil(totalW * DPR);
  off.height = Math.ceil(totalH * DPR);
  const ctx  = off.getContext('2d');
  ctx.scale(DPR, DPR);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, totalW, totalH);

  const [vmin, vmax] = rangeForMetric(state.metric);
  const ramp = RAMP_SEQ;
  const col  = v => rampSample(ramp, (v - vmin) / (vmax - vmin));

  // Shared rotated y-axis label on the far left
  ctx.save();
  ctx.translate(MRG + 7, cellsTop + maxPoles * CH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = FONTS; ctx.fillStyle = DIM; ctx.textAlign = 'center';
  ctx.fillText('trained trait', 0, 0);
  ctx.restore();

  vPvs.forEach((pv, pi) => {
    const cL    = cellsLeft[pi];
    const nPi   = nPolesA[pi];
    const diffOn  = state.diff && pv.p.base_idx >= 0;
    const baseRow = diffOn ? effectiveBaseRow(pv.p) : null;

    // --- Panel title (centred above column labels) ---
    const titleX = cL + nEvals * CW / 2;
    // Shorten title to fit: strip model name suffix for diff panels
    const raw_t = pv.p.title;
    const shortT = raw_t
      .replace(/^system-prompted:\s*/,  'sys-prompted')
      .replace(/^finetuned:\s*/,        'finetuned')
      .replace(/^DPO:\s*/,              'DPO')
      .replace(/^diff \(FT-SP\):\s*/,   'Δ(FT−SP)')
      .replace(/^diff \(DPO-SP\):\s*/,  'Δ(DPO−SP)')
      .replace(/^diff \(DPO-FT\):\s*/,  'Δ(DPO−FT)');
    ctx.font = FONTB; ctx.fillStyle = INK; ctx.textAlign = 'center';
    ctx.fillText(shortT, titleX, MRG + 13);

    // "evaluated trait →" in small dim text above rotated col labels (left-aligned per panel)
    ctx.font = FONTS; ctx.fillStyle = DIM; ctx.textAlign = 'left';
    ctx.fillText('evaluated trait →', cL, MRG + TTL_H - 2);

    // --- Column labels (rotated −60°, above cells) ---
    ctx.font = FONT; ctx.fillStyle = DIM;
    for (let k = 0; k < nEvals; k++) {
      const lbl = shortLabel(pv.p.x_labels[pv.activeCols[k]]);
      ctx.save();
      ctx.translate(cL + k * CW + CW / 2, cellsTop - 4);
      ctx.rotate(-Math.PI / 3);
      ctx.textAlign = 'left'; ctx.fillText(lbl, 0, 0);
      ctx.restore();
    }

    // --- Row labels: leftmost panel only ---
    if (pi === 0) {
      ctx.font = FONT; ctx.fillStyle = DIM; ctx.textAlign = 'right';
      for (let j = 0; j < nPi; j++) {
        const yl  = pv.p.y_labels[pv.activeRows[j]];
        const lbl = yl === 'base' ? '(base)' : yl.replace(/[+\-]$/, '');
        ctx.fillText(lbl, cL - 5, cellsTop + j * CH + CH * 0.65);
      }
    }

    // --- Cells ---
    for (let j = 0; j < nPi; j++) {
      const dYi = pv.activeRows[j];
      for (let k = 0; k < nEvals; k++) {
        const dXi  = pv.activeCols[k];
        const cell = effectiveCell(pv.p, dYi, dXi);
        const px   = cL + k * CW, py = cellsTop + j * CH;

        let grey = !cell || cell.m == null;
        if (!grey && state.threshold > 0 && cell.nt > 0 && cell.n / cell.nt <= state.threshold)
          grey = true;

        if (grey) {
          ctx.fillStyle = NULL_C; ctx.fillRect(px, py, CW, CH);
        } else {
          let mv = cell.m, sv = cell.s;
          if (diffOn && baseRow) {
            const b = baseRow[dXi];
            if (b && b.m != null) mv = mv != null ? mv - b.m : null;
          }
          if (state.normalize && sv) { const d = Math.max(sv, 1); if (mv != null) mv /= d; }
          ctx.fillStyle = mv != null ? col(mv) : NULL_C;
          ctx.fillRect(px, py, CW, CH);

          // Diagonal box (spans multi-metric evals)
          const diag = pv.p.diag || {}, diagEnd = pv.p.diag_end || {};
          const dxF = diag[String(dYi)];
          if (dxF !== undefined) {
            const dxL = diagEnd[String(dYi)] !== undefined ? diagEnd[String(dYi)] : dxF;
            if (dXi >= dxF && dXi <= dxL) {
              const lw = 1.5, h2 = lw / 2;
              const isF = dXi === dxF, isL = dXi === dxL;
              ctx.lineWidth = lw; ctx.strokeStyle = '#000'; ctx.beginPath();
              ctx.moveTo(px + (isF ? h2 : 0), py + h2);
              ctx.lineTo(px + CW - (isL ? h2 : 0), py + h2);
              ctx.moveTo(px + (isF ? h2 : 0), py + CH - h2);
              ctx.lineTo(px + CW - (isL ? h2 : 0), py + CH - h2);
              if (isF) { ctx.moveTo(px + h2, py + h2); ctx.lineTo(px + h2, py + CH - h2); }
              if (isL) { ctx.moveTo(px + CW - h2, py + h2); ctx.lineTo(px + CW - h2, py + CH - h2); }
              ctx.stroke();
            }
          }
        }
      }
    }

    // Panel border
    ctx.strokeStyle = BORDER; ctx.lineWidth = 0.5;
    ctx.strokeRect(cL - 0.5, cellsTop - 0.5, nEvals * CW + 1, nPi * CH + 1);
  });

  // --- Colorbar ---
  const lastCL = cellsLeft[nPan - 1];
  const cbX = lastCL + nEvals * CW + CB_GAP;
  const cbY = cellsTop, cbH = maxPoles * CH;
  const grad = ctx.createLinearGradient(0, cbY + cbH, 0, cbY);
  ramp.forEach(([t, [r, g, b]]) => grad.addColorStop(t, `rgb(${r},${g},${b})`));
  ctx.fillStyle = grad; ctx.fillRect(cbX, cbY, CB_W, cbH);
  ctx.strokeStyle = BORDER; ctx.lineWidth = 0.5; ctx.strokeRect(cbX, cbY, CB_W, cbH);
  [0, 0.25, 0.5, 0.75, 1].forEach(t => {
    const v  = vmin + t * (vmax - vmin);
    const ty = cbY + (1 - t) * cbH;
    ctx.fillStyle = BORDER; ctx.fillRect(cbX + CB_W, ty - 0.5, 4, 1);
    ctx.font = FONT; ctx.fillStyle = DIM; ctx.textAlign = 'left';
    ctx.fillText(v.toFixed(0), cbX + CB_W + 6, ty + 3.5);
  });
  ctx.save();
  ctx.translate(cbX - 5, cbY + cbH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.font = FONTS; ctx.fillStyle = DIM; ctx.textAlign = 'center';
  ctx.fillText(state.diff ? 'Δ score' : 'score', 0, 0);
  ctx.restore();

  // Download
  const ts = new Date().toISOString().slice(0, 16).replace('T', '_').replace(/:/g, '-');
  const a  = document.createElement('a');
  a.href = off.toDataURL('image/png');
  a.download = `figure_${ts}.png`;
  a.click();
}

document.getElementById('saveFigure').addEventListener('click', saveFigurePng);
"""


def _patch_html_for_save_figure(raw: str) -> str:
    if _SAVE_FIG_BUTTON not in raw:
        raise RuntimeError("Could not find screenshot button in HTML template.")
    if _SAVE_FIG_JS_ANCHOR not in raw:
        raise RuntimeError("Could not find screenshot JS anchor in HTML template.")
    h = raw.replace(_SAVE_FIG_BUTTON, _SAVE_FIG_BUTTON_REPLACEMENT, 1)
    h = h.replace(_SAVE_FIG_JS_ANCHOR, _SAVE_FIG_JS + _SAVE_FIG_JS_ANCHOR, 1)
    return h


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def main() -> None:
    dpo_csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DPO_CSV_DEFAULT
    if not dpo_csv.exists():
        raise SystemExit(
            f"DPO CSV not found: {dpo_csv}\n"
            "Pass the path as an argument: python new_qwen_html.py path/to/dpo.csv"
        )

    panels = build_panels(dpo_csv)

    ranges = bc.global_ranges(panels)
    dranges = bc.diff_range(panels, use_median=False)
    # No EO panel so dranges_med == dranges.
    dranges_med = bc.diff_range(panels, use_median=True)

    # evals_master and poles_master drive the filter UI controls.
    ref_panel = panels[0]  # SP panel — all three share the same x_labels
    evals_master = ref_panel["x_labels"]
    poles_master = [
        {"name": yl, "key": yk, "in_reduce": True}
        for yl, yk in zip(ref_panel["y_labels"], ref_panel["corr_y_keys"])
        if yl != "base"
    ]

    model_short = {QWEN_MODEL: bc.MODEL_SHORT.get(QWEN_MODEL, QWEN_MODEL)}
    table_model_short = {QWEN_MODEL: bc.TABLE_MODEL_SHORT.get(QWEN_MODEL, "Qwen")}

    html = (
        _patch_html_for_save_figure(_patch_html_for_scatter(_patch_html_for_dpo_diffs(bc.HTML)))
        .replace("__PANELS__", json.dumps(panels))
        .replace("__RANGES__", json.dumps(ranges))
        .replace("__DRANGES__", json.dumps(dranges))
        .replace("__DRANGES_MED__", json.dumps(dranges_med))
        .replace("__EVALS__", json.dumps(evals_master))
        .replace("__POLES__", json.dumps(poles_master))
        .replace("__MODEL_SHORT__", json.dumps(model_short))
        .replace("__TABLE_MODEL_SHORT__", json.dumps(table_model_short))
        .replace("__INITIAL_ROW_ORDER__", json.dumps(None))
        .replace("__INITIAL_ROW_SEPARATORS__", json.dumps([]))
        .replace("__INITIAL_COL_ORDER__", json.dumps(None))
        .replace("__INITIAL_COL_SEPARATORS__", json.dumps([]))
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
