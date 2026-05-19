"""Build a self-contained HTML viewer comparing the three models' normalization
matrices side by side.

Reads the committed transfer matrices (θ / logit-z, plus / minus pole) for
llama_pooled, qwen_pooled, nemotron and emits one static HTML file with
embedded data — no server, no external deps.

    python3 shared/normalization/build_viewer.py

Output: shared/normalization/results/compare_models.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

MODELS = [
    ("llama_pooled", "Llama 3.1-8B", "seed-pooled (4 seeds)"),
    ("qwen_pooled", "Qwen3-8B-Base", "seed-pooled (3 seeds)"),
    ("nemotron", "Nemotron-3 120B", "single seed"),
]
MATRICES = [
    ("theta_plus", "θ · plus pole", "theta"),
    ("theta_minus", "θ · minus pole", "theta"),
    ("logitz_plus", "logit-z · plus pole", "logitz"),
    ("logitz_minus", "logit-z · minus pole", "logitz"),
]


def _load_one(tag: str, mt: str) -> dict | None:
    base = RESULTS / tag / f"transfer_matrix_{mt}"
    if not base.with_suffix(".npy").exists():
        return None
    m = np.load(base.with_suffix(".npy"))
    se = np.load(base.parent / f"{base.name}_se.npy")
    labels = json.loads((base.parent / f"{base.name}.labels.json").read_text())
    return {
        "sources": labels["sources"],
        "targets": labels["targets"],
        "values": [[None if not np.isfinite(v) else round(float(v), 5)
                    for v in row] for row in m],
        "se": [[None if not np.isfinite(v) else round(float(v), 5)
                for v in row] for row in se],
    }


def collect() -> dict:
    data: dict = {}
    for tag, _, _ in MODELS:
        data[tag] = {}
        for mt, _, _ in MATRICES:
            d = _load_one(tag, mt)
            if d is not None:
                data[tag][mt] = d
    summaries = {}
    neutral = {}
    for tag, _, _ in MODELS:
        sp = RESULTS / tag / "run_summary.json"
        if sp.exists():
            summaries[tag] = json.loads(sp.read_text())
        npq = RESULTS / tag / "theta_neutral_per_eval.parquet"
        if npq.exists():
            import pandas as pd
            tn = pd.read_parquet(npq)
            neutral[tag] = {
                str(r["eval"]): round(float(r["theta_neutral"]), 5)
                for _, r in tn.iterrows()
                if np.isfinite(r["theta_neutral"])
            }
        else:
            neutral[tag] = {}
    return {"data": data, "summaries": summaries, "neutral": neutral}


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>normalization · model comparison</title>
<style>
  :root {
    --bg:#0f1115; --panel:#161a22; --panel-2:#1d2230;
    --ink:#e6e9ef; --muted:#8b94a7; --line:#2a3142; --accent:#5b8def;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
    font:13px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:18px 22px; border-bottom:1px solid var(--line);
    background:var(--panel); }
  h1 { margin:0 0 4px; font-size:18px; font-weight:650; }
  .sub { color:var(--muted); font-size:12px; }
  .controls { display:flex; gap:18px; align-items:center; flex-wrap:wrap;
    padding:14px 22px; background:var(--panel-2);
    border-bottom:1px solid var(--line); position:sticky; top:0; z-index:5; }
  .controls label { color:var(--muted); margin-right:6px; }
  select, button { background:var(--panel); color:var(--ink);
    border:1px solid var(--line); border-radius:6px; padding:6px 10px;
    font:inherit; cursor:pointer; }
  button.toggle.on { border-color:var(--accent); color:#fff;
    background:#22304a; }
  .legend { display:flex; align-items:center; gap:8px; margin-left:auto;
    color:var(--muted); }
  .bar { width:160px; height:12px; border-radius:3px;
    border:1px solid var(--line); }
  .grids { display:flex; gap:22px; padding:22px; align-items:flex-start;
    overflow:auto; }
  .model { background:var(--panel); border:1px solid var(--line);
    border-radius:10px; padding:14px; }
  .model h2 { margin:0 0 2px; font-size:14px; }
  .model .meta { color:var(--muted); font-size:11px; margin-bottom:10px; }
  table { border-collapse:collapse; }
  td,th { font-size:10px; }
  th.col { writing-mode:vertical-rl; transform:rotate(180deg);
    color:var(--muted); padding:4px 2px; white-space:nowrap;
    max-height:120px; font-weight:500; }
  th.row { text-align:right; color:var(--muted); padding:0 8px 0 0;
    white-space:nowrap; font-weight:500; }
  td.cell { width:15px; height:15px; border:1px solid #0c0e13;
    cursor:crosshair; }
  td.cell.diag { outline:1.5px solid #fff; outline-offset:-2px; }
  td.na { background:#20232b !important; }
  .tip { position:fixed; pointer-events:none; background:#000d;
    border:1px solid var(--line); border-radius:6px; padding:7px 9px;
    font-size:11px; max-width:260px; display:none; z-index:20; }
  .tip b { color:var(--accent); }
  .note { padding:0 22px 26px; color:var(--muted); font-size:11px;
    max-width:960px; }
  code { background:var(--panel-2); padding:1px 5px; border-radius:4px; }
</style>
</head>
<body>
<header>
  <h1>Normalization transfer matrices — model comparison</h1>
  <div class="sub">Rows = source axis (fine-tuned organism), columns = target
  eval. Diagonal (white outline) is the organism's own axis.</div>
</header>
<div class="controls">
  <span><label>matrix</label>
    <select id="mt">
      <option value="theta_plus">θ · plus pole</option>
      <option value="theta_minus">θ · minus pole</option>
      <option value="thetadiff_plus">θ − base · plus pole</option>
      <option value="thetadiff_minus">θ − base · minus pole</option>
      <option value="logitz_plus">logit-z · plus pole</option>
      <option value="logitz_minus">logit-z · minus pole</option>
    </select></span>
  <button class="toggle" id="seBtn">show SE</button>
  <button class="toggle" id="alignBtn">align axes (union)</button>
  <div class="legend">
    <span id="lo">–</span>
    <div class="bar" id="bar"></div>
    <span id="hi">+</span>
  </div>
</div>
<div class="grids" id="grids"></div>
<div class="note">
  <b>Data source.</b> Seed-pooled LLM-judge scores from the
  <code>cross-elicit</code> pipeline
  (<code>johannes/cross-elicit/results/seeds/scores_&lt;model&gt;_seed-*.json</code>),
  averaged over the available judge-score seeds per base model — Llama
  3.1-8B (4 seeds), Qwen3-8B-Base (3 seeds), Nemotron-3 120B (1 seed). θ /
  logit-z are computed on the seed-averaged per-prompt scores against the
  shared diagonal anchors in <code>configs/anchors.yaml</code>.
  <br><br>
  <b>Elicitation method.</b> Each row (<i>source</i> axis) is a separate
  <b>SFT model organism</b>: the base model LoRA-fine-tuned (rank 32, lr
  2e-5, 10 epochs, ×5 reps) on that trait's pole training set
  (<code>&lt;axis&gt;_eval_train.yaml</code>, e.g. <code>agreeableness-plus</code> /
  <code>agreeableness-minus</code>). Columns are the propensity eval battery
  measured <b>zero-shot</b> on each organism. This is the <b>fine-tuning</b>
  elicitation rung only — no few-shot, system-prompt, or RL elicitation is
  shown here.
  <br><br>
  <b>θ</b> is the raw pole position (0 = minus pole, 1 = plus pole) — a
  sequential ramp, no blue. <b>θ − base</b> subtracts the base model's
  neutral θ on that target eval, so it shows <i>spillover</i>: blue =
  negative spillover (organism pushes the eval below base), warm =
  positive. <b>logit-z</b> is already a z-score around the base mean
  (0 = no effect), same blue/warm convention.
  θ is defined only on dual-pole target axes; plus-only / degenerate axes
  are excluded as θ targets but kept for logit-z. <b>Nemotron</b> is
  single-seed and drops <code>agreeableness</code>,
  <code>neuroticism</code>, <code>resource-acquisition</code> as θ targets.
  Empty grey cells = axis absent for that model (θ − base also greys cells
  with no base-neutral for the target). Hover any cell for value ± SE.
</div>
<div class="tip" id="tip"></div>
<script>
const PAYLOAD = __PAYLOAD__;
const MODELS = __MODELS__;
const DATA = PAYLOAD.data, SUM = PAYLOAD.summaries, NEUTRAL = PAYLOAD.neutral;
let showSE = false, align = false;

function lerp(a,b,t){return a+(b-a)*t;}

// Raw θ ∈ [0,1] is a pole position (0 = minus pole, 1 = plus pole), NOT a
// signed spillover — so it gets a plain sequential ramp with no blue.
function rawColor(v){
  if(v===null||v===undefined) return null;
  let u=Math.max(0,Math.min(1,v));
  return `rgb(${lerp(40,236,u)|0},${lerp(46,196,u)|0},${lerp(58,58,u)|0})`;
}
// Signed spillover, centered on no-effect = 0:
//   θ − base: spill = θ_cell − θ_neutral(target)   (base model's θ on that eval)
//   logit-z:  spill = z      (already a z-score around the base mean)
// Blue is reserved for negative spillover (spill < 0); warm for positive.
function divColor(spill, kind){
  if(spill===null||spill===undefined) return null;
  const span = kind==="logitz" ? 3.0 : 0.5;
  let t=(spill/span+1)/2;                // 0 → blue end, 0.5 → neutral, 1 → red
  t=Math.max(0,Math.min(1,t));
  let r,g,b;
  if(t<0.5){ const u=t/0.5; r=lerp(58,46,u); g=lerp(120,52,u); b=lerp(220,64,u); }
  else { const u=(t-0.5)/0.5; r=lerp(46,224,u); g=lerp(52,72,u); b=lerp(64,72,u); }
  return `rgb(${r|0},${g|0},${b|0})`;
}
// SE is a non-negative magnitude — never blue. Neutral → amber ramp.
function seColor(e, mode){
  if(e===null||e===undefined) return null;
  const span = mode==="z" ? 1.5 : 0.25;
  let u=Math.max(0,Math.min(1,e/span));
  return `rgb(${lerp(46,232,u)|0},${lerp(52,150,u)|0},${lerp(64,40,u)|0})`;
}
function gradientCSS(mode){
  const stops=[];
  for(let i=0;i<=10;i++){
    const t=i/10;
    if(mode==="raw") stops.push(rawColor(t)+" "+(t*100)+"%");
    else { const sp=mode==="z"?3.0:0.5;
      stops.push(divColor((t*2-1)*sp,mode==="z"?"logitz":"diff")+" "+(t*100)+"%"); }
  }
  return "linear-gradient(90deg,"+stops.join(",")+")";
}

// view → underlying data key + display mode.
//   raw  : raw θ pole position (sequential, no blue)
//   diff : θ − base-model neutral θ  (diverging, blue = negative spillover)
//   z    : logit-z score            (diverging, blue = negative spillover)
const VIEW = {
  theta_plus:      ["theta_plus","raw"],
  theta_minus:     ["theta_minus","raw"],
  thetadiff_plus:  ["theta_plus","diff"],
  thetadiff_minus: ["theta_minus","diff"],
  logitz_plus:     ["logitz_plus","z"],
  logitz_minus:    ["logitz_minus","z"],
};

function render(){
  const view = document.getElementById("mt").value;
  const [mt, mode] = VIEW[view];
  document.getElementById("bar").style.background = gradientCSS(mode);
  const lo=document.getElementById("lo"), hi=document.getElementById("hi");
  if(mode==="raw"){ lo.textContent="0 (− pole)"; hi.textContent="1 (+ pole)"; }
  else if(mode==="z"){ lo.textContent="−3 z"; hi.textContent="+3 z"; }
  else { lo.textContent="−0.5 (suppress)"; hi.textContent="+0.5 (amplify)"; }

  let unionSrc=[], unionTgt=[];
  if(align){
    const S=new Set(), T=new Set();
    for(const [tag] of MODELS){ const d=DATA[tag][mt]; if(!d)continue;
      d.sources.forEach(x=>S.add(x)); d.targets.forEach(x=>T.add(x)); }
    unionSrc=[...S].sort(); unionTgt=[...T].sort();
  }

  const grids=document.getElementById("grids"); grids.innerHTML="";
  for(const [tag,name,meta] of MODELS){
    const d=DATA[tag][mt];
    const wrap=document.createElement("div"); wrap.className="model";
    const s=SUM[tag]||{};
    wrap.innerHTML=`<h2>${name}</h2><div class="meta">${meta} · `+
      `${(s.n_rows_loaded||0).toLocaleString()} rows · `+
      `${(s.flagged_evals||[]).length} flagged</div>`;
    if(!d){ wrap.innerHTML+="<div class='meta'>no matrix</div>";
      grids.appendChild(wrap); continue; }
    const srcs = align?unionSrc:d.sources;
    const tgts = align?unionTgt:d.targets;
    const si=Object.fromEntries(d.sources.map((x,i)=>[x,i]));
    const ti=Object.fromEntries(d.targets.map((x,i)=>[x,i]));
    let html="<table><tr><th></th>";
    for(const t of tgts) html+=`<th class="col">${t}</th>`;
    html+="</tr>";
    for(const sName of srcs){
      html+=`<tr><th class="row">${sName}</th>`;
      for(const tName of tgts){
        const i=si[sName], j=ti[tName];
        if(i===undefined||j===undefined){ html+=`<td class="cell na"></td>`; continue; }
        const v=d.values[i][j], e=d.se[i][j];
        if(v===null){ html+=`<td class="cell na"></td>`; continue; }
        const nb = (NEUTRAL[tag]||{})[tName];
        // diff view needs the base-model neutral θ for this target eval.
        if(mode==="diff" && (nb===undefined||nb===null)){
          html+=`<td class="cell na"></td>`; continue; }
        const spill = mode==="diff" ? v-nb : v;   // signed; z is already signed
        let c;
        if(showSE) c=seColor(e,mode);
        else if(mode==="raw") c=rawColor(v);
        else c=divColor(spill, mode==="z"?"logitz":"diff");
        const diag = sName===tName ? " diag" : "";
        html+=`<td class="cell${diag}" style="background:${c}" `+
          `data-v="${v}" data-e="${e}" data-s="${sName}" data-t="${tName}" `+
          `data-m="${name}" data-mode="${mode}" `+
          `data-nb="${nb===undefined?'':nb}" `+
          `data-sp="${mode==="raw"?'':spill}"></td>`;
      }
      html+="</tr>";
    }
    html+="</table>";
    wrap.insertAdjacentHTML("beforeend",html);
    grids.appendChild(wrap);
  }
}

const tip=document.getElementById("tip");
document.getElementById("grids").addEventListener("mousemove",e=>{
  const c=e.target.closest("td.cell");
  if(!c||c.classList.contains("na")){ tip.style.display="none"; return; }
  tip.style.display="block";
  tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+14)+"px";
  const md=c.dataset.mode;
  let body;
  if(md==="diff"){
    body=`θ ${(+c.dataset.v).toFixed(4)}`+
      `<br>base θ ${(+c.dataset.nb).toFixed(4)}`+
      `<br>spillover Δ <b>${(+c.dataset.sp).toFixed(4)}</b>`;
  } else if(md==="z"){
    body=`logit-z <b>${(+c.dataset.v).toFixed(4)}</b>`;
  } else {
    body=`θ <b>${(+c.dataset.v).toFixed(4)}</b>`+
      (c.dataset.nb!==""?`<br>base θ ${(+c.dataset.nb).toFixed(4)}`:"");
  }
  tip.innerHTML=`<b>${c.dataset.m}</b><br>${c.dataset.s} → ${c.dataset.t}`+
    `<br>${body}<br>SE ${(+c.dataset.e).toFixed(4)}`;
});
document.getElementById("grids").addEventListener("mouseleave",
  ()=>tip.style.display="none");

document.getElementById("mt").addEventListener("change",render);
document.getElementById("seBtn").addEventListener("click",e=>{
  showSE=!showSE; e.target.classList.toggle("on",showSE);
  e.target.textContent=showSE?"showing SE":"show SE"; render();
});
document.getElementById("alignBtn").addEventListener("click",e=>{
  align=!align; e.target.classList.toggle("on",align); render();
});
render();
</script>
</body>
</html>
"""


def main() -> None:
    payload = collect()
    models_js = json.dumps([[t, n, m] for t, n, m in MODELS])
    html = (HTML
            .replace("__PAYLOAD__", json.dumps(payload))
            .replace("__MODELS__", models_js))
    out = RESULTS / "compare_models.html"
    out.write_text(html)
    n = sum(len(v) for v in payload["data"].values())
    print(f"wrote {out}  ({n} matrices across {len(payload['data'])} models, "
          f"{len(html)//1024} KB)")


if __name__ == "__main__":
    main()
