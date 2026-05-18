"""Migrate eval_results from v1 → v2 for one or more propensities (gpt-5.4-mini judge).

Three modes, auto-detected per propensity by inspecting the v1 and v2 YAMLs:

  * COPY        — items and judge prompts both identical (e.g. reward-hacking).
                  Just rename existing v1 outputs into v2-named dirs. No
                  Tinker / OpenAI calls.
  * REJUDGE     — items identical, judge prompt changed (e.g. resource-acquisition).
                  Reuse (question, answer) pairs from v1 rows.jsonl, call the
                  v2 judge prompt on each, write v2-named outputs.
  * REGENERATE  — items diverge (different paraphrases or new items, e.g.
                  agreeableness, neuroticism). Run run_eval.py from scratch
                  with the v2 YAML for each (model × base + sysprompt-flavor).

Scope (same for all modes):
  * Models      : Qwen-Qwen3-8B-Base, meta-llama-Llama-3.1-8B-Instruct
                  (Nemotron skipped — already v2).
  * Phases      :
      0. regen           — only for COPY / REJUDGE modes: regen any latest-pick
                           v1 source dir whose rows.jsonl is under --min-rows
                           via run_eval.py with the v1 YAML.
      1. base            — eval_results/finetuning/{model}__base__*
      2. eval-orth       — eval_results/test_evals/<largest run>; only rejudged
                           when the propensity's judge prompt changed v1 → v2.
      3. system-prompted — eval_results/sys_prompts/{model}__base__sysprompt-*
                           (baseline-empty is populated by copying the phase-1
                           v2 base output, not by an independent rejudge).

CLI:
    python rejudge_eval.py --prop resource-acquisition          # one prop
    python rejudge_eval.py --prop reward-hacking,agreeableness,neuroticism
    python rejudge_eval.py --prop neuroticism --dry-run         # plan only
    python rejudge_eval.py --prop reward-hacking --phase base   # one phase
    python rejudge_eval.py --prop agreeableness --limit 5       # smoke test
    python rejudge_eval.py --prop neuroticism --mode regenerate # override auto-detect
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai import APIError, APITimeoutError, RateLimitError

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS_ELICIT_ROOT = SCRIPT_DIR.parent
JOHANNES_ROOT = CROSS_ELICIT_ROOT.parent
EVAL_RESULTS = CROSS_ELICIT_ROOT / "eval_results"
FT_DIR = EVAL_RESULTS / "finetuning"
SP_DIR = EVAL_RESULTS / "sys_prompts"
TE_DIR = EVAL_RESULTS / "test_evals"
SYS_PROMPTS_ROOT = CROSS_ELICIT_ROOT / "evals"           # evals/<prop>/system_prompts/<pole>.txt

load_dotenv(JOHANNES_ROOT / ".env")

JUDGE_MODEL = "gpt-5.4-mini"

MODELS = [
    "Qwen-Qwen3-8B-Base",
    "meta-llama-Llama-3.1-8B-Instruct",
]

# Dir-name encoding (org-model with dashes) → Tinker --checkpoint (org/model with slash).
MODEL_TO_HF = {
    "Qwen-Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "meta-llama-Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
}

# Used to remap paths recorded inside summary.json from the server (Linux)
# tree onto this machine before passing them to run_eval.py.
PATH_REMAP_FROM = "/home/ubuntu/spar-ood-propensities"
PATH_REMAP_TO = str(JOHANNES_ROOT.parent)

CONCURRENCY = 8
MAX_RETRIES = 5


# ─── Per-propensity configuration + mode detection ───────────────────────────

@dataclass
class PropConfig:
    prop: str                # 'resource-acquisition', 'agreeableness', ...
    eval_v1: str             # f'{prop}_eval'           (used in dir-name prefixes)
    eval_v2: str             # f'{prop}_eval_v2'
    metric_key: str          # f'{prop.replace("-","_")}_score'
    v1_yaml: Path
    v2_yaml: Path
    judge_differs: bool      # True if the v2 judge_prompts differ from v1's
    mode: str                # 'copy' | 'rejudge' | 'regenerate'


def _yaml_items(p: Path) -> list[dict]:
    items = yaml.safe_load(p.read_text()) or []
    return items if isinstance(items, list) else []


def detect_mode(prop: str, min_rows: int, override: str | None = None) -> PropConfig:
    """Build a PropConfig for `prop` with mode auto-detected (unless override).

    Mode rule:
      * shared = items present in BOTH v1.test and v2.test with identical paraphrases.
      * if shared < min_rows                  → REGENERATE (v1 can't supply n≥min_rows).
      * else if judge_prompts identical       → COPY (existing data fully serves v2).
      * else                                  → REJUDGE (reuse Q+A, swap judge prompt).
    """
    v1_path = CROSS_ELICIT_ROOT / "evals" / prop / f"{prop}_eval.yaml"
    v2_path = CROSS_ELICIT_ROOT / "evals" / prop / f"{prop}_eval_v2.yaml"
    if not v1_path.exists() or not v2_path.exists():
        raise SystemExit(f"missing yaml(s) for prop={prop}: v1={v1_path.exists()} v2={v2_path.exists()}")
    metric_key = prop.replace("-", "_") + "_score"
    eval_v1 = f"{prop}_eval"
    eval_v2 = f"{prop}_eval_v2"

    v1 = _yaml_items(v1_path)
    v2 = _yaml_items(v2_path)
    v1_test = {it["id"]: it for it in v1 if (it.get("meta") or {}).get("split") == "test"}
    v2_test = {it["id"]: it for it in v2 if (it.get("meta") or {}).get("split") == "test"}
    shared = sum(
        1 for i in (set(v1_test) & set(v2_test))
        if v1_test[i].get("paraphrases") == v2_test[i].get("paraphrases")
    )

    jp1 = (v1[0].get("judge_prompts") or {}) if v1 else {}
    jp2 = (v2[0].get("judge_prompts") or {}) if v2 else {}
    judge_differs = (jp1.get(metric_key) != jp2.get(metric_key))

    if override:
        mode = override
    elif shared < min_rows:
        mode = "regenerate"
    elif not judge_differs:
        mode = "copy"
    else:
        mode = "rejudge"

    return PropConfig(
        prop=prop, eval_v1=eval_v1, eval_v2=eval_v2,
        metric_key=metric_key, v1_yaml=v1_path, v2_yaml=v2_path,
        judge_differs=judge_differs, mode=mode,
    )

# ─── Judge plumbing (mirrors run_eval.py's helpers, kept self-contained) ──────

def fill_judge_template(template: str, question: str, answer: str) -> str:
    return template.replace("{question}", question).replace("{answer}", answer)


_SCORE_RE = re.compile(r'"?score"?\s*[:=]\s*("?(?:null|None|\d+(?:\.\d+)?)"?)', re.IGNORECASE)


def parse_judge_text(text: str) -> tuple[str, int | float | None]:
    """Return (bucket, score) where bucket in {numeric, null, fail}.

    The judge is asked to return a JSON-shaped block with a `score` field that is
    either null or a number, but in practice gpt-5.4-mini often returns just
    `null` or a bare number with no `score:` prefix. Handle both shapes.
    """
    if not text:
        return "fail", None
    stripped = text.strip().strip('"').lower()
    if stripped in ("null", "none"):
        return "null", None
    # Bare number (the whole response is e.g. "62" or "62.5")
    try:
        v = float(stripped)
        if v.is_integer():
            v = int(v)
        return "numeric", v
    except ValueError:
        pass
    m = _SCORE_RE.search(text)
    if not m:
        # try last bare integer 0-100 as a fallback
        m2 = list(re.finditer(r'\b(\d{1,3})\b', text))
        if m2:
            try:
                v = int(m2[-1].group(1))
                if 0 <= v <= 100:
                    return "numeric", v
            except ValueError:
                pass
        return "fail", None
    raw = m.group(1).strip().strip('"').lower()
    if raw in ("null", "none"):
        return "null", None
    try:
        v = float(raw)
        if v.is_integer():
            v = int(v)
        return "numeric", v
    except ValueError:
        return "fail", None


async def call_judge(client: AsyncOpenAI, prompt: str, sem: asyncio.Semaphore) -> str:
    """Single judge call with retries + exponential backoff under a semaphore."""
    delay = 1.0
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = resp.choices[0].message.content or ""
                return content
            except (RateLimitError, APITimeoutError, APIError) as e:
                last_err = e
        # outside semaphore: sleep then retry
        await asyncio.sleep(delay)
        delay = min(delay * 2, 30.0)
    raise RuntimeError(f"judge call failed after {MAX_RETRIES} attempts: {last_err!r}")


# ─── Helpers for dir naming, timestamps, etc. ─────────────────────────────────

def new_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")


_TS_TAIL_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}(?:-\d+)?$")


def dir_timestamp(name: str) -> str:
    """Extract the timestamp tail of a dir name; empty string if not present."""
    m = _TS_TAIL_RE.search(name)
    return m.group(0) if m else ""


def latest_dir(dirs: list[Path]) -> Path | None:
    if not dirs:
        return None
    return max(dirs, key=lambda p: dir_timestamp(p.name))


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_v2_template(cfg: PropConfig) -> str:
    """Return the metric-specific judge prompt template from the v2 yaml."""
    items = _yaml_items(cfg.v2_yaml)
    if not items:
        raise SystemExit(f"v2 yaml is empty: {cfg.v2_yaml}")
    jp = items[0].get("judge_prompts") or {}
    if cfg.metric_key not in jp:
        raise SystemExit(f"{cfg.metric_key} not in judge_prompts of {cfg.v2_yaml}")
    return jp[cfg.metric_key]


def template_sha(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:16]


# ─── Plan ─────────────────────────────────────────────────────────────────────

def plan_base_dirs(cfg: PropConfig) -> list[tuple[Path, str]]:
    """Returns [(source_dir, output_dir_name), ...] for the base phase."""
    out = []
    for m in MODELS:
        cands = sorted(FT_DIR.glob(f"{cfg.eval_v1}__{m}__base__*"))
        chosen = latest_dir(cands)
        if not chosen:
            print(f"  WARNING: no v1 base dir for model={m}")
            continue
        out.append((chosen, f"{cfg.eval_v2}__{m}__base__{new_timestamp()}"))
    return out


def plan_sysprompts_dirs(cfg: PropConfig) -> list[tuple[Path, str]]:
    """For each (model × sysprompt-flavor), pick the latest-timestamp v1 dir.

    Skips `sysprompt-baseline-empty`: those cells are populated by copying
    phase 1's v2 base output (handled by `copy_base_into_sysprompts`).
    """
    by_flavor: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for m in MODELS:
        for p in SP_DIR.glob(f"{cfg.eval_v1}__{m}__base__sysprompt-*"):
            if not p.is_dir():
                continue
            # parts: ['<eval>_eval', model, 'base', 'sysprompt-<flavor>', timestamp]
            parts = p.name.split("__")
            if len(parts) < 5:
                continue
            flavor = parts[3]  # 'sysprompt-<flavor>'
            if flavor == "sysprompt-baseline-empty":
                continue
            by_flavor[(m, flavor)].append(p)
    out = []
    for (m, flavor), dirs in sorted(by_flavor.items()):
        chosen = latest_dir(dirs)
        if not chosen:
            continue
        out.append((chosen, f"{cfg.eval_v2}__{m}__base__{flavor}__{new_timestamp()}"))
    return out


async def copy_base_into_sysprompts(
    *, cfg: PropConfig, template: str | None, client: AsyncOpenAI | None,
    sem: asyncio.Semaphore | None, limit: int | None, dry_run: bool,
) -> list[Path]:
    """For each model, ensure a v2 base finetuning dir exists, then copy it into
    eval_results/sys_prompts/ with a sysprompt-baseline-empty rename.

    Normally phase 1 (base) has already created the v2 base dir, in which case
    this just does the copy. Fallback when no v2 base is on disk yet:
      * REJUDGE mode → rejudge the v1 base inline (needs template+client+sem).
      * COPY mode    → copy the v1 base as-is into a v2-named dir.
      * REGENERATE   → run run_eval.py with v2 yaml inline.

    Each call writes a fresh-timestamped dir; the score aggregator picks the
    latest. Returns the list of written paths.
    """
    written: list[Path] = []
    print("\n--- sysprompt-baseline-empty: copy from v2 base ---")
    for m in MODELS:
        base_v2 = latest_v2_base_dir_for(cfg, m)
        if not base_v2:
            v1_base = latest_base_dir_for(cfg, m)
            if not v1_base:
                print(f"  WARNING: no v1 base dir for {m}; cannot produce baseline-empty cell.")
                continue
            new_v2_name = f"{cfg.eval_v2}__{m}__base__{new_timestamp()}"
            new_v2_dir = FT_DIR / new_v2_name
            print(f"  {m}: no v2 base on disk — falling back ({cfg.mode}) on v1 base")
            print(f"      src: {v1_base}")
            print(f"      out: {new_v2_dir}")
            if not dry_run:
                if cfg.mode == "rejudge":
                    if template is None or client is None or sem is None:
                        raise SystemExit("rejudge fallback requires template/client/sem")
                    await rejudge_rows_jsonl(
                        cfg=cfg, src=v1_base, out_dir=new_v2_dir, template=template,
                        client=client, sem=sem, limit=limit,
                        label=f"fallback-base/{m}",
                    )
                elif cfg.mode == "copy":
                    new_v2_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(v1_base, new_v2_dir)
                    _patch_summary_eval_yaml(new_v2_dir / "summary.json", cfg)
                elif cfg.mode == "regenerate":
                    regen_one(cfg=cfg, model_dir_name=m, flavor=None,
                              dry_run=False, use_v2_yaml=True, max_test_items=limit)
                    base_v2 = latest_v2_base_dir_for(cfg, m)
                    if not base_v2:
                        print(f"  WARNING: regenerate fallback did not produce v2 base for {m}")
                        continue
            base_v2 = base_v2 or new_v2_dir
        new_name = f"{cfg.eval_v2}__{m}__base__sysprompt-baseline-empty__{new_timestamp()}"
        dest = SP_DIR / new_name
        print(f"  {m}: {base_v2.name}  →  sys_prompts/{new_name}")
        if dry_run:
            written.append(dest)
            continue
        SP_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copytree(base_v2, dest)
        sp = dest / "summary.json"
        if sp.exists():
            try:
                d = json.loads(sp.read_text())
                d["system_prompt"] = ""
                d["system_prompt_source"] = "copied-from-base"
                d["copied_from"] = str(base_v2)
                sp.write_text(json.dumps(d, indent=2))
            except Exception as e:
                print(f"    note: could not patch summary.json: {e!r}")
        written.append(dest)
    return written


def _patch_summary_eval_yaml(sp: Path, cfg: PropConfig) -> None:
    """Update summary.json's eval_yaml pointer to the v2 path."""
    if not sp.exists():
        return
    try:
        d = json.loads(sp.read_text())
        d["eval_yaml"] = str(cfg.v2_yaml)
        d.setdefault("copied_from_v1", True)
        sp.write_text(json.dumps(d, indent=2))
    except Exception as e:
        print(f"    note: could not patch summary.json {sp}: {e!r}")


def plan_test_evals_run() -> tuple[Path, str] | None:
    """Pick the test_evals run with the largest n_diagonal; output is <name>_v2."""
    cands = []
    for d in TE_DIR.iterdir():
        if not d.is_dir():
            continue
        mp = d / "matrices.json"
        if not mp.exists():
            continue
        try:
            meta = json.loads(mp.read_text())["_meta"]
            cands.append((meta.get("n_diagonal", 0), d))
        except Exception:
            continue
    if not cands:
        return None
    cands.sort(key=lambda t: t[0], reverse=True)
    chosen = cands[0][1]
    return chosen, f"{chosen.name}_v2"


# ─── Stage 0: scan for under-sized v1 dirs + regenerate via run_eval.py ──────

def rows_count(p: Path) -> int:
    rp = p / "rows.jsonl"
    return sum(1 for _ in open(rp)) if rp.exists() else 0


def remap_path(s: str) -> str:
    """Server-side absolute path → local absolute path."""
    if not s:
        return s
    return s.replace(PATH_REMAP_FROM, PATH_REMAP_TO)


def sysprompt_args_for_flavor(cfg: PropConfig, flavor: str, src_summary: dict | None) -> list[str]:
    """Return the run_eval.py argv tail for this sysprompt flavor.

    Conventions:
      * 'baseline-empty'    → no system-prompt flags (default behaviour).
      * '<prop>--<pole>'    → --system-prompt-file evals/<prop>/system_prompts/<pole>.txt
      * any other token     → --system-prompt-file evals/<cfg.prop>/system_prompts/<token>.txt
                              (e.g. 'hi','lo' for resource-acquisition;
                               'agreeable' for agreeableness, 'neurotic' for
                               neuroticism, 'reward_hacking' for reward-hacking)
    Falls back to the source dir's summary.json `system_prompt_source` (after
    remapping) when the heuristic can't resolve.
    """
    if flavor == "baseline-empty":
        return []
    if "--" in flavor:
        prop, pole = flavor.split("--", 1)
        sp_file = SYS_PROMPTS_ROOT / prop / "system_prompts" / f"{pole}.txt"
    else:
        sp_file = SYS_PROMPTS_ROOT / cfg.prop / "system_prompts" / f"{flavor}.txt"
    if not sp_file.exists() and src_summary:
        sp_recorded = remap_path(src_summary.get("system_prompt_source") or "")
        if sp_recorded and os.path.exists(sp_recorded):
            sp_file = Path(sp_recorded)
    if not sp_file.exists():
        raise FileNotFoundError(
            f"can't resolve sysprompt file for prop={cfg.prop} flavor={flavor!r} "
            f"(tried {sp_file})"
        )
    return ["--system-prompt-file", str(sp_file)]


def parse_run_eval_output_dir(stdout: str) -> Path | None:
    """run_eval.py prints `Output dir: <abs path>` early in its lifetime."""
    m = re.search(r"^Output dir:\s*(.+)$", stdout, re.MULTILINE)
    return Path(m.group(1).strip()) if m else None


def regen_one(*, cfg: PropConfig, model_dir_name: str, flavor: str | None,
              dry_run: bool, use_v2_yaml: bool = False,
              max_test_items: int | None = None) -> Path | None:
    """Regenerate a single base or sysprompt run via run_eval.py.

    flavor=None         → base run (dest: eval_results/finetuning/).
    flavor=<f>          → sysprompt run (dest: eval_results/sys_prompts/, renamed
                          to include `sysprompt-<f>` before the timestamp).
    use_v2_yaml=True    → run with v2 yaml (REGENERATE mode), producing a
                          v2-named dir directly. Otherwise run with v1 yaml.
    max_test_items=N    → pass --max-test-items N to run_eval.py (smoke test).
    Returns the final dest dir (already moved/renamed), or None when dry-run.
    """
    hf = MODEL_TO_HF.get(model_dir_name)
    if not hf:
        raise SystemExit(f"unknown model: {model_dir_name}")
    eval_yaml = cfg.v2_yaml if use_v2_yaml else cfg.v1_yaml
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_eval.py"),
        "--checkpoint", hf,
        "--eval", str(eval_yaml),
    ]
    if max_test_items is not None:
        cmd += ["--max-test-items", str(max_test_items)]
    src_summary = None
    if flavor is not None:
        # Best-effort: load any existing dir's summary.json for fallback.
        existing = sorted(SP_DIR.glob(f"{cfg.eval_v1}__{model_dir_name}__base__sysprompt-{flavor}__*"))
        if existing:
            sp = existing[-1] / "summary.json"
            if sp.exists():
                try:
                    src_summary = json.loads(sp.read_text())
                except Exception:
                    src_summary = None
        cmd += sysprompt_args_for_flavor(cfg, flavor, src_summary)

    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return None

    # Belt-and-suspenders: macOS occasionally re-hides the editable .pth file,
    # which Python 3.14 then silently ignores → tinker_cookbook fails to import.
    # Inject the source dir into PYTHONPATH for the subprocess so the import
    # works regardless of .pth state.
    env = os.environ.copy()
    tc_src = "/Users/jo/Documents/code/tinker-cookbook"
    if Path(tc_src).is_dir():
        prev = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = tc_src + (os.pathsep + prev if prev else "")

    # Stream stdout so the user sees Tinker / judge progress live.
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        cwd=str(SCRIPT_DIR), bufsize=1, env=env,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(f"    | {line}")
        captured.append(line)
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"run_eval.py exited with code {rc}")

    out_dir = parse_run_eval_output_dir("".join(captured))
    if not out_dir:
        raise SystemExit("could not parse Output dir from run_eval.py stdout")

    # Sysprompt runs: move from eval_results/ → eval_results/sys_prompts/ with
    # an inserted sysprompt-<flavor> segment in the dir name (mirrors what
    # sys_prompt_offdiag.py does).
    if flavor is None:
        # Base run already lands in eval_results/. We want it in eval_results/finetuning/.
        new_path = FT_DIR / out_dir.name
        FT_DIR.mkdir(parents=True, exist_ok=True)
        shutil.move(str(out_dir), str(new_path))
        print(f"    moved → {new_path}")
        return new_path
    parts = out_dir.name.split("__")
    if len(parts) < 3:
        raise SystemExit(f"unexpected output dir layout: {out_dir.name}")
    new_name = "__".join(parts[:-1] + [f"sysprompt-{flavor}", parts[-1]])
    SP_DIR.mkdir(parents=True, exist_ok=True)
    new_path = SP_DIR / new_name
    shutil.move(str(out_dir), str(new_path))
    print(f"    moved → {new_path}")
    return new_path


def latest_base_dir_for(cfg: PropConfig, model_dir_name: str) -> Path | None:
    """Latest v1 base finetuning dir for this (prop, model)."""
    return latest_dir(sorted(FT_DIR.glob(f"{cfg.eval_v1}__{model_dir_name}__base__*")))


def latest_v2_base_dir_for(cfg: PropConfig, model_dir_name: str) -> Path | None:
    """Latest v2 base finetuning dir for this (prop, model)."""
    return latest_dir(sorted(FT_DIR.glob(f"{cfg.eval_v2}__{model_dir_name}__base__*")))


def has_v2_sp_dir(cfg: PropConfig, model_dir_name: str, flavor: str) -> Path | None:
    """Latest v2 sysprompt dir for this (prop, model, flavor) — None if none on disk.

    `flavor` is the bare flavor token (e.g. 'agreeable', 'caring-about-aesthetics--caring').
    """
    dirs = sorted(SP_DIR.glob(
        f"{cfg.eval_v2}__{model_dir_name}__base__sysprompt-{flavor}__*"
    ))
    return latest_dir(dirs)


def evalorth_already_rejudged(cfg: PropConfig) -> bool:
    """True if test_evals/<run>_v2/matrices.json has cfg.metric_key in its rejudged_metrics list."""
    plan = plan_test_evals_run()
    if not plan:
        return False
    _src, out_name = plan
    mp = TE_DIR / out_name / "matrices.json"
    if not mp.exists():
        return False
    try:
        meta = json.loads(mp.read_text()).get("_meta", {})
        return cfg.metric_key in (meta.get("rejudged_metrics") or [])
    except Exception:
        return False


def scan_undersized(cfg: PropConfig, min_rows: int) -> dict:
    """Return a manifest of (phase, model, flavor, src_dir, n_rows) to regenerate.

    Sysprompt-baseline-empty entries are pruned when a sufficient base finetuning
    dir exists for that model — the base run is functionally equivalent (same
    model, no finetune, empty sysprompt), so we'll reuse it later via a copy
    instead of regenerating + rejudging it separately.
    """
    items: list[dict] = []
    skipped_baseline_empty: list[dict] = []

    # Phase 1: base
    for m in MODELS:
        chosen = latest_base_dir_for(cfg, m)
        if not chosen:
            items.append({"phase": "base", "model": m, "flavor": None,
                          "src_dir": None, "n_rows": 0, "missing": True})
            continue
        n = rows_count(chosen)
        if n < min_rows:
            items.append({"phase": "base", "model": m, "flavor": None,
                          "src_dir": str(chosen), "n_rows": n, "missing": False})

    # Phase 3: sysprompts (latest per flavor)
    by_flavor: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for m in MODELS:
        for p in SP_DIR.glob(f"{cfg.eval_v1}__{m}__base__sysprompt-*"):
            if not p.is_dir():
                continue
            parts = p.name.split("__")
            if len(parts) < 5:
                continue
            flavor = parts[3].removeprefix("sysprompt-")
            by_flavor[(m, flavor)].append(p)
    for (m, flavor), dirs in sorted(by_flavor.items()):
        chosen = latest_dir(dirs)
        n = rows_count(chosen) if chosen else 0
        if n < min_rows:
            if flavor == "baseline-empty":
                base_chosen = latest_base_dir_for(cfg, m)
                base_n = rows_count(base_chosen) if base_chosen else 0
                if base_n >= min_rows:
                    skipped_baseline_empty.append({
                        "phase": "sysprompts", "model": m, "flavor": flavor,
                        "src_dir": str(chosen) if chosen else None,
                        "n_rows": n, "base_n_rows": base_n,
                        "base_dir": str(base_chosen),
                        "reason": "will reuse base v2 output (copy after phase 1)",
                    })
                    continue
            items.append({"phase": "sysprompts", "model": m, "flavor": flavor,
                          "src_dir": str(chosen) if chosen else None,
                          "n_rows": n, "missing": chosen is None})
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "prop": cfg.prop,
        "mode": cfg.mode,
        "min_rows": min_rows,
        "n_items": len(items),
        "items": items,
        "skipped_baseline_empty": skipped_baseline_empty,
    }


async def phase_regen(cfg: PropConfig, min_rows: int, dry_run: bool,
                      limit: int | None = None):
    """Stage 0: regenerate any under-sized v1 dirs via run_eval.py with the v1 yaml.

    Only meaningful for COPY / REJUDGE modes (where downstream phases consume
    the v1 dirs). For REGENERATE mode this is a no-op since phase 1+3 already
    re-run everything from scratch with the v2 yaml.
    """
    if cfg.mode == "regenerate":
        print(f"\n  STAGE 0 skipped: mode={cfg.mode} regenerates everything anyway.")
        return
    print("\n" + "=" * 78)
    print(f"STAGE 0 — REGENERATE under-sized v1 dirs (--min-rows={min_rows})  [{cfg.prop}]")
    print("=" * 78)
    manifest = scan_undersized(cfg, min_rows)
    manifest_path = SCRIPT_DIR / f"rejudge_missing_manifest__{cfg.prop}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest written → {manifest_path}")
    if manifest["skipped_baseline_empty"]:
        print(f"  baseline-empty entries skipped (will reuse base v2 after phase 1):")
        for it in manifest["skipped_baseline_empty"]:
            print(f"    {it['model']}/{it['flavor']}  n={it['n_rows']}  "
                  f"(base has n={it['base_n_rows']})")
    items = manifest["items"]
    if not items:
        print("  nothing under threshold — skipping stage 0.")
        return
    print(f"  {len(items)} dir(s) to regenerate:")
    for it in items:
        flv = it["flavor"] or "(base)"
        print(f"    [{it['phase']:<10}] {it['model']}/{flv}  n={it['n_rows']}  src={it['src_dir']}")
    if dry_run:
        print("  --dry-run: skipping run_eval.py invocations.")
        return
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY not set. Expected it in "
            f"{JOHANNES_ROOT / '.env'} (via python-dotenv) or in the env."
        )
    for i, it in enumerate(items, 1):
        print(f"\n--- regen [{i}/{len(items)}]: {it['phase']} / {it['model']} / {it['flavor'] or '(base)'} ---")
        regen_one(cfg=cfg, model_dir_name=it["model"], flavor=it["flavor"],
                  dry_run=False, use_v2_yaml=False, max_test_items=limit)
    print("\n  ── stage 0 complete ──")


# ─── Per-dir rejudge ──────────────────────────────────────────────────────────

async def rejudge_rows_jsonl(
    *,
    cfg: PropConfig,
    src: Path,
    out_dir: Path,
    template: str,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    limit: int | None,
    label: str,
) -> dict:
    """Rejudge a rows.jsonl-style v1 dir; write the new rows.jsonl + summary.json."""
    rows_in_path = src / "rows.jsonl"
    summary_in_path = src / "summary.json"
    if not rows_in_path.exists():
        print(f"  [{label}] SKIP: no rows.jsonl at {src}")
        return {"skipped": True}

    rows_in: list[dict] = []
    with open(rows_in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows_in.append(json.loads(line))
    if limit is not None:
        rows_in = rows_in[:limit]
    n = len(rows_in)
    print(f"  [{label}] {n} rows → {out_dir.name}")

    out_dir.mkdir(parents=True, exist_ok=True)
    rows_out_path = out_dir / "rows.jsonl"

    # Fire all judge calls in parallel under the semaphore.
    async def judge_one(idx: int, r: dict) -> tuple[int, str]:
        prompt = fill_judge_template(template, r["question"], r["answer"])
        text = await call_judge(client, prompt, sem)
        return idx, text

    tasks = [asyncio.create_task(judge_one(i, r)) for i, r in enumerate(rows_in)]
    t0 = time.time()
    done = 0
    progress_every = max(1, n // 20)
    numeric: list[float] = []
    n_null = 0
    n_fail = 0
    new_rows: list[dict | None] = [None] * n

    for fut in asyncio.as_completed(tasks):
        idx, text = await fut
        bucket, score = parse_judge_text(text)
        r = rows_in[idx]
        new_rows[idx] = {**r, "judge_response": text, "bucket": bucket, "score": score}
        done += 1
        if bucket == "numeric":
            numeric.append(float(score))
        elif bucket == "null":
            n_null += 1
        else:
            n_fail += 1
        if done % progress_every == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("inf")
            mean_str = f"{statistics.fmean(numeric):.1f}" if numeric else "-"
            print(
                f"    [{label}] {done}/{n}  "
                f"numeric={len(numeric)} null={n_null} fail={n_fail}  mean={mean_str}  "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s rate={rate:.1f}/s"
            )

    # write rows.jsonl in original order
    with open(rows_out_path, "w") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")

    # build summary.json: prefer to inherit from the source summary, but
    # update eval_yaml and metrics block.
    summary: dict = {}
    if summary_in_path.exists():
        try:
            summary = json.loads(summary_in_path.read_text())
        except Exception:
            summary = {}
    summary["eval_yaml"] = str(cfg.v2_yaml)
    summary["judge_model"] = JUDGE_MODEL
    summary["metrics"] = {
        cfg.metric_key: {
            "n_total": n,
            "n_numeric": len(numeric),
            "n_nulls": n_null,
            "n_fails": n_fail,
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "mean": statistics.fmean(numeric) if numeric else None,
            "std": statistics.stdev(numeric) if len(numeric) >= 2 else None,
        }
    }
    summary.setdefault("rejudged_from", str(src))
    summary["rejudged_at"] = datetime.now().isoformat(timespec="seconds")
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return {
        "n_total": n,
        "n_numeric": len(numeric),
        "n_null": n_null,
        "n_fail": n_fail,
        "mean": statistics.fmean(numeric) if numeric else None,
        "out": str(out_dir),
    }


async def rejudge_test_evals(
    *,
    cfg: PropConfig,
    src: Path,
    out_dir: Path,
    template: str,
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    limit: int | None,
) -> dict:
    """Rejudge <metric_key> rows in a test_evals run; rewrite matrices.json.

    If out_dir already has judgments.jsonl + matrices.json (from a previous
    prop's run in this script invocation), read from there so the updates
    accumulate across propensities. Otherwise read from src.
    """
    # Prefer out_dir if it already exists (incremental mode) — that way running
    # several props in sequence keeps each one's rejudge.
    j_out = out_dir / "judgments.jsonl"
    m_out = out_dir / "matrices.json"
    if j_out.exists() and m_out.exists():
        j_in, m_in = j_out, m_out
        print(f"  [test_evals] incremental: reading existing out_dir files")
    else:
        j_in = src / "judgments.jsonl"
        m_in = src / "matrices.json"
        if not j_in.exists() or not m_in.exists():
            print(f"  [test_evals] SKIP: incomplete src dir {src}")
            return {"skipped": True}
        print(f"  [test_evals] reading from src {j_in}")

    judgments: list[dict] = []
    with open(j_in) as f:
        for line in f:
            line = line.strip()
            if line:
                judgments.append(json.loads(line))
    print(f"  [test_evals] total judgments: {len(judgments)}")

    target_idx = [i for i, r in enumerate(judgments) if r.get("judge_key") == cfg.metric_key]
    print(f"  [test_evals] {cfg.prop} ({cfg.metric_key}) rows to rejudge: {len(target_idx)}")
    if limit is not None:
        target_idx = target_idx[:limit]
        print(f"  [test_evals] --limit {limit}: will rejudge {len(target_idx)} only")

    out_dir.mkdir(parents=True, exist_ok=True)

    new_sha = template_sha(template)

    async def judge_one(idx: int) -> tuple[int, str]:
        r = judgments[idx]
        prompt = fill_judge_template(template, r["question"], r["answer"])
        text = await call_judge(client, prompt, sem)
        return idx, text

    tasks = [asyncio.create_task(judge_one(i)) for i in target_idx]
    t0 = time.time()
    done = 0
    n = len(target_idx)
    progress_every = max(1, n // 20)
    numeric: list[float] = []
    n_null = 0
    n_fail = 0

    for fut in asyncio.as_completed(tasks):
        idx, text = await fut
        bucket, score = parse_judge_text(text)
        r = judgments[idx]
        if bucket == "numeric":
            r["raw_response"] = text
            r["score"] = float(score)
            r["score_status"] = "score"
            numeric.append(float(score))
        elif bucket == "null":
            r["raw_response"] = text
            r["score"] = None
            r["score_status"] = "null"
            n_null += 1
        else:
            r["raw_response"] = text
            r["score"] = None
            r["score_status"] = "fail"
            n_fail += 1
        r["judge_prompt_sha"] = new_sha
        done += 1
        if done % progress_every == 0 or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else float("inf")
            mean_str = f"{statistics.fmean(numeric):.1f}" if numeric else "-"
            print(
                f"    [test_evals] {done}/{n}  "
                f"numeric={len(numeric)} null={n_null} fail={n_fail}  mean={mean_str}  "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s rate={rate:.1f}/s"
            )

    # Write updated judgments.jsonl
    print(f"  [test_evals] writing judgments.jsonl ({len(judgments)} rows) → {out_dir}")
    with open(out_dir / "judgments.jsonl", "w") as f:
        for r in judgments:
            f.write(json.dumps(r) + "\n")

    # Recompute matrices.json's scores/counts for cfg.metric_key row.
    matrices = json.loads(m_in.read_text())
    col_labels: list[str] = matrices["_meta"]["col_labels"]
    col_idx = {c: i for i, c in enumerate(col_labels)}

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    counts_seen: dict[tuple[str, str], int] = defaultdict(int)
    for r in judgments:
        if r.get("judge_key") != cfg.metric_key:
            continue
        cl = r.get("col_label")
        if cl is None:
            continue
        counts_seen[(cfg.metric_key, cl)] += 1
        if r.get("score_status") == "score" and r.get("score") is not None:
            grouped[(cfg.metric_key, cl)].append(float(r["score"]))

    scores = matrices["scores"]
    counts = matrices["counts"]
    new_scores_row = [None] * len(col_labels)
    new_counts_row = [0] * len(col_labels)
    for cl, i in col_idx.items():
        vals = grouped.get((cfg.metric_key, cl), [])
        new_counts_row[i] = counts_seen.get((cfg.metric_key, cl), 0)
        new_scores_row[i] = (statistics.fmean(vals) if vals else None)
    scores[cfg.metric_key] = new_scores_row
    counts[cfg.metric_key] = new_counts_row
    matrices["scores"] = scores
    matrices["counts"] = counts
    matrices["_meta"]["model"] = JUDGE_MODEL
    # Track which metrics have been rejudged in this v2 dir.
    rejudged_metrics = matrices["_meta"].get("rejudged_metrics") or []
    if cfg.metric_key not in rejudged_metrics:
        rejudged_metrics.append(cfg.metric_key)
    matrices["_meta"]["rejudged_metrics"] = rejudged_metrics
    matrices["_meta"]["rejudged_from"] = str(src)
    matrices["_meta"]["rejudged_at"] = datetime.now().isoformat(timespec="seconds")
    with open(out_dir / "matrices.json", "w") as f:
        json.dump(matrices, f, indent=2)

    print(f"  [test_evals] wrote matrices.json → {out_dir}")
    return {
        "n_rejudged": n,
        "n_numeric": len(numeric),
        "n_null": n_null,
        "n_fail": n_fail,
        "mean": statistics.fmean(numeric) if numeric else None,
        "out": str(out_dir),
    }


# ─── Phase drivers ────────────────────────────────────────────────────────────

def _copy_dir_to_v2(cfg: PropConfig, src: Path, out_dir: Path) -> dict:
    """Plain file-copy for COPY mode: clone v1 dir → v2-named dir; patch summary.json."""
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, out_dir)
    _patch_summary_eval_yaml(out_dir / "summary.json", cfg)
    n = rows_count(out_dir)
    return {"mode": "copy", "n_total": n, "out": str(out_dir)}


async def phase_base(cfg: PropConfig, template, client, sem, limit, dry_run,
                     force: bool = False):
    print("\n" + "=" * 78)
    print(f"PHASE 1/3 — BASE MODELS  [{cfg.prop}, mode={cfg.mode}]")
    print("=" * 78)

    if cfg.mode == "regenerate":
        for m in MODELS:
            existing = latest_v2_base_dir_for(cfg, m)
            if existing and not force:
                print(f"\n--- base regen: {cfg.prop} / {m}  SKIP (already at {existing.name}, n={rows_count(existing)})")
                continue
            print(f"\n--- base regen: {cfg.prop} / {m} ---")
            if dry_run:
                print(f"  (dry-run) would run run_eval.py --checkpoint {MODEL_TO_HF[m]} --eval {cfg.v2_yaml}")
                continue
            regen_one(cfg=cfg, model_dir_name=m, flavor=None,
                      dry_run=False, use_v2_yaml=True, max_test_items=limit)
        return

    plan = plan_base_dirs(cfg)
    # Filter: skip plan entries where a v2 base already exists for that model.
    if not force:
        kept = []
        for src, out_name in plan:
            # out_name shape: "<eval_v2>__<model>__base__<ts>". Recover the model.
            try:
                model = out_name.split("__")[1]
            except IndexError:
                model = "?"
            existing = latest_v2_base_dir_for(cfg, model)
            if existing:
                print(f"  SKIP {model}: already at {existing.name} (n={rows_count(existing)})")
            else:
                kept.append((src, out_name))
        plan = kept
    print(f"  planned: {len(plan)} dir(s)")
    for src, out_name in plan:
        print(f"    {src.name}  →  {out_name}")
    if dry_run:
        print("  --dry-run: skipping API calls.")
        return
    results = []
    for src, out_name in plan:
        out = FT_DIR / out_name
        print(f"\n--- base [{cfg.mode}]: {src.name} ---")
        if cfg.mode == "copy":
            r = _copy_dir_to_v2(cfg, src, out)
            print(f"    copied  (n={r['n_total']})  → {out.name}")
        else:  # rejudge
            r = await rejudge_rows_jsonl(
                cfg=cfg, src=src, out_dir=out, template=template, client=client,
                sem=sem, limit=limit, label=f"base/{src.name.split('__')[1]}",
            )
        results.append(r)
    print("\n  ── base phase summary ──")
    for r in results:
        print(f"    {r}")


async def phase_evalorth(cfg: PropConfig, template, client, sem, limit, dry_run,
                         force: bool = False):
    print("\n" + "=" * 78)
    print(f"PHASE 2/3 — EVAL-ORTHOGONALITY (test_evals)  [{cfg.prop}]")
    print("=" * 78)
    if not cfg.judge_differs:
        print(f"  judge prompt for {cfg.metric_key} is identical v1↔v2 — skipping eval-orth rejudge.")
        return
    if not force and evalorth_already_rejudged(cfg):
        print(f"  SKIP: {cfg.metric_key} already in matrices.json _meta.rejudged_metrics.")
        return
    plan = plan_test_evals_run()
    if not plan:
        print("  no test_evals run found.")
        return
    src, out_name = plan
    print(f"  planned: {src.name}  →  {out_name}")
    if dry_run:
        print("  --dry-run: skipping API calls.")
        return
    out = TE_DIR / out_name
    r = await rejudge_test_evals(
        cfg=cfg, src=src, out_dir=out, template=template, client=client,
        sem=sem, limit=limit,
    )
    print(f"\n  ── eval-orth phase summary ── {r}")


async def phase_sysprompts(cfg: PropConfig, template, client, sem, limit, dry_run,
                           force: bool = False):
    print("\n" + "=" * 78)
    print(f"PHASE 3/3 — SYSTEM-PROMPTED  [{cfg.prop}, mode={cfg.mode}]")
    print("=" * 78)
    # First: populate baseline-empty cells by copying the phase-1 v2 base outputs.
    # Skip if an existing v2 baseline-empty dir already covers each model.
    need_baseline_copy = force or any(
        not has_v2_sp_dir(cfg, m, "baseline-empty") for m in MODELS
    )
    if need_baseline_copy:
        await copy_base_into_sysprompts(
            cfg=cfg, template=template, client=client, sem=sem, limit=limit, dry_run=dry_run,
        )
    else:
        print("\n  baseline-empty copies already on disk; skipping copy.")

    if cfg.mode == "regenerate":
        by_flavor: dict[tuple[str, str], list[Path]] = defaultdict(list)
        for m in MODELS:
            for p in SP_DIR.glob(f"{cfg.eval_v1}__{m}__base__sysprompt-*"):
                if not p.is_dir():
                    continue
                parts = p.name.split("__")
                if len(parts) < 5:
                    continue
                flavor = parts[3].removeprefix("sysprompt-")
                if flavor == "baseline-empty":
                    continue
                by_flavor[(m, flavor)].append(p)
        pairs = sorted(by_flavor.keys())
        # Skip pairs that already have a v2 dir on disk (resume semantics).
        if not force:
            kept_pairs = []
            skipped = 0
            for mfl in pairs:
                if has_v2_sp_dir(cfg, *mfl):
                    skipped += 1
                else:
                    kept_pairs.append(mfl)
            if skipped:
                print(f"  resume: {skipped} (model × flavor) pairs already have v2 dirs — skipping.")
            pairs = kept_pairs
        print(f"\n  planned: {len(pairs)} (model × flavor) regen jobs")
        if dry_run:
            for m, fl in pairs[:5]:
                print(f"    {m} / {fl}")
            if len(pairs) > 5:
                print(f"    ... +{len(pairs) - 5} more")
            print("  --dry-run: skipping run_eval.py invocations.")
            return
        for i, (m, fl) in enumerate(pairs, 1):
            print(f"\n--- sp regen [{i}/{len(pairs)}]: {cfg.prop} / {m} / {fl} ---")
            regen_one(cfg=cfg, model_dir_name=m, flavor=fl,
                      dry_run=False, use_v2_yaml=True, max_test_items=limit)
        return

    # COPY or REJUDGE: iterate over the v1 sysprompt dirs.
    plan = plan_sysprompts_dirs(cfg)
    if not force:
        # Skip (model, flavor) pairs already present in v2.
        kept = []
        skipped = 0
        for src, out_name in plan:
            parts = src.name.split("__")
            try:
                model = parts[1]
                flavor = parts[3].removeprefix("sysprompt-")
            except IndexError:
                kept.append((src, out_name)); continue
            if has_v2_sp_dir(cfg, model, flavor):
                skipped += 1
            else:
                kept.append((src, out_name))
        if skipped:
            print(f"  resume: {skipped} (model × flavor) pairs already have v2 dirs — skipping.")
        plan = kept
    print(f"\n  planned: {len(plan)} dir(s) (baseline-empty excluded — already copied)")
    by_model = defaultdict(int)
    for src, _ in plan:
        parts = src.name.split("__")
        if len(parts) >= 2:
            by_model[parts[1]] += 1
    for m, c in by_model.items():
        print(f"    {m}: {c} dirs")
    if dry_run:
        print("  --dry-run: skipping API calls.")
        return
    results = []
    for i, (src, out_name) in enumerate(plan, 1):
        out = SP_DIR / out_name
        flavor = src.name.split("__")[3] if "__" in src.name else "?"
        model = src.name.split("__")[1] if "__" in src.name else "?"
        print(f"\n--- sp [{i}/{len(plan)}] [{cfg.mode}]: {model} / {flavor} ---")
        if cfg.mode == "copy":
            # COPY mode: check size first; if v1 < min_rows we shouldn't have
            # picked this dir (stage 0 would have regenerated), but defensively
            # honour a small-rows fallback by skipping with a warning.
            n_src = rows_count(src)
            if n_src < 1:
                print(f"    SKIP empty src: {src}")
                continue
            r = _copy_dir_to_v2(cfg, src, out)
            print(f"    copied  (n={r['n_total']})  → {out.name}")
        else:  # rejudge
            r = await rejudge_rows_jsonl(
                cfg=cfg, src=src, out_dir=out, template=template, client=client,
                sem=sem, limit=limit, label=f"sp/{model}/{flavor}",
            )
        results.append((model, flavor, r))
    print("\n  ── sysprompts phase summary ──")
    for m, flv, r in results:
        print(f"    {m} / {flv}: {r}")


# ─── Main ─────────────────────────────────────────────────────────────────────

async def run_one_prop(prop: str, args, client: AsyncOpenAI, sem: asyncio.Semaphore):
    cfg = detect_mode(prop, args.min_rows, override=args.mode)
    # Template needed for any judge-based mode (rejudge, or fallback during regenerate
    # for the baseline-empty copy step if no v2 base exists). load_v2_template
    # returns the v2 judge prompt template; safe to load even in copy mode.
    template = load_v2_template(cfg)

    print("\n" + "#" * 78)
    print(f"# PROP: {cfg.prop}  mode={cfg.mode}  judge_differs={cfg.judge_differs}")
    print(f"#   v1 yaml: {cfg.v1_yaml}")
    print(f"#   v2 yaml: {cfg.v2_yaml}")
    print(f"#   metric : {cfg.metric_key}")
    print(f"#   judge sha (v2): {template_sha(template)}  ({len(template)} chars)")
    print("#" * 78)

    if args.phase in ("regen", "all") and not args.skip_regen:
        await phase_regen(cfg, args.min_rows, args.dry_run, limit=args.limit)
    if args.phase == "regen":
        return
    if args.phase in ("base", "all"):
        await phase_base(cfg, template, client, sem, args.limit, args.dry_run,
                         force=args.force)
    if args.phase in ("evalorth", "all"):
        await phase_evalorth(cfg, template, client, sem, args.limit, args.dry_run,
                             force=args.force)
    if args.phase in ("sysprompts", "all"):
        await phase_sysprompts(cfg, template, client, sem, args.limit, args.dry_run,
                               force=args.force)


async def main_async(args):
    if args.phase != "regen" and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY not set. Expected it in "
            f"{JOHANNES_ROOT / '.env'} (via python-dotenv) or in the env."
        )

    props = [s.strip() for s in args.prop.split(",") if s.strip()]
    if not props:
        raise SystemExit("--prop must list at least one propensity.")

    print("=" * 78)
    print(f"REJUDGE EVAL — props={props}  judge={JUDGE_MODEL}  concurrency={CONCURRENCY}")
    print("=" * 78)
    print(f"  --dry-run    : {args.dry_run}")
    print(f"  --limit      : {args.limit}")
    print(f"  --phase      : {args.phase}")
    print(f"  --min-rows   : {args.min_rows}")
    print(f"  --skip-regen : {args.skip_regen}")
    print(f"  --mode       : {args.mode or '(auto-detect)'}")

    # Preview the resolved mode for each prop up front so the user can sanity check.
    print("\n  detected modes:")
    for p in props:
        cfg = detect_mode(p, args.min_rows, override=args.mode)
        print(f"    {p:<24s} → mode={cfg.mode}  judge_differs={cfg.judge_differs}")

    t0 = time.time()
    client = AsyncOpenAI() if args.phase != "regen" else None
    sem = asyncio.Semaphore(CONCURRENCY)
    for p in props:
        await run_one_prop(p, args, client, sem)
    print(f"\nDONE in {time.time() - t0:.0f}s.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prop", required=True,
                   help="Propensity name, or comma-separated list "
                        "(e.g. 'reward-hacking,agreeableness,neuroticism').")
    p.add_argument("--mode", choices=["copy", "rejudge", "regenerate"], default=None,
                   help="Override the auto-detected mode (per-prop).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan; do not call the OpenAI API or Tinker.")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only N rows per dir (smoke test).")
    p.add_argument("--phase", choices=["regen", "base", "evalorth", "sysprompts", "all"],
                   default="all",
                   help="Which phase(s) to run (default: all, which runs stage 0 then 1-3).")
    p.add_argument("--min-rows", type=int, default=100,
                   help="Threshold: dir is 'under-sized' if rows.jsonl is below this (default 100). "
                        "Also used to decide between rejudge and regenerate modes.")
    p.add_argument("--skip-regen", action="store_true",
                   help="Skip stage 0 even when --phase=all (rejudge/copy whatever exists).")
    p.add_argument("--force", action="store_true",
                   help="Re-run all (model × flavor) entries even when a v2 dir already exists.")
    args = p.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
