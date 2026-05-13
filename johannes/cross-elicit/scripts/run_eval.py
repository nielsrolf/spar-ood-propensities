"""
Run a cross-elicit eval against a tinker checkpoint (or a base model).

Pipeline (per item with meta.split == 'test'):
  1. For each paraphrase, sample SAMPLES_PER_PARAPHRASE answers from the
     model under test (tinker, temperature=1).
  2. For each (metric, question, answer): format the metric's judge prompt,
     call the judge LLM (default OpenAI), and store the full text response.
  3. Parse each judge text via regex → numeric / null / 'fail' bucket.

Per-row records and a summary go to cross-elicit/eval_results/<run-dir>.
Final stdout: per-metric min / mean / max / n_nulls / n_numeric / n_fails.

────────────────────────────────────────────────────────────────────────────
USAGE EXAMPLES
────────────────────────────────────────────────────────────────────────────

All examples assume the cwd is `cross-elicit/scripts/`. `OPENAI_API_KEY` +
`TINKER_API_KEY` are loaded automatically from `<repo>/johannes/.env` (the
grandparent of this script's directory) if present; otherwise set them in
the environment yourself.

If `tinker_cookbook` isn't pip-installed, point at a local checkout via the
`TINKER_COOKBOOK_PATH` env var, e.g.
`TINKER_COOKBOOK_PATH=/path/to/tinker-cookbook python run_eval.py ...`.

# 1. Run a finetune.py LoRA checkpoint at a specific epoch.
#    --checkpoint can be relative (resolved from cwd) or absolute.
python run_eval.py \\
    --checkpoint ../models/effort-plus-meta-llama-Llama-3.1-8B-Instruct-<ts> \\
    --epoch 2 \\
    --eval ../evals/effort/effort_eval.yaml

# 2. Latest epoch from a run dir (omit --epoch).
python run_eval.py \\
    --checkpoint ../models/effort-minus-meta-llama-Llama-3.1-8B-Instruct-<ts> \\
    --eval ../evals/effort/effort_eval.yaml

# 3. Evaluate the BASE model (no LoRA) — pass a HF-style name to --checkpoint.
#    --epoch is ignored in this mode.
python run_eval.py \\
    --checkpoint meta-llama/Llama-3.1-8B-Instruct \\
    --eval ../evals/effort/effort_eval.yaml

# 4. Quick smoke test: shrink the test split to N randomly-sampled items
#    via --max-test-items (overrides MAX_TEST_ITEMS in the config block).
python run_eval.py \\
    --checkpoint ../models/effort-plus-meta-llama-Llama-3.1-8B-Instruct-<ts> \\
    --epoch 2 \\
    --max-test-items 10

# 5. Run the model under a system prompt (inline string or file). The system
#    prompt is applied to the model under test only; it's recorded in
#    summary.json under "system_prompt" / "system_prompt_source".
python run_eval.py \\
    --checkpoint meta-llama/Llama-3.1-8B-Instruct \\
    --eval ../evals/effort/effort_eval.yaml \\
    --system-prompt "You are a careful assistant."
python run_eval.py \\
    --checkpoint meta-llama/Llama-3.1-8B-Instruct \\
    --eval ../evals/effort/effort_eval.yaml \\
    --system-prompt-file ../evals/effort/system_prompts/some_prompt.txt

# Outputs land in cross-elicit/eval_results/<eval>__<ckpt-label>__<ts>/
#   rows.jsonl     — one row per (item, paraphrase, sample, metric)
#   summary.json   — per-metric min/mean/max + counts
"""

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime

import yaml
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Paths — derived from this file's location so the script works in any clone.
# Layout: <repo>/.../johannes/cross-elicit/scripts/run_eval.py
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CROSS_ELICIT_ROOT = os.path.dirname(SCRIPT_DIR)
JOHANNES_ROOT = os.path.dirname(CROSS_ELICIT_ROOT)

# Load API keys from johannes/.env (OPENAI_API_KEY, TINKER_API_KEY, ...) if
# present; silently no-ops otherwise so the env can supply them directly.
load_dotenv(os.path.join(JOHANNES_ROOT, ".env"))

# Allow pointing at a local tinker-cookbook checkout when it isn't pip-installed.
_TINKER_COOKBOOK_PATH = os.environ.get("TINKER_COOKBOOK_PATH")
if _TINKER_COOKBOOK_PATH:
    sys.path.append(_TINKER_COOKBOOK_PATH)

import tinker  # noqa: E402
from tinker import types  # noqa: E402
from tinker_cookbook import renderers  # noqa: E402
from tinker_cookbook.model_info import get_recommended_renderer_name  # noqa: E402
from tinker_cookbook.tokenizer_utils import get_tokenizer  # noqa: E402


def select_renderer_name(base_model: str) -> str:
    """Renderer name to use for `base_model`. Nemotron-3 is a hybrid
    thinking/non-thinking model; we deliberately default to the
    non-thinking variant so all callers (this script, finetune.py,
    sys_prompt_diag, sys_prompt_offdiag, run_overnight_everything) get
    consistent, reasoning-off behavior. Other models keep the
    tinker_cookbook recommended default."""
    name = get_recommended_renderer_name(base_model)
    if name == "nemotron3":
        return "nemotron3_disable_thinking"
    return name

from openai import AsyncOpenAI, RateLimitError  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Config — edit these
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_EVAL_YAML = os.path.join(CROSS_ELICIT_ROOT, "evals", "effort", "effort_eval.yaml")
EVAL_RESULTS_DIR = os.path.join(CROSS_ELICIT_ROOT, "eval_results")

SAMPLES_PER_PARAPHRASE = 1
JUDGE_MODEL = "gpt-5.4-mini"

# Concurrency for tinker (sampling) and openai (judge) calls is AIMD-controlled
# (see AdaptiveLimiter below): starts at *_START, +1 per AIMD_SUCCESS_STEP
# successes up to *_MAX, halves on rate-limit error down to *_MIN. All
# overridable via env vars so several runs can share API quota.
SAMPLE_CONCURRENCY_START = int(os.environ.get("SAMPLE_CONCURRENCY", "32"))
SAMPLE_CONCURRENCY_MIN = int(os.environ.get("SAMPLE_CONCURRENCY_MIN", "1"))
SAMPLE_CONCURRENCY_MAX = int(os.environ.get("SAMPLE_CONCURRENCY_MAX", "128"))
JUDGE_CONCURRENCY_START = int(os.environ.get("JUDGE_CONCURRENCY", "256"))
JUDGE_CONCURRENCY_MIN = int(os.environ.get("JUDGE_CONCURRENCY_MIN", "1"))
JUDGE_CONCURRENCY_MAX = int(os.environ.get("JUDGE_CONCURRENCY_MAX", "1024"))
AIMD_SUCCESS_STEP = int(os.environ.get("AIMD_SUCCESS_STEP", "3"))
AIMD_MAX_RETRIES = int(os.environ.get("AIMD_MAX_RETRIES", "8"))

# Quick-mode: limit how many test-split items are evaluated.
# None  → use every test-split item.
# int N → random-sample N items from the test split (seed for reproducibility).
# Overridable on the CLI via --max-test-items / --no-max-test-items.
MAX_TEST_ITEMS: int | None = None
TEST_SAMPLE_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive concurrency (AIMD)
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveLimiter:
    """AIMD-controlled async concurrency limiter.

    `target` is the live cap on in-flight calls. on_success() bumps target by 1
    every `success_step` successes (up to `hi`); on_rate_limit() halves target
    (down to `lo`), debounced so a burst of simultaneous 429s only halves once.
    Use as `async with limiter: ...` to take a slot.
    """

    def __init__(self, name: str, start: int, lo: int, hi: int, success_step: int):
        self.name = name
        self._target = max(lo, min(hi, start))
        self._lo = lo
        self._hi = hi
        self._success_step = success_step
        self._in_flight = 0
        self._success_count = 0
        self._last_shrink = 0.0
        self._cond = asyncio.Condition()

    @property
    def target(self) -> int:
        return self._target

    def state_str(self) -> str:
        return f"{self.name} target={self._target} in_flight={self._in_flight}"

    async def __aenter__(self):
        async with self._cond:
            while self._in_flight >= self._target:
                await self._cond.wait()
            self._in_flight += 1
        return self

    async def __aexit__(self, exc_type, exc, tb):
        async with self._cond:
            self._in_flight -= 1
            self._cond.notify()

    async def on_success(self):
        async with self._cond:
            self._success_count += 1
            if self._success_count >= self._success_step and self._target < self._hi:
                self._target += 1
                self._success_count = 0
                self._cond.notify()
                print(f"  [limiter:{self.name}] +1 → {self._target}")

    async def on_rate_limit(self):
        async with self._cond:
            now = time.monotonic()
            if now - self._last_shrink < 1.0:
                return  # debounce: a burst of simultaneous 429s only halves once
            self._last_shrink = now
            self._success_count = 0
            if self._target > self._lo:
                old = self._target
                self._target = max(self._lo, self._target // 2)
                print(f"  [limiter:{self.name}] rate-limit → {old}→{self._target}")


def _is_rate_limit_err(e: BaseException) -> bool:
    if isinstance(e, RateLimitError):
        return True
    status = getattr(e, "status_code", None) or getattr(e, "status", None)
    if status == 429:
        return True
    s = repr(e).lower()
    return any(t in s for t in (
        "rate limit", "ratelimit", "429", "too many requests", "quota exceeded",
    ))


def _retry_after_of(e: BaseException) -> float | None:
    resp = getattr(e, "response", None)
    headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        ra = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:
        return None
    if ra is None:
        return None
    try:
        return float(ra)
    except (TypeError, ValueError):
        return None


async def call_with_aimd(limiter: AdaptiveLimiter, fn, max_retries: int = AIMD_MAX_RETRIES):
    """Run `fn()` (zero-arg, returns awaitable) under `limiter` with AIMD.

    On rate-limit errors: signals the limiter to halve, sleeps Retry-After (or
    exponential backoff fallback), and retries up to `max_retries`. Other
    exceptions propagate immediately.
    """
    backoff = 1.0
    last_err: BaseException | None = None
    wait = 0.0
    for _ in range(max_retries):
        async with limiter:
            try:
                result = await fn()
            except Exception as e:
                if _is_rate_limit_err(e):
                    await limiter.on_rate_limit()
                    last_err = e
                    wait = _retry_after_of(e) or backoff
                else:
                    raise
            else:
                await limiter.on_success()
                return result
        await asyncio.sleep(wait)
        backoff = min(backoff * 2, 60.0)
    raise last_err if last_err is not None else RuntimeError("call_with_aimd: retry exhausted")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint resolution
# ─────────────────────────────────────────────────────────────────────────────

def resolve_checkpoint(spec: str, epoch: int | None) -> tuple[str, str | None, str]:
    """Return (base_model, model_path_or_None, label).

    `spec` is either a path to a finetune.py-style run directory (containing
    run_config.json + checkpoints.jsonl) or a HuggingFace-style base-model name.
    """
    if os.path.isdir(spec):
        run_dir = os.path.abspath(spec)
        with open(os.path.join(run_dir, "run_config.json")) as f:
            run_config = json.load(f)
        base_model = run_config["model_name"]

        ckpt_jsonl = os.path.join(run_dir, "checkpoints.jsonl")
        with open(ckpt_jsonl) as f:
            entries = [json.loads(l) for l in f if l.strip()]
        epoch_entries = [
            e for e in entries
            if isinstance(e.get("name"), str)
            and e["name"].startswith("epoch_")
            and "sampler_path" in e
        ]
        if not epoch_entries:
            epoch_entries = [e for e in entries if "sampler_path" in e]
        if not epoch_entries:
            raise SystemExit(f"No usable entries with sampler_path in {ckpt_jsonl}")

        if epoch is None:
            chosen = max(epoch_entries, key=lambda e: e.get("epoch") or 0)
        else:
            matches = [e for e in epoch_entries if e.get("epoch") == epoch]
            if not matches:
                raise SystemExit(f"No checkpoint with epoch={epoch} in {ckpt_jsonl}")
            chosen = matches[-1]

        label = f"{os.path.basename(run_dir)}__epoch{chosen.get('epoch')}"
        return base_model, chosen["sampler_path"], label

    base_model = spec
    label = base_model.replace("/", "-") + "__base"
    return base_model, None, label


# ─────────────────────────────────────────────────────────────────────────────
# Sampling
# ─────────────────────────────────────────────────────────────────────────────

def _content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(p["text"])
        return "\n".join(parts)
    return str(content)


async def sample_answers(
    sampling_client,
    renderer,
    prompt: str,
    n: int,
    sample_limiter: AdaptiveLimiter,
    system_prompt: str | None = None,
) -> list[str]:
    messages: list[renderers.Message] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    model_input = renderer.build_generation_prompt(messages)
    sampling_params = types.SamplingParams(
        temperature=1.0,
        stop=renderer.get_stop_sequences(),
    )
    result = await call_with_aimd(
        sample_limiter,
        lambda: sampling_client.sample_async(
            prompt=model_input, num_samples=n, sampling_params=sampling_params,
        ),
    )
    answers: list[str] = []
    for seq in result.sequences:
        parsed, _ = renderer.parse_response(seq.tokens)
        answers.append(_content_to_str(parsed["content"]))
    return answers


# ─────────────────────────────────────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────────────────────────────────────

NULL_RE = re.compile(r"^(null|none|no\s*information)$", re.IGNORECASE)
INT_RE = re.compile(r"-?\d+")


def parse_judge_text(text: str) -> tuple[str, int | None]:
    """Returns ('numeric', int) | ('null', None) | ('fail', None)."""
    if not isinstance(text, str):
        return ("fail", None)
    s = text.strip().strip('"').strip("'").strip().rstrip(".")
    if NULL_RE.match(s):
        return ("null", None)
    m = INT_RE.search(s)
    if m is None:
        return ("fail", None)
    n = int(m.group())
    if 0 <= n <= 100:
        return ("numeric", n)
    return ("fail", None)


async def call_judge(client: AsyncOpenAI, prompt: str, judge_limiter: AdaptiveLimiter,
                     model: str | None = None) -> str:
    try:
        resp = await call_with_aimd(
            judge_limiter,
            lambda: client.chat.completions.create(
                model=model or JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
            ),
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"__JUDGE_ERROR__: {e!r}"


def fill_judge_template(template: str, question: str, answer: str) -> str:
    # Plain replacement — judge templates contain other curly braces beyond
    # {question}/{answer}, which would break str.format.
    return template.replace("{question}", question).replace("{answer}", answer)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

async def run(args):
    base_model, model_path, ckpt_label = resolve_checkpoint(args.checkpoint, args.epoch)

    system_prompt: str | None = None
    system_prompt_source: str | None = None
    if args.system_prompt is not None and args.system_prompt_file is not None:
        raise SystemExit("Pass at most one of --system-prompt / --system-prompt-file.")
    if args.system_prompt is not None:
        system_prompt = args.system_prompt
        system_prompt_source = "cli"
    elif args.system_prompt_file is not None:
        with open(args.system_prompt_file) as f:
            system_prompt = f.read()
        system_prompt_source = os.path.abspath(args.system_prompt_file)
    if system_prompt is not None:
        preview = system_prompt.replace("\n", " ")
        if len(preview) > 100:
            preview = preview[:100] + "…"
        print(f"System prompt ({len(system_prompt)} chars, src={system_prompt_source}): {preview}")

    with open(args.eval) as f:
        items = yaml.safe_load(f) or []
    test_items = [
        it for it in items
        if isinstance(it.get("meta"), dict)
        and (it["meta"].get("split") or "").strip() == "test"
    ]
    print(f"Loaded {len(items)} items from {args.eval}; {len(test_items)} in test split.")
    if not test_items:
        raise SystemExit("No test-split items. Nothing to do.")

    max_items = args.max_test_items
    if max_items is not None and len(test_items) > max_items:
        rng = random.Random(TEST_SAMPLE_SEED)
        test_items = rng.sample(test_items, max_items)
        print(
            f"  → max_test_items={max_items} (seed={TEST_SAMPLE_SEED}); "
            f"sampled {len(test_items)} items: "
            f"{', '.join(it['id'] for it in test_items)}"
        )

    eval_name = os.path.splitext(os.path.basename(args.eval))[0]
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    out_dir = os.path.join(EVAL_RESULTS_DIR, f"{eval_name}__{ckpt_label}__{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    rows_path = os.path.join(out_dir, "rows.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")
    print(f"Output dir: {out_dir}")

    # Tinker setup
    service_client = tinker.ServiceClient()
    sc_kwargs = {"base_model": base_model}
    if model_path is not None:
        sc_kwargs["model_path"] = model_path
    print(f"Tinker: base_model={base_model}  model_path={model_path or '(base)'}")
    sampling_client = service_client.create_sampling_client(**sc_kwargs)
    tokenizer = get_tokenizer(base_model)
    renderer = renderers.get_renderer(
        select_renderer_name(base_model), tokenizer
    )

    # OpenAI judge client
    judge_client = AsyncOpenAI()
    judge_limiter = AdaptiveLimiter(
        "judge", JUDGE_CONCURRENCY_START, JUDGE_CONCURRENCY_MIN,
        JUDGE_CONCURRENCY_MAX, AIMD_SUCCESS_STEP,
    )
    sample_limiter = AdaptiveLimiter(
        "sample", SAMPLE_CONCURRENCY_START, SAMPLE_CONCURRENCY_MIN,
        SAMPLE_CONCURRENCY_MAX, AIMD_SUCCESS_STEP,
    )

    # Phase 1: sample model answers (in parallel across paraphrases)
    n_paraphrases = sum(len(it.get("paraphrases") or []) for it in test_items)
    print(
        f"\nSampling {SAMPLES_PER_PARAPHRASE} answers/paraphrase across "
        f"{n_paraphrases} paraphrases  "
        f"(concurrency start={SAMPLE_CONCURRENCY_START}, "
        f"range [{SAMPLE_CONCURRENCY_MIN},{SAMPLE_CONCURRENCY_MAX}])..."
    )

    async def _sample_with_meta(key, prompt):
        t0 = time.time()
        try:
            answers = await sample_answers(
                sampling_client, renderer, prompt,
                SAMPLES_PER_PARAPHRASE, sample_limiter,
                system_prompt=system_prompt,
            )
            return key, answers, time.time() - t0, None
        except Exception as e:
            return key, [], time.time() - t0, e

    sample_tasks = []
    for it in test_items:
        item_id = it["id"]
        for p_idx, paraphrase in enumerate(it.get("paraphrases") or []):
            sample_tasks.append(
                asyncio.create_task(_sample_with_meta((item_id, p_idx), paraphrase))
            )
    print(f"  queued {len(sample_tasks)} sampling tasks; awaiting first completion...")

    answers_map: dict[tuple[str, int], list[str]] = {}
    t_phase1 = time.time()
    done = 0
    total = len(sample_tasks)
    for fut in asyncio.as_completed(sample_tasks):
        key, answers, dt, err = await fut
        done += 1
        elapsed = time.time() - t_phase1
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else float("inf")
        if err is not None:
            print(
                f"  [{done}/{total}] {key[0]}#{key[1]} FAILED in {dt:.1f}s: {err!r}  "
                f"[{sample_limiter.state_str()}]"
            )
        else:
            n_chars = sum(len(a) for a in answers)
            print(
                f"  [{done}/{total}] {key[0]}#{key[1]} ok in {dt:.1f}s "
                f"({len(answers)} ans, {n_chars} chars)  "
                f"[elapsed {elapsed:.0f}s, ETA {eta:.0f}s]  "
                f"[{sample_limiter.state_str()}]"
            )
        answers_map[key] = answers
    print(f"Sampling done in {time.time() - t_phase1:.0f}s.")

    # Phase 2: judge every (item, paraphrase, sample, metric)
    print(
        f"\nJudging with {JUDGE_MODEL} (default; per-metric `judge_models:` overrides honored) "
        f"(concurrency start={JUDGE_CONCURRENCY_START}, "
        f"range [{JUDGE_CONCURRENCY_MIN},{JUDGE_CONCURRENCY_MAX}])..."
    )
    judge_jobs: list[tuple[dict, asyncio.Task]] = []
    judge_model_by_metric: dict[str, str] = {}  # for the run summary
    for it in test_items:
        item_id = it["id"]
        judge_prompts = it.get("judge_prompts") or {}
        # Per-metric model override: optional `judge_models: {metric: model}` mapping.
        # Falls back to global JUDGE_MODEL for any metric not listed.
        judge_models = it.get("judge_models") or {}
        for p_idx, paraphrase in enumerate(it.get("paraphrases") or []):
            answers = answers_map.get((item_id, p_idx), [])
            for s_idx, ans in enumerate(answers):
                for metric, template in judge_prompts.items():
                    filled = fill_judge_template(template, paraphrase, ans)
                    metric_model = judge_models.get(metric, JUDGE_MODEL)
                    judge_model_by_metric[metric] = metric_model
                    meta = {
                        "item_id": item_id,
                        "paraphrase_idx": p_idx,
                        "sample_idx": s_idx,
                        "metric": metric,
                        "question": paraphrase,
                        "answer": ans,
                        "judge_model": metric_model,
                    }
                    task = asyncio.create_task(
                        call_judge(judge_client, filled, judge_limiter, model=metric_model)
                    )
                    judge_jobs.append((meta, task))
    print(f"  queued {len(judge_jobs)} judge calls; awaiting...")

    rows: list[dict] = []
    t_phase2 = time.time()
    progress_every = max(1, len(judge_jobs) // 20)  # ~20 progress lines total

    # Running per-metric tallies for live display.
    metric_numeric: dict[str, list[int]] = {}
    metric_null: dict[str, int] = {}
    metric_fail: dict[str, int] = {}

    with open(rows_path, "w") as fout:
        for i, (meta, task) in enumerate(judge_jobs, 1):
            judge_text = await task
            bucket, score = parse_judge_text(judge_text)
            row = {
                **meta,
                "judge_response": judge_text,
                "bucket": bucket,
                "score": score,
            }
            rows.append(row)
            fout.write(json.dumps(row) + "\n")
            fout.flush()

            metric = meta["metric"]
            if bucket == "numeric":
                metric_numeric.setdefault(metric, []).append(score)
            elif bucket == "null":
                metric_null[metric] = metric_null.get(metric, 0) + 1
            else:
                metric_fail[metric] = metric_fail.get(metric, 0) + 1

            print(
                f"  [{i}/{len(judge_jobs)}] {meta['item_id']}#"
                f"{meta['paraphrase_idx']}.{meta['sample_idx']} "
                f"{metric} → {bucket}{f'={score}' if score is not None else ''}"
            )

            if i % progress_every == 0 or i == len(judge_jobs):
                elapsed = time.time() - t_phase2
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(judge_jobs) - i) / rate if rate > 0 else float("inf")
                print(
                    f"  ── progress {i}/{len(judge_jobs)} "
                    f"[elapsed {elapsed:.0f}s, ETA {eta:.0f}s]  "
                    f"[{judge_limiter.state_str()}]"
                )
                for m in sorted(set(metric_numeric) | set(metric_null) | set(metric_fail)):
                    nums = metric_numeric.get(m, [])
                    n_null = metric_null.get(m, 0)
                    n_fail = metric_fail.get(m, 0)
                    mean_str = f"{statistics.fmean(nums):.1f}" if nums else "-"
                    min_str = min(nums) if nums else "-"
                    max_str = max(nums) if nums else "-"
                    print(
                        f"     [{m}] numeric={len(nums)} null={n_null} fail={n_fail}  "
                        f"min={min_str} mean={mean_str} max={max_str}"
                    )
    print(f"Wrote {len(rows)} judge rows to {rows_path} (judging took {time.time()-t_phase2:.0f}s)")

    # Phase 3: per-metric stats
    metrics_seen: list[str] = []
    by_metric: dict[str, list[dict]] = {}
    for r in rows:
        m = r["metric"]
        if m not in by_metric:
            by_metric[m] = []
            metrics_seen.append(m)
        by_metric[m].append(r)

    summary: dict = {
        "checkpoint": {
            "spec": args.checkpoint,
            "epoch": args.epoch,
            "base_model": base_model,
            "model_path": model_path,
            "label": ckpt_label,
        },
        "eval_yaml": args.eval,
        "n_test_items": len(test_items),
        "samples_per_paraphrase": SAMPLES_PER_PARAPHRASE,
        "judge_model": JUDGE_MODEL,
        "judge_model_by_metric": judge_model_by_metric,  # records any per-metric overrides
        "system_prompt": system_prompt,
        "system_prompt_source": system_prompt_source,
        "metrics": {},
    }

    print("\n" + "=" * 70)
    print(f"Results — {eval_name}  |  ckpt: {ckpt_label}")
    print("=" * 70)
    for metric in metrics_seen:
        rs = by_metric[metric]
        numeric = [r["score"] for r in rs if r["bucket"] == "numeric"]
        n_null = sum(1 for r in rs if r["bucket"] == "null")
        n_fail = sum(1 for r in rs if r["bucket"] == "fail")
        stats = {
            "n_total": len(rs),
            "n_numeric": len(numeric),
            "n_nulls": n_null,
            "n_fails": n_fail,
            "min": min(numeric) if numeric else None,
            "max": max(numeric) if numeric else None,
            "mean": statistics.fmean(numeric) if numeric else None,
            "std": statistics.stdev(numeric) if len(numeric) >= 2 else None,
        }
        summary["metrics"][metric] = stats
        mean_str = f"{stats['mean']:.2f}" if stats["mean"] is not None else "-"
        std_str = f"{stats['std']:.2f}" if stats["std"] is not None else "-"
        min_str = stats["min"] if stats["min"] is not None else "-"
        max_str = stats["max"] if stats["max"] is not None else "-"
        print(
            f"\n[{metric}]  n={stats['n_total']}  "
            f"numeric={stats['n_numeric']}  null={stats['n_nulls']}  fail={stats['n_fails']}"
        )
        print(f"  min={min_str}  mean={mean_str}  std={std_str}  max={max_str}")

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to a finetune.py run dir, or a HuggingFace base-model name."
    )
    parser.add_argument(
        "--epoch", type=int, default=None,
        help="Epoch index to load (run-dir checkpoints only). Default: latest."
    )
    parser.add_argument(
        "--eval", default=DEFAULT_EVAL_YAML,
        help="Path to an eval YAML (test split is auto-filtered)."
    )
    parser.add_argument(
        "--max-test-items", type=int, default=MAX_TEST_ITEMS,
        help=(
            "Random-sample this many items from the test split for a quick run. "
            f"Default in-code: {MAX_TEST_ITEMS} (None = use full test split). "
            "Pass any positive int to override."
        ),
    )
    parser.add_argument(
        "--system-prompt", default=None,
        help=(
            "System prompt for the model under test (not the judge). "
            "Default: no system prompt. Mutually exclusive with --system-prompt-file."
        ),
    )
    parser.add_argument(
        "--system-prompt-file", default=None,
        help=(
            "Path to a file whose contents are used as the system prompt for the "
            "model under test. Mutually exclusive with --system-prompt."
        ),
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
