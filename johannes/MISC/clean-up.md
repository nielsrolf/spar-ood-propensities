# Repo Slim-Down + HF Migration Plan

Goal: move all eval-result data files (`rows.jsonl`, `summary.json`,
`coherence_rows.jsonl`, `coherence_summary.json`, `judgments.jsonl`) out of git
and into the HF dataset `jo-chen/cross-elicit-evals` (private). Then rewrite
git history to reclaim ~2 GB of pack file. After this, future evals
auto-push to HF and never enter git.

Scope: `johannes/` only. `niels/`, `shared/`, etc. are out of scope.

Legend: ✅ done · ⏳ in progress · 🔲 not started · ⚡ parallel-safe (can do
while backfill runs)

---

## Phase 1 — Cleanup of stray top-level eval dirs ✅

302 empty / partial eval-output dirs at the top of `eval_results/` (created
by `run_eval.py` runs that were never moved into `finetuning/` etc.).

**Affected:**
- `johannes/cross-elicit/eval_results/*_eval__*/` — all 302 deleted.

---

## Phase 2 — Build `eval_sync.py` ✅

CLI for HF push/pull/verify/backfill, single-file. Authenticates via
`johannes/.env` HF_TOKEN.

**Affected:**
- `johannes/cross-elicit/scripts/eval_sync.py` (new, ~150 lines).

Subcommands tested: `create-repo`, `push`, `pull`, `verify`, `backfill`.

---

## Phase 3 — Backfill HF ✅ (12:07 → 12:31, ~24 min)

One-shot upload of every existing matching file in `eval_results/` (31,829
files, 5.1 GB) via `upload_large_folder`. Resumable, batched commits.

**Affected (read-only):**
- `johannes/cross-elicit/eval_results/finetuning/**/{rows,coherence_rows}.jsonl`
- `johannes/cross-elicit/eval_results/finetuning/**/{summary,coherence_summary}.json`
- `johannes/cross-elicit/eval_results/sys_prompts/**/{rows,coherence_rows}.jsonl`
- `johannes/cross-elicit/eval_results/sys_prompts/**/{summary,coherence_summary}.json`
- `johannes/cross-elicit/eval_results/test_evals/**/judgments.jsonl`

**On HF:** `jo-chen/cross-elicit-evals` (private). Layout mirrors local 1:1.
Final remote count: 31,830 files (31,829 patterns + auto `.gitattributes`).

---

## Phase 4 — Verify backfill complete ✅ (12:45)

`eval_sync.py verify`:
- local: 31,829 files matching patterns
- remote: 31,829 files matching patterns
- only-local: 0
- only-remote: 0

---

## Phase 5 — Wire auto-push into orchestrators ✅ (12:39 → 12:42)

Add a `push_or_mark_pending(eval_dir)` helper in `eval_sync.py` (best-effort
upload; on failure, drop `.push_pending` marker for `verify` to retry). Then
call it at six sites (one line each):

**Files to modify:**

| File | Line | Insert after |
|---|---|---|
| `johannes/cross-elicit/scripts/run_all_evals.py` | ~55 | `shutil.move(out_dir, new_path)` in `move_to_finetuning()` |
| `johannes/cross-elicit/scripts/finetune.py` | ~95 | `shutil.move(out_dir, new_path)` in `_move_eval_output_to_finetuning()` |
| `johannes/cross-elicit/scripts/sys_prompt_diag.py` | ~184 | `shutil.move(out_dir, new_path)` in `relabel_with_sysprompt()` |
| `johannes/cross-elicit/scripts/sys_prompt_offdiag.py` | ~164 | `shutil.move(out_dir, new_path)` in `relabel_with_sysprompt()` |
| `johannes/cross-elicit/scripts/orthogonality_of_evals.py` | ~626 | `judgments.jsonl` write block (end of save loop) |
| `johannes/cross-elicit/scripts/judge_coherence.py` | ~433 | `coherence_summary.json` write (per-folder) |

Plus extend `eval_sync.py` with:
- `push_or_mark_pending(eval_dir: Path) -> None` — programmatic API
- `verify --push-pending` — scan eval_results for `.push_pending` markers, retry

**Files to modify:**
- `johannes/cross-elicit/scripts/eval_sync.py` (add helper + flag)

---

## Phase 6 — Gitignore the data files ✅ (12:42)

Prevent re-bloat from any future commit.

**Files to modify:**
- `.gitignore` (top-level) — append patterns:
  - `johannes/cross-elicit/eval_results/**/rows.jsonl`
  - `johannes/cross-elicit/eval_results/**/summary.json`
  - `johannes/cross-elicit/eval_results/**/coherence_rows.jsonl`
  - `johannes/cross-elicit/eval_results/**/coherence_summary.json`
  - `johannes/cross-elicit/eval_results/**/judgments.jsonl`
  - (already there) `johannes/cross-elicit/models/**/code.diff`
  - `johannes/cross-elicit/eval_results/**/.push_pending`

---

## Phase 7 — Pre-commit framework hook ✅ (12:43)

Local guard so anyone who staged a forbidden file gets caught before commit.

**Files to create:**
- `.pre-commit-config.yaml` (top-level) — uses `pre-commit-hooks/check-added-large-files` + a small custom shell hook that fails if any staged path matches the five filename patterns under `johannes/cross-elicit/eval_results/`.
- `johannes/cross-elicit/scripts/check_no_eval_data.sh` (or inline in the YAML).

Each developer runs `pre-commit install` once after cloning — to be added to
the README.

---

## Phase 8 — GitHub Action server-side guard ✅ (12:43)

CI fails if a PR diff adds any forbidden file. Catches anyone who skipped
the local hook.

**Files to create:**
- `.github/workflows/no-large-evals.yml`

Logic: `git diff --name-only origin/main...HEAD | grep -E ...` and exit 1 on
match.

---

## Phase 9 — README / CONTRIBUTING update ✅ (12:44)

Document the new workflow.

**Files to modify:**
- `README.md` (top-level) — section: "Eval data lives on HF: after cloning,
  run `huggingface-cli login` then `python johannes/cross-elicit/scripts/eval_sync.py pull`. Run
  `pre-commit install` once."
- `johannes/cross-elicit/README.md` (if exists, else create) — eval-specific
  details, push behavior, .push_pending recovery.

---

## Phase 10 — Maintenance-window prep 🔲

Coordinate before the destructive step.

**Affected:** repo settings on GitHub (temporarily allow force-push to main),
team Slack/email announcement.

---

## Phase 11 — Tag pre-rewrite backup 🔲

Insurance.

**Affected:**
- `git tag pre-hf-rewrite-2026-05-08 main`
- `git push origin pre-hf-rewrite-2026-05-08`
- Optional: push tag to a personal fork as belt-and-braces.

---

## Phase 12 — Run `git filter-repo` 🔲

Strip the data files from all of history.

**Patterns to remove (per `git filter-repo --invert-paths --path-glob ...`):**
- `johannes/cross-elicit/eval_results/**/rows.jsonl`
- `johannes/cross-elicit/eval_results/**/summary.json`
- `johannes/cross-elicit/eval_results/**/coherence_rows.jsonl`
- `johannes/cross-elicit/eval_results/**/coherence_summary.json`
- `johannes/cross-elicit/eval_results/**/judgments.jsonl`
- `johannes/cross-elicit/models/**/code.diff`

**Affected:** every commit gets a new SHA. `.git/objects/pack` shrinks from
4.2 GB → ~2.3 GB (estimated).

---

## Phase 13 — Force-push 🔲

`git push --force-with-lease origin main`.

**Affected:** GitHub remote `main`.

---

## Phase 14 — Re-clone verification 🔲

Fresh clone in a tmpdir, confirm pack < 2.5 GB and that `eval_sync.py pull`
populates `eval_results/` correctly.

**Affected:** none (read-only).

---

# Where we are now (2026-05-08 12:45)

✅ **Phases 1, 2, 3, 4, 5, 6, 7, 8, 9 done.**

Remaining: **Phases 10–14** — maintenance window, tag pre-rewrite-backup,
`git filter-repo`, `--force-with-lease` push, re-clone verify. Destructive
steps; needs your "go" + a coordination signal.
