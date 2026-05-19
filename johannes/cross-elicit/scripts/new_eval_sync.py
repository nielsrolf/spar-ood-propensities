"""
Sync `cross-elicit/new_eval_results/` to/from the HF dataset repo
`jo-chen/cross-elicit-evals`.

Unlike `eval_sync.py`, this script ONLY touches `new_eval_results/` — it never
reads or writes to `cross-elicit/results/`. The HF repo and the
`new_eval_results/` HF prefix are the same as in `eval_sync.py`, so the two
scripts share a remote layout.

Three nested scope levels — each a superset of the previous. Flags `--allnumbers`
and `--fullevals` select the level; passing both is treated as `--fullevals`
(the strict superset).

  Level 1 (default, no flag) — summary files:
      `new_eval_results/scores/*.json` matching:
          basemodel_scores_*.json
          finetuned_scores_*.json
          sysprompts_scores_*.json
          eval-orthogonality_scores.json
      Non-recursive (the `scores/eval-orthogonality/` subdir is not included).

  Level 2 (`--allnumbers`) — summary files + raw numbers:
      Level 1
      + `new_eval_results/raw_numbers/**/*.{pt,pkl,pickle}` (recursive) —
        tensor files and general pickle files.

  Level 3 (`--fullevals`) — summary files + raw numbers + full eval dirs:
      Level 2
      + every file under `new_eval_results/` whose basename is in
        EVAL_PATTERNS (`rows.jsonl`, `summary.json`, `coherence_*.{json,jsonl}`,
        `judgments.jsonl`, `matrices.json`, `eval-orthogonality_scores.json`).
      Filtering by basename keeps logs / `.DS_Store` / HTML out.

Subcommands (all level-aware unless noted):
  push          [--allnumbers|--fullevals]
                               Bulk upload at the selected level.
  push-dir      <eval_dir>     Upload one eval dir under new_eval_results/
                               (EVAL_PATTERNS only — not level-aware).
  pull          [--allnumbers|--fullevals] [--filter G]
                               Snapshot-download into new_eval_results/.
                               --filter narrows by a path-glob under
                               new_eval_results/; only meaningful with
                               --fullevals.
  verify        [--allnumbers|--fullevals] [--push] [--show]
                               Diff local vs HF at the selected level. With
                               --push, first tries a single-commit
                               upload_folder; on non-rate-limit failure, falls
                               back to create_commit batches of --batch-size
                               files.
  backfill      [--allnumbers|--fullevals]
                               Bulk upload via upload_large_folder
                               (batched, resumable, parallel).
  create-repo                  One-time: create the private dataset repo on HF.

Examples:
  # Pull only the 10 score JSONs (default level):
  python new_eval_sync.py pull

  # Push summary files + raw_numbers/*.pt:
  python new_eval_sync.py push --allnumbers

  # Pull everything (summary + raw + full eval dirs):
  python new_eval_sync.py pull --fullevals

  # Pull just one sub-tree at level 3:
  python new_eval_sync.py pull --fullevals --filter 'finetuning/*agreeableness*'

  # Push one finished eval dir:
  python new_eval_sync.py push-dir ../new_eval_results/finetuning/qwen3-8b/agreeableness/seed0

  # Diff and push anything missing on HF at level 2:
  python new_eval_sync.py verify --allnumbers --show --push

  # Bulk (re)upload everything, resumable + parallel:
  python new_eval_sync.py backfill --fullevals --workers 8
"""
from __future__ import annotations

import argparse
import os
import sys
from fnmatch import fnmatch
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import CommitOperationAdd, HfApi, snapshot_download
from huggingface_hub.errors import HfHubHTTPError

SCRIPT_DIR = Path(__file__).resolve().parent
CROSS_ELICIT_ROOT = SCRIPT_DIR.parent
JOHANNES_ROOT = CROSS_ELICIT_ROOT.parent
EVAL_RESULTS_DIR = CROSS_ELICIT_ROOT / "new_eval_results"
SCORES_DIR = EVAL_RESULTS_DIR / "scores"
RAW_NUMBERS_DIR = EVAL_RESULTS_DIR / "raw_numbers"

load_dotenv(JOHANNES_ROOT / ".env")

REPO_ID = "jo-chen/cross-elicit-evals"
REPO_TYPE = "dataset"

# Local cross-elicit/new_eval_results/<x>  ↔  HF  new_eval_results/<x>
EVAL_RESULTS_HF_PREFIX = "new_eval_results"

# Level 1: score JSONs directly in scores/ (non-recursive).
SCORES_PATTERNS = (
    "basemodel_scores_*.json",
    "finetuned_scores_*.json",
    "sysprompts_scores_*.json",
    "eval-orthogonality_scores.json",
)

# Level 2: raw number files under raw_numbers/ (recursive). Tensor files
# (.pt) and general pickle files (.pkl, .pickle).
RAW_NUMBERS_EXTS = (".pt", ".pkl", ".pickle")

# Level 3: eval-dir file basenames matched anywhere under new_eval_results/.
# Mirrors PATTERNS in eval_sync.py.
EVAL_PATTERNS = (
    "rows.jsonl",
    "summary.json",
    "coherence_rows.jsonl",
    "coherence_summary.json",
    "judgments.jsonl",
    "matrices.json",
    "eval-orthogonality_scores.json",
)


def _api(require_token: bool = True) -> HfApi:
    """HF API client. The dataset is public, so read-only ops (pull, verify
    without --push) work without HF_TOKEN; pass require_token=False for those."""
    token = os.environ.get("HF_TOKEN")
    if not token and require_token:
        sys.exit("HF_TOKEN not set (check johannes/.env).")
    return HfApi(token=token) if token else HfApi()


def _rel_under_eval_results(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(EVAL_RESULTS_DIR))
    except ValueError:
        sys.exit(f"{path} is not under {EVAL_RESULTS_DIR}.")


def _resolve_level(args: argparse.Namespace) -> int:
    """1 = scores; 2 = +raw_numbers; 3 = +full eval dirs. --fullevals wins
    over --allnumbers since it's a strict superset."""
    if getattr(args, "fullevals", False):
        return 3
    if getattr(args, "allnumbers", False):
        return 2
    return 1


def _level_label(level: int) -> str:
    return {1: "scores", 2: "scores+rawnumbers", 3: "fullevals"}[level]


# ---------------------------------------------------------------------------
# Local / remote file enumeration
# ---------------------------------------------------------------------------

def _local_files(level: int) -> set[str]:
    """Files matching the given level, returned as paths relative to
    EVAL_RESULTS_DIR. Always cumulative."""
    out: set[str] = set()
    # Level 1: scores/<top-level json>
    if SCORES_DIR.exists():
        for pat in SCORES_PATTERNS:
            for p in SCORES_DIR.glob(pat):
                if p.is_file():
                    out.add(f"scores/{p.name}")
    # Level 2: raw_numbers/**/*.{pt,pkl,pickle}
    if level >= 2 and RAW_NUMBERS_DIR.exists():
        for p in RAW_NUMBERS_DIR.rglob("*"):
            if p.is_file() and p.suffix in RAW_NUMBERS_EXTS:
                out.add(str(p.relative_to(EVAL_RESULTS_DIR)))
    # Level 3: eval-pattern files anywhere under new_eval_results/
    if level >= 3 and EVAL_RESULTS_DIR.exists():
        for root, _dirs, files in os.walk(EVAL_RESULTS_DIR):
            root_path = Path(root)
            for f in files:
                if f in EVAL_PATTERNS:
                    out.add(str((root_path / f).relative_to(EVAL_RESULTS_DIR)))
    return out


def _remote_files(api: HfApi, level: int) -> set[str]:
    """Files under HF `new_eval_results/` at the given level, returned relative
    to that prefix so they line up with _local_files()."""
    prefix = f"{EVAL_RESULTS_HF_PREFIX}/"
    out: set[str] = set()
    for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE):
        if not f.startswith(prefix):
            continue
        rel = f[len(prefix):]
        name = Path(rel).name
        # Level 1: scores/<top-level json>
        if rel.startswith("scores/") and "/" not in rel[len("scores/"):]:
            if any(fnmatch(name, pat) for pat in SCORES_PATTERNS):
                out.add(rel)
                continue
        # Level 2: raw_numbers/**/*.{pt,pkl,pickle}
        if level >= 2 and rel.startswith("raw_numbers/") \
                and any(name.endswith(ext) for ext in RAW_NUMBERS_EXTS):
            out.add(rel)
            continue
        # Level 3: eval-pattern files anywhere
        if level >= 3 and name in EVAL_PATTERNS:
            out.add(rel)
    return out


def _hf_allow_patterns(level: int, filter_glob: str | None = None) -> list[str]:
    """HF snapshot_download allow_patterns covering the given level.

    `filter_glob` (if provided) narrows the level-3 sweep to a sub-tree; it
    doesn't affect the scores or raw_numbers globs because those are already
    location-specific.
    """
    base = f"{EVAL_RESULTS_HF_PREFIX}/"
    patterns: list[str] = [f"{base}scores/{pat}" for pat in SCORES_PATTERNS]
    if level >= 2:
        patterns += [f"{base}raw_numbers/**/*{ext}" for ext in RAW_NUMBERS_EXTS]
    if level >= 3:
        if filter_glob:
            patterns += [f"{base}{filter_glob}/{pat}" for pat in EVAL_PATTERNS]
        else:
            patterns += [f"{base}**/{pat}" for pat in EVAL_PATTERNS]
    return patterns


# ---------------------------------------------------------------------------
# Push helpers
# ---------------------------------------------------------------------------

def _is_rate_limit(exc: HfHubHTTPError) -> bool:
    resp = getattr(exc, "response", None)
    return getattr(resp, "status_code", None) == 429


def _push_robust(
    api: HfApi,
    folder_path: Path,
    relpaths: list[str],
    path_in_repo: str | None,
    batch_size: int,
    label: str,
) -> None:
    """Push `relpaths` (relative to folder_path) to HF.

    Tries a single `upload_folder` commit first. On non-rate-limit failure,
    falls back to explicit `create_commit` calls of ~batch_size files each.
    429 is re-raised immediately.
    """
    if not relpaths:
        return
    n = len(relpaths)
    in_repo_prefix = f"{path_in_repo}/" if path_in_repo else ""
    print(f"\n[{label}] {n} file(s) — trying single-commit upload_folder...")
    try:
        api.upload_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=str(folder_path),
            path_in_repo=path_in_repo,
            allow_patterns=list(relpaths),
            commit_message=f"{label}: {n} files",
        )
        print(f"  [done] 1 commit")
        return
    except HfHubHTTPError as e:
        if _is_rate_limit(e):
            raise
        print(f"  [warn] single-commit failed: {e!s:.200}")

    ops = [
        CommitOperationAdd(
            path_in_repo=f"{in_repo_prefix}{r}",
            path_or_fileobj=str(folder_path / r),
        )
        for r in relpaths
    ]
    n_batches = (n + batch_size - 1) // batch_size
    print(f"  [fallback] batching into {n_batches} commit(s) of ≤{batch_size} files each")
    for i in range(0, n, batch_size):
        idx = i // batch_size + 1
        batch = ops[i:i + batch_size]
        api.create_commit(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            operations=batch,
            commit_message=f"{label}: batch {idx}/{n_batches} ({len(batch)} files)",
        )
        print(f"  [commit {idx}/{n_batches}] {len(batch)} files")


def push_level(level: int, batch_size: int = 1000) -> int:
    """Push all local files at the given level to HF in one commit (with
    fallback). Returns number of files pushed."""
    relpaths = sorted(_local_files(level))
    if not relpaths:
        return 0
    api = _api()
    _push_robust(
        api=api,
        folder_path=EVAL_RESULTS_DIR,
        relpaths=relpaths,
        path_in_repo=EVAL_RESULTS_HF_PREFIX,
        batch_size=batch_size,
        label=f"push {_level_label(level)}",
    )
    return len(relpaths)


def push_eval_dir(eval_dir: Path) -> int:
    """Upload EVAL_PATTERNS files in `eval_dir` as a single commit. Used by
    `push-dir` for one-off targeted pushes."""
    eval_dir = eval_dir.resolve()
    files = [p for p in eval_dir.iterdir() if p.is_file() and p.name in EVAL_PATTERNS]
    if not files:
        return 0
    rel = _rel_under_eval_results(eval_dir)
    path_in_repo = f"{EVAL_RESULTS_HF_PREFIX}/{rel}"
    api = _api()
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(eval_dir),
        path_in_repo=path_in_repo,
        allow_patterns=list(EVAL_PATTERNS),
        commit_message=f"push {path_in_repo} ({len(files)} files)",
    )
    return len(files)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_push(args: argparse.Namespace) -> None:
    level = _resolve_level(args)
    n = push_level(level, batch_size=args.batch_size)
    if n == 0:
        sys.exit(
            f"No files to push at level {level} ({_level_label(level)}). "
            f"Check {EVAL_RESULTS_DIR}."
        )
    print(f"Pushed {n} file(s) at level {level} ({_level_label(level)}) "
          f"→ {REPO_ID}:{EVAL_RESULTS_HF_PREFIX}/")


def cmd_push_dir(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.is_dir():
        sys.exit(f"Not a directory: {eval_dir}")
    n = push_eval_dir(eval_dir)
    if n == 0:
        sys.exit(f"No matching files in {eval_dir} (looked for: {', '.join(EVAL_PATTERNS)}).")
    rel = _rel_under_eval_results(eval_dir)
    print(f"Pushed {n} file(s) from {eval_dir} → {REPO_ID}:{EVAL_RESULTS_HF_PREFIX}/{rel}/")


def cmd_pull(args: argparse.Namespace) -> None:
    level = _resolve_level(args)
    if args.filter and level < 3:
        sys.exit("--filter requires --fullevals (it narrows the level-3 sweep).")
    api = _api(require_token=False)
    allow = _hf_allow_patterns(level, filter_glob=args.filter)
    target = EVAL_RESULTS_DIR if level >= 2 else SCORES_DIR
    print(f"Pulling {REPO_ID} → {target} "
          f"(level {level}: {_level_label(level)}, filter: {args.filter or '*'})")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(CROSS_ELICIT_ROOT),
        allow_patterns=allow,
        token=api.token,
    )
    print("Done.")


def cmd_verify(args: argparse.Namespace) -> None:
    level = _resolve_level(args)
    api = _api(require_token=args.push)
    local = _local_files(level)
    remote = _remote_files(api, level)
    only_local = sorted(local - remote)
    only_remote = sorted(remote - local)
    label = f"level {level} ({_level_label(level)})"
    print(f"[{label}] local: {len(local):6d}  remote: {len(remote):6d}  "
          f"only-local: {len(only_local)}  only-remote: {len(only_remote)}")

    if args.show:
        for p in only_local[:20]:
            print(f"  L> {p}")
        if len(only_local) > 20:
            print(f"  ... +{len(only_local) - 20} more")
        for p in only_remote[:20]:
            print(f"  R> {p}")
        if len(only_remote) > 20:
            print(f"  ... +{len(only_remote) - 20} more")

    if args.push:
        _push_robust(
            api,
            folder_path=EVAL_RESULTS_DIR,
            relpaths=only_local,
            path_in_repo=EVAL_RESULTS_HF_PREFIX,
            batch_size=args.batch_size,
            label=f"verify-push {label}",
        )


def cmd_backfill(args: argparse.Namespace) -> None:
    level = _resolve_level(args)
    api = _api()
    allow = _hf_allow_patterns(level)
    print(
        f"Backfilling {EVAL_RESULTS_DIR} → {REPO_ID} "
        f"(level {level}: {_level_label(level)}, workers: {args.workers})"
    )
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(CROSS_ELICIT_ROOT),
        allow_patterns=allow,
        num_workers=args.workers,
    )
    print("Backfill done.")


def cmd_create_repo(args: argparse.Namespace) -> None:
    api = _api()
    print(f"Creating {REPO_TYPE} repo {REPO_ID} (private={args.private})...")
    url = api.create_repo(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        private=args.private,
        exist_ok=True,
    )
    print(f"OK: {url}")


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------

def _add_level_flags(sp: argparse.ArgumentParser) -> None:
    """Three-level scope flags. --fullevals wins if both are given."""
    sp.add_argument("--allnumbers", action="store_true",
                    help="level 2: include raw_numbers/**/*.pt as well")
    sp.add_argument("--fullevals", action="store_true",
                    help="level 3: include raw_numbers + full eval dirs "
                         "(supersedes --allnumbers if both are passed)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("push", help="bulk upload at the selected level")
    _add_level_flags(sp)
    sp.add_argument("--batch-size", type=int, default=1000,
                    help="files per commit if single-commit upload fails")
    sp.set_defaults(func=cmd_push)

    sp = sub.add_parser("push-dir", help="upload one eval dir under new_eval_results/")
    sp.add_argument("eval_dir")
    sp.set_defaults(func=cmd_push_dir)

    sp = sub.add_parser("pull", help="download from HF into new_eval_results/")
    _add_level_flags(sp)
    sp.add_argument("--filter",
                    help="(with --fullevals) path-glob under new_eval_results/, "
                         "e.g. 'finetuning/*agreeableness*'")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("verify", help="diff local vs HF at the selected level")
    _add_level_flags(sp)
    sp.add_argument("--push", action="store_true", help="upload anything missing on HF")
    sp.add_argument("--show", action="store_true", help="list a sample of differences")
    sp.add_argument("--batch-size", type=int, default=1000,
                    help="files per commit if single-commit upload fails")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("backfill", help="bulk upload all matching files")
    _add_level_flags(sp)
    sp.add_argument("--workers", type=int, default=8)
    sp.set_defaults(func=cmd_backfill)

    sp = sub.add_parser("create-repo", help="one-time: create HF dataset repo")
    sp.add_argument("--public", dest="private", action="store_false")
    sp.set_defaults(private=True, func=cmd_create_repo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
