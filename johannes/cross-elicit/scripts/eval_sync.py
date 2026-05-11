"""
Sync eval result files to/from the HF dataset repo `jo-chen/cross-elicit-evals`.

The HF dataset mirrors the on-disk layout. Two kinds of artifacts are tracked:

  * Raw eval data — files under cross-elicit/eval_results/ (mapped to HF root).
    Only these filenames are uploaded:
      rows.jsonl, summary.json, coherence_rows.jsonl, coherence_summary.json,
      judgments.jsonl, matrices.json
  * Summary results — files under cross-elicit/results/ (mapped to HF results/).
    Only these glob patterns are uploaded:
      finetuned_scores_*.json, sysprompts_scores_*.json, eval-orthogonality_scores.json

Subcommands:
  push          <eval_dir>     Upload one finished eval dir (path under eval_results/).
  push-results                 Upload all summary score JSONs from results/.
  pull          [--filter G]   Snapshot-download from HF. Pulls both raw eval data
                               (into eval_results/) and summaries (into results/).
                               --results-only skips raw data.
                               --filter only applies to raw data.
  verify        [--push]       List files missing on either side (raw + summaries).
                               With --push, first attempts a single-commit
                               upload_folder; on non-rate-limit failure, falls
                               back to explicit create_commit batches of
                               --batch-size files each (default 1000).
                               --results-only restricts to summaries.
  backfill                     Bulk-upload all matching files via
                               upload_large_folder (batched, resumable, parallel).
                               --results-only restricts to summaries.
  create-repo                  One-time: create the private dataset repo on HF.
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
EVAL_RESULTS_DIR = CROSS_ELICIT_ROOT / "eval_results"
RESULTS_DIR = CROSS_ELICIT_ROOT / "results"

load_dotenv(JOHANNES_ROOT / ".env")

REPO_ID = "jo-chen/cross-elicit-evals"
REPO_TYPE = "dataset"

PATTERNS = (
    "rows.jsonl",
    "summary.json",
    "coherence_rows.jsonl",
    "coherence_summary.json",
    "judgments.jsonl",
    "matrices.json",
)

RESULTS_HF_PREFIX = "results"
RESULTS_PATTERNS = (
    "finetuned_scores_*.json",
    "sysprompts_scores_*.json",
    "eval-orthogonality_scores.json",
)


def _api() -> HfApi:
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN not set (check johannes/.env).")
    return HfApi(token=token)


def _rel_under_eval_results(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(EVAL_RESULTS_DIR))
    except ValueError:
        sys.exit(f"{path} is not under {EVAL_RESULTS_DIR}.")


def _matching_files(eval_dir: Path) -> list[Path]:
    return [p for p in eval_dir.iterdir() if p.is_file() and p.name in PATTERNS]


def push_eval_dir(eval_dir: Path) -> int:
    """Upload matching files in eval_dir to HF as a single commit. Returns
    number uploaded. Raises on failure.

    Uses upload_folder so the ~6 PATTERNS files in one eval dir cost one
    commit, not six (HF caps datasets at 128 commits/hour).
    """
    eval_dir = eval_dir.resolve()
    files = _matching_files(eval_dir)
    if not files:
        return 0
    rel = _rel_under_eval_results(eval_dir)
    # allow_patterns are matched relative to folder_path with fnmatch; using
    # the bare filenames is enough since we point folder_path at eval_dir.
    api = _api()
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(eval_dir),
        path_in_repo=rel,
        allow_patterns=list(PATTERNS),
        commit_message=f"push {rel} ({len(files)} files)",
    )
    return len(files)


def _local_results_files() -> set[str]:
    """Basenames of summary files present in cross-elicit/results/."""
    out: set[str] = set()
    if not RESULTS_DIR.exists():
        return out
    for pat in RESULTS_PATTERNS:
        for p in RESULTS_DIR.glob(pat):
            if p.is_file():
                out.add(p.name)
    return out


def _remote_results_files(api: HfApi) -> set[str]:
    """Basenames of summary files present under HF results/ prefix."""
    prefix = f"{RESULTS_HF_PREFIX}/"
    out: set[str] = set()
    for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE):
        if not f.startswith(prefix):
            continue
        base = f[len(prefix):]
        if "/" in base:
            continue
        if any(fnmatch(base, pat) for pat in RESULTS_PATTERNS):
            out.add(base)
    return out


def push_results() -> int:
    """Upload all matching summary files in results/ to HF as one commit.
    Returns count."""
    if not RESULTS_DIR.exists():
        return 0
    files: list[Path] = []
    for pat in RESULTS_PATTERNS:
        files.extend(p for p in RESULTS_DIR.glob(pat) if p.is_file())
    if not files:
        return 0
    api = _api()
    api.upload_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(RESULTS_DIR),
        path_in_repo=RESULTS_HF_PREFIX,
        allow_patterns=list(RESULTS_PATTERNS),
        commit_message=f"push results ({len(files)} files)",
    )
    return len(files)


def push_or_mark_pending(eval_dir: str | Path) -> None:
    """Best-effort auto-push for orchestrators. On any failure, drops a
    `.push_pending` marker in `eval_dir` so `verify --push-pending` can
    retry later. Never raises."""
    eval_dir = Path(eval_dir)
    if not eval_dir.is_dir():
        return
    try:
        n = push_eval_dir(eval_dir)
        marker = eval_dir / ".push_pending"
        if marker.exists():
            marker.unlink()
        if n:
            print(f"  [HF] pushed {n} file(s) from {eval_dir.name}")
    except Exception as e:
        marker = eval_dir / ".push_pending"
        try:
            marker.write_text(f"{type(e).__name__}: {e}\n")
        except Exception:
            pass
        print(f"  [HF] push failed for {eval_dir.name}: {e} (marked .push_pending)")


def cmd_push(args: argparse.Namespace) -> None:
    eval_dir = Path(args.eval_dir).resolve()
    if not eval_dir.is_dir():
        sys.exit(f"Not a directory: {eval_dir}")
    n = push_eval_dir(eval_dir)
    if n == 0:
        sys.exit(f"No matching files in {eval_dir} (looked for: {', '.join(PATTERNS)}).")
    rel = _rel_under_eval_results(eval_dir)
    print(f"Pushed {n} file(s) from {eval_dir} → {REPO_ID}:{rel}/")


def cmd_pull(args: argparse.Namespace) -> None:
    api = _api()
    if not args.results_only:
        allow = [f"*{pat}" for pat in PATTERNS]
        if args.filter:
            allow = [f"{args.filter}/{p}" for p in allow]
        print(f"Pulling eval data from {REPO_ID} → {EVAL_RESULTS_DIR} (filter: {args.filter or '*'})")
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            local_dir=str(EVAL_RESULTS_DIR),
            allow_patterns=allow,
            token=api.token,
        )
    results_allow = [f"{RESULTS_HF_PREFIX}/{p}" for p in RESULTS_PATTERNS]
    print(f"Pulling summaries from {REPO_ID} → {RESULTS_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=str(CROSS_ELICIT_ROOT),
        allow_patterns=results_allow,
        token=api.token,
    )
    print("Done.")


def _local_files() -> set[str]:
    out: set[str] = set()
    if not EVAL_RESULTS_DIR.exists():
        return out
    for root, _dirs, files in os.walk(EVAL_RESULTS_DIR):
        for f in files:
            if f in PATTERNS:
                p = Path(root) / f
                out.add(str(p.relative_to(EVAL_RESULTS_DIR)))
    return out


def _remote_files(api: HfApi) -> set[str]:
    return {
        f for f in api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)
        if Path(f).name in PATTERNS
    }


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

    First attempts a single `upload_folder` commit (cheapest in commit budget).
    If that fails for any non-rate-limit reason — e.g. HF rejects the request
    because the file count or payload exceeded a per-commit cap — falls back
    to explicit `create_commit` calls of ~batch_size files each.

    A 429 is re-raised immediately, since splitting into more commits would
    just hit the rate limit harder.
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

    # Fallback: explicit batched create_commit. CommitOperationAdd takes the
    # full in-repo path so we don't need path_in_repo on create_commit itself.
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


def cmd_verify(args: argparse.Namespace) -> None:
    api = _api()

    only_local: list[str] = []
    only_remote: list[str] = []
    if not args.results_only:
        local = _local_files()
        remote = _remote_files(api)
        only_local = sorted(local - remote)
        only_remote = sorted(remote - local)
        print(f"[eval data] local: {len(local):6d}  remote: {len(remote):6d}  "
              f"only-local: {len(only_local)}  only-remote: {len(only_remote)}")

    local_r = _local_results_files()
    remote_r = _remote_results_files(api)
    only_local_r = sorted(local_r - remote_r)
    only_remote_r = sorted(remote_r - local_r)
    print(f"[summaries] local: {len(local_r):6d}  remote: {len(remote_r):6d}  "
          f"only-local: {len(only_local_r)}  only-remote: {len(only_remote_r)}")

    if args.show:
        for p in only_local[:20]:
            print(f"  L> {p}")
        if len(only_local) > 20:
            print(f"  ... +{len(only_local) - 20} more")
        for p in only_remote[:20]:
            print(f"  R> {p}")
        if len(only_remote) > 20:
            print(f"  ... +{len(only_remote) - 20} more")
        for p in only_local_r[:20]:
            print(f"  L> results/{p}")
        if len(only_local_r) > 20:
            print(f"  ... +{len(only_local_r) - 20} more")
        for p in only_remote_r[:20]:
            print(f"  R> results/{p}")
        if len(only_remote_r) > 20:
            print(f"  ... +{len(only_remote_r) - 20} more")

    if args.push:
        _push_robust(
            api,
            folder_path=EVAL_RESULTS_DIR,
            relpaths=only_local,
            path_in_repo=None,
            batch_size=args.batch_size,
            label="verify-push eval-data",
        )
        _push_robust(
            api,
            folder_path=RESULTS_DIR,
            relpaths=only_local_r,
            path_in_repo=RESULTS_HF_PREFIX,
            batch_size=args.batch_size,
            label="verify-push summaries",
        )

    if args.push_pending:
        markers = list(EVAL_RESULTS_DIR.rglob(".push_pending"))
        if not markers:
            print("\nNo .push_pending markers found.")
        else:
            print(f"\nRetrying {len(markers)} pending dir(s)...")
            for m in markers:
                push_or_mark_pending(m.parent)


def cmd_backfill(args: argparse.Namespace) -> None:
    api = _api()
    if not args.results_only:
        allow = [f"**/{pat}" for pat in PATTERNS]
        print(
            f"Backfilling {EVAL_RESULTS_DIR} → {REPO_ID} "
            f"(patterns: {allow}, workers: {args.workers})"
        )
        api.upload_large_folder(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            folder_path=str(EVAL_RESULTS_DIR),
            allow_patterns=allow,
            num_workers=args.workers,
        )
    results_allow = [f"{RESULTS_HF_PREFIX}/{p}" for p in RESULTS_PATTERNS]
    print(
        f"Backfilling {RESULTS_DIR} → {REPO_ID}/{RESULTS_HF_PREFIX}/ "
        f"(patterns: {results_allow}, workers: {args.workers})"
    )
    api.upload_large_folder(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        folder_path=str(CROSS_ELICIT_ROOT),
        allow_patterns=results_allow,
        num_workers=args.workers,
    )
    print("Backfill done.")


def cmd_push_results(args: argparse.Namespace) -> None:
    n = push_results()
    if n == 0:
        sys.exit(
            f"No matching summary files in {RESULTS_DIR} "
            f"(looked for: {', '.join(RESULTS_PATTERNS)})."
        )
    print(f"Pushed {n} summary file(s) → {REPO_ID}:{RESULTS_HF_PREFIX}/")


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("push", help="upload one eval dir")
    sp.add_argument("eval_dir")
    sp.set_defaults(func=cmd_push)

    sp = sub.add_parser("push-results", help="upload summary score JSONs from results/")
    sp.set_defaults(func=cmd_push_results)

    sp = sub.add_parser("pull", help="download eval data + summaries from HF")
    sp.add_argument("--filter", help="path-glob prefix for raw eval data, e.g. 'finetuning/*agreeableness*'")
    sp.add_argument("--results-only", action="store_true",
                    help="skip raw eval data; only fetch summary score JSONs")
    sp.set_defaults(func=cmd_pull)

    sp = sub.add_parser("verify", help="diff local vs HF")
    sp.add_argument("--push", action="store_true", help="upload anything missing on HF")
    sp.add_argument("--push-pending", action="store_true",
                    help="retry dirs with .push_pending markers")
    sp.add_argument("--show", action="store_true", help="list a sample of differences")
    sp.add_argument("--results-only", action="store_true",
                    help="restrict diff/push to summary score JSONs only")
    sp.add_argument("--batch-size", type=int, default=1000,
                    help="files per commit when single-commit upload_folder "
                         "fails and we fall back to batched create_commit "
                         "(default: 1000)")
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("backfill", help="bulk upload all matching files")
    sp.add_argument("--workers", type=int, default=8)
    sp.add_argument("--results-only", action="store_true",
                    help="skip raw eval data; only backfill summary score JSONs")
    sp.set_defaults(func=cmd_backfill)

    sp = sub.add_parser("create-repo", help="one-time: create HF dataset repo")
    sp.add_argument("--public", dest="private", action="store_false")
    sp.set_defaults(private=True, func=cmd_create_repo)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
