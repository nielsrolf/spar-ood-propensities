#!/usr/bin/env bash
# pre-commit hook body: fails if any staged path is a forbidden eval data file.
# Eval data lives on HF (jo-chen/cross-elicit-evals), not in git.
set -euo pipefail

forbidden=()
for path in "$@"; do
  case "$path" in
    johannes/cross-elicit/eval_results/*/rows.jsonl \
    | johannes/cross-elicit/eval_results/*/summary.json \
    | johannes/cross-elicit/eval_results/*/coherence_rows.jsonl \
    | johannes/cross-elicit/eval_results/*/coherence_summary.json \
    | johannes/cross-elicit/eval_results/*/judgments.jsonl \
    | johannes/cross-elicit/eval_results/*/matrices.json \
    | johannes/cross-elicit/models/*/code.diff)
      forbidden+=("$path")
      ;;
  esac
done

if [ ${#forbidden[@]} -ne 0 ]; then
  echo "ERROR: refusing to commit cross-elicit eval data / model code.diff files." >&2
  echo "These belong on the HF dataset jo-chen/cross-elicit-evals, not in git." >&2
  echo "Offending paths:" >&2
  for p in "${forbidden[@]}"; do
    echo "  - $p" >&2
  done
  echo "" >&2
  echo "If this is a fresh result: it should already auto-push via" >&2
  echo "  python johannes/cross-elicit/scripts/eval_sync.py push <dir>" >&2
  echo "Then unstage and recommit." >&2
  exit 1
fi
