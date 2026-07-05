#!/usr/bin/env bash
# generate.sh — run blinded prediction for each hypothesis in an isolated
# headless Claude Code process. Each invocation has its own context; the
# only shared state is the filesystem, which the permission layer + the
# OS-level RESULTS lockout constrain.
#
# Discovery: globs hypotheses/H*.md. Original locked run-set was H1-H6, H8,
# H9 once each plus H7a/H7b triplicates (H7a_r1..r3, H7b_r1..r3); those
# specs are still on disk and will be picked up by the glob. H7a/H7b use
# intentionally underspecified stub prompts and the triplicate runs
# measure run-to-run variance of that underspecification — do NOT
# paraphrase the three copies, or the spread stops being a clean variance
# estimate.
#
# Skip rule: a hypothesis is SKIPPED if predictions/<id>/logitz_plus.csv
# already exists and is newer than hypotheses/<id>.md. Pass --force to
# regenerate. This keeps nightly LW-submission runs cheap.
#
# Usage:  ./scripts/generate.sh                 # all stale hypotheses
#         ./scripts/generate.sh H5 H7a_r2       # a subset
#         ./scripts/generate.sh --force         # regenerate everything
set -euo pipefail
cd "$(dirname "$0")/.."

FORCE=0
HYPS=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *) HYPS+=("$arg") ;;
  esac
done

if [ ${#HYPS[@]} -eq 0 ]; then
  shopt -s nullglob
  for spec in hypotheses/H*.md; do
    base="${spec##*/}"
    HYPS+=("${base%.md}")
  done
  shopt -u nullglob
fi

# Refuse to run if RESULTS is readable by this user — blinding precondition.
if [ -n "$(find ./RESULTS -type f ! -name '.gitkeep' 2>/dev/null)" ]; then
  if [ -r "$(find ./RESULTS -type f ! -name '.gitkeep' | head -1)" ]; then
    echo "ABORT: RESULTS/ contains readable files. Lock it before generating."
    echo "See README 'OS-level blinding'. Generation must run blind."
    exit 1
  fi
fi

for H in "${HYPS[@]}"; do
  SPEC="./hypotheses/${H}.md"
  if [ ! -f "$SPEC" ]; then
    echo "SKIP ${H}: ${SPEC} not found."
    continue
  fi
  if [ ! -s "$SPEC" ]; then
    echo "SKIP ${H}: ${SPEC} is empty — write the spec before running."
    continue
  fi
  OUT="./predictions/${H}/logitz_plus.csv"
  if [ "$FORCE" -eq 0 ] && [ -f "$OUT" ] && [ "$OUT" -nt "$SPEC" ]; then
    echo "SKIP ${H}: predictions up-to-date (use --force to regenerate)."
    continue
  fi
  echo "=== generating ${H} ==="
  mkdir -p "./predictions/${H}"
  claude -p "Run the hypothesis-predictor agent for hypothesis ${H}. \
Read ./BLINDED_PROMPT.md and ./hypotheses/${H}.md. Follow the protocol \
exactly: Turn 1 inspection first, then Turn 2 generation. Write outputs \
ONLY to ./predictions/${H}/. You may not read ./RESULTS or any other \
hypothesis's predictions." \
    --append-system-prompt "$(cat ./BLINDED_PROMPT.md)" \
    > "./logs/${H}.log" 2>&1 || echo "  ${H} exited non-zero — check logs/${H}.log"

  if [ -f "./predictions/${H}/NEEDS_CLARIFICATION.md" ]; then
    echo "  ${H} needs clarification — see predictions/${H}/NEEDS_CLARIFICATION.md"
  fi
done

echo
echo "Done. Review logs/ and any NEEDS_CLARIFICATION.md files."
echo "When predictions look complete, run ./scripts/freeze.sh"
