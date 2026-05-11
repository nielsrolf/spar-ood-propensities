#!/bin/bash
# End-to-end driver for the cross-elicit human audit.
#
#   ./run.sh prep        # aggregate + build_configs + sample + merge_priors
#   ./run.sh annotate    # serve annotation GUI
#   ./run.sh judge       # run alt judges over all configs
#   ./run.sh analyze     # compare alt judges to human labels
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

case "${1:-prep}" in
  prep)
    python3 aggregate.py
    python3 build_configs.py
    python3 sample.py "${@:2}"
    python3 merge_priors.py
    ;;
  annotate)
    cd "$HERE/../propensity_audit"
    python3 annotate.py --configs-dir "$HERE/configs" --port "${PORT:-8781}"
    ;;
  judge)
    python3 run_alt_judges.py "${@:2}"
    ;;
  analyze)
    python3 analyze.py "${@:2}"
    ;;
  *)
    echo "Usage: $0 {prep|annotate|judge|analyze}" >&2
    exit 1
    ;;
esac
