#!/usr/bin/env bash
# Full pipeline: circuits -> mutants -> simulator check -> 10-run ablation -> analysis.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY src/circuits.py
$PY src/mutants.py
$PY src/verify_sim.py
$PY src/run_ablation.py --runs 10
$PY src/analyze.py > /dev/null
echo "Done. Tables and figures are in results/ (summary: results/RESULTS.md)."
