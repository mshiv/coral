#!/usr/bin/env bash
# Activate the coral environment before calling. No sbatch or simulation here.
set -euo pipefail
python -m coral.analysis.chapter_figure_bundle "$@"
