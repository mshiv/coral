#!/usr/bin/env bash
# Existing, completed paired runs only: replot, export arrays, verify old metrics.
# Usage: bash workflows/build_chapter_diagnostics.sh SCRATCH_ROOT NEW_OUTPUT_DIR
set -euo pipefail
scratch_root="${1:?Pass the coral scratch root}"
output_root="${2:?Pass a new output directory}"
if [[ -e "$output_root" ]]; then
  echo "Output already exists: $output_root; choose a new directory" >&2
  exit 1
fi
native_root="$scratch_root/runs/intervention_dynamics_4m_v1"
regional_root="$scratch_root/runs/intervention_regional_30m_v1"
native_args=()
for kind in depave living_shoreline marsh_restoration road_raise; do
  native_args+=(--pair "native4m_$kind" "$native_root/living_shoreline/control" "$native_root/$kind/intervention")
done
python -m coral.analysis.replot_chapter_pairs \
  --previous-reports reports/adapt/paired --out-dir "$output_root/native" \
  --cell-m 4 --waterline 1.114 --error-limit 0.1 "${native_args[@]}"
regional_args=()
for kind in living_shoreline living_shoreline_fractional marsh_restoration road_raise; do
  regional_args+=(--pair "regional30m_$kind" "$regional_root/living_shoreline_fractional/control" "$regional_root/$kind/intervention")
done
regional_args+=(--pair regional30m_floodwall "$scratch_root/runs/wall_redistribution_30m/control" "$scratch_root/runs/wall_redistribution_30m/wall")
python -m coral.analysis.replot_chapter_pairs \
  --previous-reports reports/adapt/paired --out-dir "$output_root/regional" \
  --cell-m 30 --waterline 1.114 --error-limit 0.1 "${regional_args[@]}"
