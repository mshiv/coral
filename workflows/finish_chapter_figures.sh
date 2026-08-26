#!/usr/bin/env bash
# Postprocessing only. Run from CORAL with its Python environment active.
set -euo pipefail
: "${CORAL:?Export CORAL first}"
: "${SCR:?Export SCR first}"
cd "$CORAL"
OUT="${1:?Supply a new output directory}"
if [[ -e "$OUT" ]]; then
  echo "Refusing existing output directory: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
export MPLBACKEND=Agg
export MPLCONFIGDIR="$OUT/matplotlib"
STATUS="$OUT/status.tsv"
run_step() {
  local label="$1"
  shift
  if "$@" >"$OUT/$label.log" 2>&1; then
    printf 'OK\t%s\n' "$label" | tee -a "$STATUS"
  else
    printf 'FAILED\t%s\tsee %s.log\n' "$label" "$label" | tee -a "$STATUS"
  fi
}
run_step physics_export python -m coral.analysis.chapter_physics_capsule \
  --manifest "$CORAL/reports/chapter4_hpc_v3/bundle_manifest.json" \
  --runs-root "$SCR/runs" --out "$OUT/physics"
if [[ -f "$OUT/physics/baseline_event.npz" ]]; then
  run_step event_evolution python -m coral.analysis.chapter_physics_capsule \
    --plot "$OUT/physics/baseline_event.npz" --out "$OUT/coral_event_evolution.png"
fi
REPORTS="$CORAL/reports/emulator/pp4_e01"
for stem in emulator_unet_perkind emulator_unet_combos emulator_unet_siting_targeted \
  emulator_unet_holdout_kind_floodwall emulator_unet_holdout_kind_living_shoreline \
  emulator_unet_holdout_slr_High2100; do
  if [[ ! -f "$REPORTS/$stem.pt" || ! -f "$REPORTS/$stem.report.json" ]]; then
    printf 'MISSING\t%s checkpoint/report\n' "$stem" | tee -a "$STATUS"
    continue
  fi
  run_step "$stem" python -m coral.analysis.emulator_vs_physics \
    --ckpt "$REPORTS/$stem.pt" --report "$REPORTS/$stem.report.json" \
    --ens "$SCR/runs/pp4_e01" --dem "$SCR/runs/pp4_base/SUB_DEM_pp4_spinup.asc" \
    --waterline 1.114 --publication --out "$OUT/$stem.png" --export-npz "$OUT/$stem.npz"
done
if [[ -f "$OUT/emulator_unet_perkind.npz" && -f "$OUT/emulator_unet_combos.npz" && \
      -f "$OUT/emulator_unet_holdout_kind_floodwall.npz" && -f "$OUT/emulator_unet_holdout_slr_High2100.npz" ]]; then
  run_step compact_holdouts python -m coral.analysis.emulator_holdout_composite \
    --case "Per-kind=$OUT/emulator_unet_perkind.npz" \
    --case "Combinations=$OUT/emulator_unet_combos.npz" \
    --case "Unseen floodwalls=$OUT/emulator_unet_holdout_kind_floodwall.npz" \
    --case "Unseen high SLR=$OUT/emulator_unet_holdout_slr_High2100.npz" \
    --out "$OUT/coral_emulator_compact.png"
fi
printf '\nOutputs: %s\nReview status.tsv; missing source identities are not silently substituted.\n' "$OUT"
if grep -Eq '^(FAILED|MISSING)' "$STATUS"; then exit 1; fi
