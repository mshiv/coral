#!/bin/bash
# Bundle the small evaluation products and original selection reports for transfer off HPC.
set -euo pipefail

CORAL_ROOT=${CORAL_ROOT:-/storage/home/hcoda1/6/smurugan9/data/dev/coral}
if [ -n "${EMULATOR_REPORT_ROOT:-}" ]; then
  REPORT_ROOT="$EMULATOR_REPORT_ROOT"
elif [ -d /storage/project/r-arobel3-0/smurugan9/dev/coral/reports/emulator/pp4_e01/local_response ]; then
  REPORT_ROOT=/storage/project/r-arobel3-0/smurugan9/dev/coral/reports/emulator/pp4_e01
else
  REPORT_ROOT="$CORAL_ROOT/reports/emulator/pp4_e01"
fi
OUT="$REPORT_ROOT/chapter4_local_response_bundle.tar.gz"

test -d "$REPORT_ROOT/local_response" || {
  echo "missing $REPORT_ROOT/local_response; wait for the evaluation jobs" >&2
  exit 1
}

tar -czf "$OUT" \
  -C "$REPORT_ROOT" \
  local_response \
  emulator_unet_perkind.report.json \
  emulator_unet_combos.report.json \
  emulator_unet_siting_targeted.report.json \
  emulator_unet_holdout_kind_floodwall.report.json \
  emulator_unet_holdout_kind_living_shoreline.report.json \
  emulator_unet_holdout_slr_High2100.report.json

echo "$OUT"
du -h "$OUT"
