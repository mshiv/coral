#!/bin/bash
# Feed a large ensemble through a QOS submit limit.
#
# Two separate limits apply and they are often confused:
#
#   MaxArraySize            caps the maximum array task INDEX, not the task count. With
#                           MaxArraySize=1001 the largest legal index is 1000, so
#                           --array=1002-1506 is rejected. run_array.sbatch reads
#                           IDX_OFFSET + SLURM_ARRAY_TASK_ID to address members past that.
#   MaxSubmitJobsPerUser    caps how many jobs are queued at once, counting pending AND
#                           running, with each array element counted individually. The %N
#                           throttle does not help here: it limits what runs, while every
#                           submitted element counts against this the moment it is queued.
#
# So an ensemble larger than the submit limit cannot be queued up front at all, and
# --dependency does not help because dependent jobs are still submitted jobs. This script
# submits one chunk, waits for the queue to drain below a threshold, submits the next.
#
# Run it detached from a login node so it survives your session:
#   cd $SCR/runs/train30m
#   nohup bash $CORAL/workflows/slurm/submit_chunks.sh > submit_chunks.log 2>&1 &
#   tail -f submit_chunks.log
#
# It is safe to interrupt and rerun: --start resumes from any member, and nothing is
# resubmitted unless you ask for it.

set -uo pipefail

ENSEMBLE=${ENSEMBLE:-$PWD}
SCRIPT=${SCRIPT:-run_array.sbatch}
THROTTLE=${THROTTLE:-}            # concurrent RUNNING members per chunk; default unthrottled
CHUNK=${CHUNK:-}                  # members per chunk; default is derived from the limits
START=${START:-1}                 # first member (1-based line in run_dirs.txt)
END=${END:-}                      # last member; default is all of them
POLL=${POLL:-120}                 # seconds between queue checks
DRYRUN=${DRYRUN:-0}

while [ $# -gt 0 ]; do
  case "$1" in
    --start) START=$2; shift 2 ;;
    --end) END=$2; shift 2 ;;
    --chunk) CHUNK=$2; shift 2 ;;
    --throttle) THROTTLE=$2; shift 2 ;;
    --poll) POLL=$2; shift 2 ;;
    --dry-run) DRYRUN=1; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

cd "$ENSEMBLE" || exit 1
[ -s run_dirs.txt ] || { echo "no run_dirs.txt in $ENSEMBLE"; exit 1; }
[ -s "$SCRIPT" ] || { echo "no $SCRIPT in $ENSEMBLE"; exit 1; }
grep -q IDX_OFFSET "$SCRIPT" || {
  echo "FAIL: $SCRIPT has no IDX_OFFSET support, so chunks past the array index limit would"
  echo "      silently run the wrong members. Regenerate the ensemble, or patch the script."
  exit 1; }

N=$(wc -l < run_dirs.txt)
END=${END:-$N}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- discover the limits rather than assuming them
MAXARR=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/{gsub(/ /,"",$2); print $2}')
MAXARR=${MAXARR:-1001}
MAXIDX=$((MAXARR - 1))
QOS=$(sacctmgr -n show assoc where user="$USER" format=QOS%40 2>/dev/null | head -1 | tr -d ' ' | cut -d, -f1)
MAXSUB=$(sacctmgr -n show qos "$QOS" format=MaxSubmitJobsPU 2>/dev/null | tr -d ' ')
[ -z "$MAXSUB" ] && MAXSUB=$(sacctmgr -n show qos "$QOS" format=MaxSubmitJobs 2>/dev/null | tr -d ' ')
if ! [ "$MAXSUB" -eq "$MAXSUB" ] 2>/dev/null; then
  MAXSUB=500
  log "could not read a submit limit from the QOS, assuming $MAXSUB. Override with --chunk."
fi

# leave headroom so an interactive job or a stray sbatch does not trip the limit
HEADROOM=$(( MAXSUB / 10 )); [ "$HEADROOM" -lt 10 ] && HEADROOM=10
CEILING=$(( MAXSUB - HEADROOM ))
if [ -z "$CHUNK" ]; then
  CHUNK=$CEILING
  [ "$CHUNK" -gt "$MAXIDX" ] && CHUNK=$MAXIDX
fi
[ "$CHUNK" -lt 1 ] && { echo "computed chunk size $CHUNK is not usable"; exit 1; }

# Default to unthrottled within a chunk. Throttling only helps when the QOS caps RUNNING jobs
# (MaxJobsPU) or when the CPU budget (MaxTRESPU) would be exceeded. On a QOS that caps only
# submission, a %N throttle just idles slots the scheduler would otherwise fill.
CPUS=$(grep -oE '^#SBATCH --cpus-per-task=([0-9]+)' "$SCRIPT" | grep -oE '[0-9]+$')
CPUS=${CPUS:-8}
MAXCPU=$(sacctmgr -n show qos "$QOS" format=MaxTRESPU%40 2>/dev/null | tr -d ' ' \
         | tr ',' '\n' | grep -oE '^cpu=[0-9]+' | cut -d= -f2)
if [ -n "${MAXCPU:-}" ] && [ "$MAXCPU" -gt 0 ] 2>/dev/null; then
  BYCPU=$(( MAXCPU / CPUS ))
  log_cpu="MaxTRESPU cpu=$MAXCPU at $CPUS cpus per member allows $BYCPU concurrent"
else
  BYCPU=$CHUNK; log_cpu="no CPU cap found on the QOS"
fi
if [ -z "$THROTTLE" ]; then
  THROTTLE=$CHUNK
  [ "$BYCPU" -lt "$THROTTLE" ] && THROTTLE=$BYCPU
fi

TOTAL=$(( END - START + 1 ))
log "ensemble $ENSEMBLE"
log "members $START..$END of $N   script $SCRIPT"
log "QOS $QOS: MaxSubmit $MAXSUB, working ceiling $CEILING (headroom $HEADROOM)"
log "MaxArraySize $MAXARR, largest legal index $MAXIDX"
log "$log_cpu"
log "chunk $CHUNK members, throttle %$THROTTLE running, poll ${POLL}s"
log "$(( (TOTAL + CHUNK - 1) / CHUNK )) chunks to submit"

# count my queued jobs; -r expands array elements, which is how the limit counts them
mine() { squeue -u "$USER" -h -r -t PD,R 2>/dev/null | wc -l | tr -d ' '; }

pos=$START
chunk_no=0
while [ "$pos" -le "$END" ]; do
  left=$(( END - pos + 1 ))
  take=$CHUNK; [ "$left" -lt "$take" ] && take=$left
  offset=$(( pos - 1 ))

  # wait for room. `take` new elements must fit under the ceiling alongside what is queued.
  while :; do
    q=$(mine)
    room=$(( CEILING - q ))
    [ "$room" -ge "$take" ] && break
    log "queued $q, need room for $take, have $room. waiting ${POLL}s"
    sleep "$POLL"
  done

  chunk_no=$(( chunk_no + 1 ))
  if [ "$offset" -eq 0 ]; then
    CMD=(sbatch --array=1-"$take"%"$THROTTLE" "$SCRIPT")
  else
    CMD=(sbatch --export=ALL,IDX_OFFSET="$offset" --array=1-"$take"%"$THROTTLE" "$SCRIPT")
  fi
  log "chunk $chunk_no: members $pos..$(( pos + take - 1 ))  ->  ${CMD[*]}"
  if [ "$DRYRUN" -eq 1 ]; then
    log "  (dry run, not submitted)"
  else
    OUT=$("${CMD[@]}" 2>&1)
    RC=$?
    log "  $OUT"
    if [ "$RC" -ne 0 ]; then
      log "  submission failed, backing off ${POLL}s and retrying this chunk"
      sleep "$POLL"; continue
    fi
  fi
  pos=$(( pos + take ))
done

log "all $TOTAL members submitted"
log "progress:  ls $ENSEMBLE/*/results_*/*.max 2>/dev/null | wc -l    # climbs to $N"
