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
MINCHUNK=${MINCHUNK:-25}          # do not bother submitting fewer than this at a time
QOS=${QOS:-}                      # QOS to read limits from; default is the job's own
DRYRUN=${DRYRUN:-0}
RERUN=${RERUN:-0}         # rebuild the member list from those still lacking a .max
WALLTIME=${WALLTIME:-}    # override the launcher's -t, e.g. after a run of TIMEOUTs

while [ $# -gt 0 ]; do
  case "$1" in
    --start) START=$2; shift 2 ;;
    --end) END=$2; shift 2 ;;
    --chunk) CHUNK=$2; shift 2 ;;
    --throttle) THROTTLE=$2; shift 2 ;;
    --poll) POLL=$2; shift 2 ;;
    --min-chunk) MINCHUNK=$2; shift 2 ;;
    --qos) QOS=$2; shift 2 ;;
    --time) WALLTIME=$2; shift 2 ;;
    --dry-run) DRYRUN=1; shift ;;
    --rerun-missing) RERUN=1; shift ;;
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

# Rerun mode: build a member list from the run dirs that have no .max, so a partially
# completed ensemble can be finished without resubmitting what already succeeded. Members
# are addressed by line number, so this needs its own list file and its own launcher
# pointing at that list rather than a filter applied to the original.
if [ "$RERUN" -eq 1 ]; then
  MISS=run_dirs_missing.txt
  : > "$MISS" 2>/dev/null || { echo "FAIL: cannot write $MISS in $PWD (quota? permissions?)"; exit 1; }
  NTOT=0; NDONE=0
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    NTOT=$(( NTOT + 1 ))
    if ls "$d"/results_*/*.max >/dev/null 2>&1; then
      NDONE=$(( NDONE + 1 ))
    else
      echo "$d" >> "$MISS" || { echo "FAIL: write to $MISS failed partway (quota?)."; exit 1; }
    fi
  done < run_dirs.txt
  NM=$(wc -l < "$MISS")
  # A partial write leaves a short list, which would silently submit only some of the missing
  # members and look like success. present + missing must account for every planned member.
  if [ "$(( NDONE + NM ))" -ne "$NTOT" ]; then
    echo "FAIL: $NDONE finished + $NM missing = $(( NDONE + NM )), but run_dirs.txt has $NTOT."
    echo "      The missing list is incomplete, most likely a truncated write. Free space and"
    echo "      rerun; do not submit from this list."
    exit 1
  fi
  echo "rerun mode: $NDONE of $NTOT members already have a .max"
  if [ "$NM" -eq 0 ]; then
    echo "every member already has a .max, nothing to rerun"; exit 0
  fi
  SCRIPT_RERUN=run_array_missing.sbatch
  sed "s|/run_dirs\.txt\"|/$MISS\"|" "$SCRIPT" > "$SCRIPT_RERUN"
  grep -q "$MISS" "$SCRIPT_RERUN" || {
    echo "FAIL: could not point $SCRIPT_RERUN at $MISS; check the run_dirs path in $SCRIPT"
    exit 1; }
  SCRIPT=$SCRIPT_RERUN
  echo "rerun mode: $NM lack a .max -> $MISS"
fi

N=$(wc -l < "$( [ "$RERUN" -eq 1 ] && echo run_dirs_missing.txt || echo run_dirs.txt )")
END=${END:-$N}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- discover the limits rather than assuming them
MAXARR=$(scontrol show config 2>/dev/null | awk -F= '/MaxArraySize/{gsub(/ /,"",$2); print $2}')
MAXARR=${MAXARR:-1001}
MAXIDX=$((MAXARR - 1))
# Read limits from the QOS the job will actually run under, not just the first one listed.
# An association commonly lists several (on PACE, "embers,inferno"), and picking alphabetically
# lands on the preemptible backfill tier with a far smaller submit limit, which silently
# throttles the whole ensemble.
if [ -z "$QOS" ]; then
  QOS=$(grep -oE '^#SBATCH[[:space:]]+(--qos|-q)[=[:space:]]+[A-Za-z0-9_-]+' "$SCRIPT" \
        | grep -oE '[A-Za-z0-9_-]+$' | tail -1)
fi
if [ -z "$QOS" ]; then                       # the association's DEFAULT QOS is what a job gets
  QOS=$(sacctmgr -n show assoc where user="$USER" format=DefaultQOS%30 2>/dev/null \
        | tr -d ' ' | grep -v '^$' | head -1)
fi
if [ -z "$QOS" ]; then
  QOS=$(sacctmgr -n show assoc where user="$USER" format=QOS%60 2>/dev/null \
        | tr -d ' ' | grep -v '^$' | head -1 | cut -d, -f1)
  log "no default QOS found; falling back to $QOS. Pass --qos if the limits below look wrong."
fi
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
[ -n "$WALLTIME" ] && log "walltime overridden to $WALLTIME"
log "$(( (TOTAL + CHUNK - 1) / CHUNK )) chunks to submit"
if [ "$(( (TOTAL + CHUNK - 1) / CHUNK ))" -gt 10 ]; then
  log "WARNING: that is a lot of chunks for $TOTAL members. If the QOS above is not the one"
  log "         your jobs run under, rerun with --qos <name>. Check with:"
  log "         sacctmgr -n show assoc where user=\$USER format=QOS%60,DefaultQOS%20"
fi

# count my queued jobs; -r expands array elements, which is how the limit counts them
mine() { squeue -u "$USER" -h -r -t PD,R 2>/dev/null | wc -l | tr -d ' '; }

pos=$START
chunk_no=0
while [ "$pos" -le "$END" ]; do
  left=$(( END - pos + 1 ))
  take=$CHUNK; [ "$left" -lt "$take" ] && take=$left
  offset=$(( pos - 1 ))

  # Wait for room, then take as much of the chunk as fits. Demanding room for a whole chunk
  # deadlocks whenever something else occupies part of the queue indefinitely: the ceiling is
  # never fully free, so nothing is ever submitted. Shrinking to the available room instead
  # keeps progress going, at the cost of more, smaller submissions.
  while :; do
    q=$(mine)
    room=$(( CEILING - q ))
    if [ "$room" -ge "$take" ]; then
      break
    elif [ "$room" -ge "$MINCHUNK" ]; then
      log "queued $q, room for $room of $take. submitting $room now, the rest follows"
      take=$room
      break
    fi
    log "queued $q, room $room is below the $MINCHUNK minimum. waiting ${POLL}s"
    sleep "$POLL"
  done

  chunk_no=$(( chunk_no + 1 ))
  CMD=(sbatch)
  [ -n "$WALLTIME" ] && CMD+=(-t "$WALLTIME")
  [ "$offset" -ne 0 ] && CMD+=(--export=ALL,IDX_OFFSET="$offset")
  CMD+=(--array=1-"$take"%"$THROTTLE" "$SCRIPT")
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
