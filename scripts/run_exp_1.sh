#!/usr/bin/env bash
set -euo pipefail

# Defaults
NUM_WORKERS=8
SEED_START=101
SEED_STOP=117        # exclusive, to mirror Python's range()
GROUP_SIZE=0         # 0 => computed so num_groups <= NUM_WORKERS
ONLY_GROUP=""        # if set, runs exactly this group index (foreground)
USE_CACHE=1          # pass --use-cached-models by default

# --- arg parse ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--num-workers) NUM_WORKERS="$2"; shift 2 ;;
    -s|--seed-start)  SEED_START="$2";  shift 2 ;;
    -e|--seed-stop)   SEED_STOP="$2";   shift 2 ;;
    -g|--group-size)  GROUP_SIZE="$2";  shift 2 ;;
    --only-group)     ONLY_GROUP="$2";  shift 2 ;;
    --no-cache)       USE_CACHE=0;      shift ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

  -n, --num-workers N     number of parallel groups to run (default: 4)
  -s, --seed-start S      inclusive seed start (default: 101)
  -e, --seed-stop E       exclusive seed stop   (default: 111)
  -g, --group-size G      seeds per group (default: computed from num-workers)
      --only-group IDX    run only this group (foreground; no parallel fan-out)
      --no-cache          omit --use-cached-models when calling Python
EOF
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

TOTAL=$(( SEED_STOP - SEED_START ))
if (( TOTAL <= 0 )); then
  echo "seed-stop must be > seed-start" >&2
  exit 1
fi

# Compute group size if not provided so that num_groups <= NUM_WORKERS.
if (( GROUP_SIZE <= 0 )); then
  GROUP_SIZE=$(( (TOTAL + NUM_WORKERS - 1) / NUM_WORKERS ))  # ceiling div
fi
NUM_GROUPS=$(( (TOTAL + GROUP_SIZE - 1) / GROUP_SIZE ))      # ceiling div

# cd to repo root (directory containing src/) so imports stay clean
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
if [[ -d "$SCRIPT_DIR/src" ]]; then
  cd "$SCRIPT_DIR"
elif [[ -d "$SCRIPT_DIR/../src" ]]; then
  cd "$SCRIPT_DIR/.."
else
  echo "Cannot find src/ relative to $SCRIPT_DIR" >&2
  exit 1
fi

mkdir -p logs
TS="$(date +%Y%m%d_%H%M%S)"

PY_ARGS_BASE=( -m src.experiment_superposition --seed-start "$SEED_START" --seed-stop "$SEED_STOP" --group-size "$GROUP_SIZE" )
if (( USE_CACHE == 1 )); then PY_ARGS_BASE+=( --use-cached-models ); fi

# graceful shutdown: forward INT/TERM to children
trap 'echo "Stopping..."; pkill -P $$ || true; wait || true; exit 130' INT TERM

if [[ -n "$ONLY_GROUP" ]]; then
  idx="$ONLY_GROUP"
  if (( idx < 0 || idx >= NUM_GROUPS )); then
    echo "--only-group $idx out of range [0, $((NUM_GROUPS-1))]" >&2
    exit 2
  fi
  LOG="logs/run_${TS}_group_${idx}.log"
  echo "Running group $idx (foreground). Logging to $LOG"
  python "${PY_ARGS_BASE[@]}" --group-idx "$idx" |& tee "$LOG"
  exit $?
fi

echo "Launching $NUM_GROUPS group(s) (up to $NUM_WORKERS in parallel). Seeds: [$SEED_START, $SEED_STOP)  Group size: $GROUP_SIZE"

pids=()
fail=0

launch_group () {
  local idx="$1"
  local LOG="logs/run_${TS}_group_${idx}.log"
  echo "  -> group $idx  log: $LOG"
  python "${PY_ARGS_BASE[@]}" --group-idx "$idx" &> "$LOG" &
  pids+=($!)
}

# Start groups while capping concurrency at NUM_WORKERS
for (( idx=0; idx<NUM_GROUPS; idx++ )); do
  # Wait while we have >= NUM_WORKERS running jobs
  while (( $(jobs -rp | wc -l) >= NUM_WORKERS )); do
    sleep 0.2
  done
  launch_group "$idx"
done

# Wait for all and capture failures
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    fail=1
  fi
done

exit $fail
