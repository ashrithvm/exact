#!/bin/bash
# ============================================================
# run_experiments.sh
# Runs all 5 fault-tolerance experiments, 9 trials each.
# Trials within a batch run in parallel; batches are sequential.
#
# Usage:
#   bash scripts/run_experiments.sh [OPTIONS]
#
# Options:
#   --trials     <int>   Trials per experiment      (default: 9)
#   --parallel   <int>   Max simultaneous runs      (default: 3)
#   --wallclock  <int>   Seconds per run            (default: 600)
#   --skip_build         Skip cmake/make
#   -h, --help
#
# Output layout:
#   test_output/experiments/
#     clean/trial_1 ... trial_9
#     dropout_1/trial_1 ... trial_9
#     dropout_3/trial_1 ... trial_9
#     graceful_1/trial_1 ... trial_9
#     graceful_3/trial_1 ... trial_9
# ============================================================

set -e

# ---- Defaults -----------------------------------------------
TRIALS=9
PARALLEL=3
WALLCLOCK=600
SKIP_BUILD=false

# ---- Parse args ---------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trials)    TRIALS="$2";   shift 2 ;;
        --parallel)  PARALLEL="$2"; shift 2 ;;
        --wallclock) WALLCLOCK="$2"; shift 2 ;;
        --skip_build) SKIP_BUILD=true; shift ;;
        -h|--help)
            sed -n '/^# Usage/,/^# ===/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# ---- Paths --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXACT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
FT_SCRIPT="$SCRIPT_DIR/base_run/p2p_fault_tolerance.sh"
BASE_OUTPUT="$EXACT_DIR/test_output/experiments"

mkdir -p "$BASE_OUTPUT"

# ---- Build once up front ------------------------------------
BUILD_FLAG=""
if [ "$SKIP_BUILD" = true ]; then
    BUILD_FLAG="--skip_build"
else
    echo "=== Building once before experiments start ===================="
    cd "$EXACT_DIR/build_p2p"
    make -j4 examm_mpi
    echo "=== Build complete ============================================"
    BUILD_FLAG="--skip_build"   # skip on all subsequent calls
fi

# ---- Experiment definitions ---------------------------------
# Format: "name|extra_args"
declare -a EXPERIMENTS=(
    "clean|--no_dropout --no_shutdown"
    "dropout_1|--dropout_ranks \"1\" --dropout_after 180 --recovery_after 60 --no_shutdown"
    "dropout_3|--dropout_ranks \"1 2 3\" --dropout_after 180 --recovery_after 60 --no_shutdown"
    "graceful_1|--no_dropout --shutdown_ranks \"1\" --shutdown_after 180"
    "graceful_3|--no_dropout --shutdown_ranks \"1 2 3\" --shutdown_after 180"
)

# ---- Helper: run one trial ----------------------------------
run_trial() {
    local exp_name="$1"
    local exp_args="$2"
    local trial="$3"
    local out_name="experiments/${exp_name}/trial_${trial}"

    echo "  [START] $exp_name trial $trial"
    eval bash "$FT_SCRIPT" \
        --wallclock "$WALLCLOCK" \
        --output "$out_name" \
        $BUILD_FLAG \
        $exp_args \
        > "$BASE_OUTPUT/${exp_name}_trial_${trial}.log" 2>&1
    echo "  [DONE]  $exp_name trial $trial"
}

export -f run_trial
export FT_SCRIPT BASE_OUTPUT WALLCLOCK BUILD_FLAG

# ---- Run all experiments ------------------------------------
TOTAL=$(( ${#EXPERIMENTS[@]} * TRIALS ))
DONE=0

echo ""
echo "=== Starting experiments: ${#EXPERIMENTS[@]} types x ${TRIALS} trials = ${TOTAL} total runs"
echo "    Max parallel: $PARALLEL   Wallclock per run: ${WALLCLOCK}s"
echo "    Output: $BASE_OUTPUT"
echo ""

for exp_entry in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r exp_name exp_args <<< "$exp_entry"
    mkdir -p "$BASE_OUTPUT/$exp_name"
    echo "--- Experiment: $exp_name ---"
    echo "    Args: $exp_args"

    PIDS=()
    for (( t=1; t<=TRIALS; t++ )); do
        # Wait if we've hit the parallel limit
        while [ "${#PIDS[@]}" -ge "$PARALLEL" ]; do
            # Check for any finished jobs
            NEW_PIDS=()
            for pid in "${PIDS[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    NEW_PIDS+=("$pid")
                else
                    wait "$pid" || true
                    DONE=$(( DONE + 1 ))
                fi
            done
            PIDS=("${NEW_PIDS[@]}")
            [ "${#PIDS[@]}" -ge "$PARALLEL" ] && sleep 5
        done

        # Launch trial in background
        run_trial "$exp_name" "$exp_args" "$t" &
        PIDS+=($!)
    done

    # Wait for all trials of this experiment to finish
    for pid in "${PIDS[@]}"; do
        wait "$pid" || true
        DONE=$(( DONE + 1 ))
    done

    echo "    Experiment $exp_name complete ($DONE/$TOTAL total runs done)"
    echo ""
done

echo "=== All experiments complete. Results in: $BASE_OUTPUT"
echo "    Run aggregate_results.py to generate comparison plots:"
echo "    python scripts/aggregate_results.py $BASE_OUTPUT"
