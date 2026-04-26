#!/bin/bash
# ============================================================
# p2p_fault_tolerance.sh
# Builds and runs the decentralized P2P EXAMM fault-tolerance
# simulation with all three failure modes.
#
# Failure modes
# -------------
# SOFT DROPOUT   -- rank goes silent (stops heartbeats) but the
#                   process keeps running.  Models a network
#                   partition or a severely overloaded node.
#                   Token ring is unbroken; peers detect via
#                   heartbeat timeout.  Rank can recover and rejoin.
#
# HARD CRASH     -- rank stops ALL MPI activity immediately.
#                   No last-words send.  Token ring is severed.
#                   Peers detect via heartbeat timeout; the token
#                   is regenerated and routed around the dead node
#                   via the logical-ring mechanism.  Rank can recover.
#
# GRACEFUL SHUTDOWN -- rank broadcasts its best genome to every
#                   active peer, announces departure, then keeps
#                   forwarding the token until consensus.
#                   Permanent: rank does not rejoin.
#
# Usage:
#   bash p2p_fault_tolerance.sh [OPTIONS]
#
# Options (all optional — defaults shown):
#   --np                  <int>    Number of MPI ranks              (default: 4)
#   --wallclock           <int>    Max run time in seconds          (default: 600)
#
#   --dropout_rank        <int>    Rank for soft dropout            (default: 2)
#   --dropout_after       <float>  Seconds before soft dropout      (default: 20)
#   --recovery_after      <float>  Seconds before soft dropout recovers (default: 20)
#   --no_dropout                   Disable soft dropout             (disabled by default)
#
#   --hard_crash_rank     <int>    Rank for hard crash              (default: 1)
#   --hard_crash_after    <float>  Seconds before hard crash fires  (default: 180)
#   --hard_crash_recovery <float>  Seconds before hard crash recovers (default: 60)
#   --no_hard_crash                Disable hard crash
#
#   --shutdown_rank       <int>    Rank for graceful shutdown       (default: 3)
#   --shutdown_after      <float>  Seconds before shutdown fires    (default: 180)
#   --no_shutdown                  Disable graceful shutdown        (disabled by default)
#
#   --hb_interval         <int>    Heartbeat interval in ms         (default: 1000)
#   --hb_timeout          <int>    Heartbeat timeout in ms          (default: 5000)
#   --snapshot_interval   <int>    Periodic top-k snapshot push (ms)(default: 10000, 0=off)
#   --experiment          <str>    Experiment name (subfolder under test_output/experiments/) (default: hard_crash)
#   --trial               <int>    Trial number within the experiment (default: 1)
#   --skip_build                   Skip cmake/make step
#   -h, --help                     Show this message
#
# Default run:
#   wallclock 600s
#   t=180s rank 1 hard crash    (recovers at t=240s)
#   dropout and graceful shutdown disabled by default
#
# Example — only test hard crash:
#   bash p2p_fault_tolerance.sh --no_dropout --no_shutdown
#
# Example — custom timing:
#   bash p2p_fault_tolerance.sh --hard_crash_after 180 --hard_crash_recovery 60
# ============================================================

set -e

# ---- Defaults -----------------------------------------------
NP=4
WALLCLOCK=600

DROPOUT_RANK=2
DROPOUT_AFTER=20
RECOVERY_AFTER=20
ENABLE_DROPOUT=false

HARD_CRASH_RANK=1
HARD_CRASH_AFTER=180
HARD_CRASH_RECOVERY=60
ENABLE_HARD_CRASH=true

SHUTDOWN_RANK=3
SHUTDOWN_AFTER=180
ENABLE_SHUTDOWN=false

HB_INTERVAL=1000
HB_TIMEOUT=5000
SNAPSHOT_INTERVAL=10000
EXPERIMENT_NAME="hard_crash"
TRIAL_NUM="1"
SKIP_BUILD=false

# ---- Parse args ---------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)                  NP="$2";                shift 2 ;;
        --wallclock)           WALLCLOCK="$2";          shift 2 ;;

        --dropout_rank)        DROPOUT_RANK="$2";       shift 2 ;;
        --dropout_after)       DROPOUT_AFTER="$2";      shift 2 ;;
        --recovery_after)      RECOVERY_AFTER="$2";     shift 2 ;;
        --no_dropout)          ENABLE_DROPOUT=false;    shift ;;

        --hard_crash_rank)     HARD_CRASH_RANK="$2";    shift 2 ;;
        --hard_crash_after)    HARD_CRASH_AFTER="$2";   shift 2 ;;
        --hard_crash_recovery) HARD_CRASH_RECOVERY="$2"; shift 2 ;;
        --no_hard_crash)       ENABLE_HARD_CRASH=false; shift ;;

        --shutdown_rank)       SHUTDOWN_RANK="$2";      shift 2 ;;
        --shutdown_after)      SHUTDOWN_AFTER="$2";     shift 2 ;;
        --no_shutdown)         ENABLE_SHUTDOWN=false;   shift ;;

        --hb_interval)         HB_INTERVAL="$2";        shift 2 ;;
        --hb_timeout)          HB_TIMEOUT="$2";         shift 2 ;;
        --snapshot_interval)   SNAPSHOT_INTERVAL="$2";  shift 2 ;;
        --experiment)          EXPERIMENT_NAME="$2";    shift 2 ;;
        --trial)               TRIAL_NUM="$2";          shift 2 ;;
        --skip_build)          SKIP_BUILD=true;          shift ;;
        -h|--help)
            sed -n '/^# Usage/,/^# ===/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *)
            echo "Unknown option: $1  (run with --help to see usage)"
            exit 1 ;;
    esac
done

# ---- Paths --------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXACT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$EXACT_DIR/build_p2p"
DATASET_DIR="$EXACT_DIR/datasets/2019_ngafid_transfer"
EXPERIMENTS_DIR="$EXACT_DIR/test_output/experiments"
OUTPUT_DIR="$EXPERIMENTS_DIR/$EXPERIMENT_NAME/trial_$TRIAL_NUM"

# ---- Build --------------------------------------------------
if [ "$SKIP_BUILD" = false ]; then
    echo "=== Building ==================================================="
    cd "$BUILD_DIR"
    make -j4 examm_mpi
    echo "=== Build complete ============================================="
fi

# ---- Prepare output -----------------------------------------
mkdir -p "$OUTPUT_DIR"

# ---- Assemble fault-tolerance flags -------------------------
FT_FLAGS=""

if [ "$ENABLE_DROPOUT" = true ]; then
    FT_FLAGS="$FT_FLAGS \
  --dropout_rank               $DROPOUT_RANK \
  --dropout_after_seconds      $DROPOUT_AFTER \
  --recovery_after_seconds     $RECOVERY_AFTER"
fi

if [ "$ENABLE_HARD_CRASH" = true ]; then
    FT_FLAGS="$FT_FLAGS \
  --hard_crash_rank                    $HARD_CRASH_RANK \
  --hard_crash_after_seconds           $HARD_CRASH_AFTER \
  --hard_crash_recovery_after_seconds  $HARD_CRASH_RECOVERY"
fi

if [ "$ENABLE_SHUTDOWN" = true ]; then
    FT_FLAGS="$FT_FLAGS \
  --shutdown_rank              $SHUTDOWN_RANK \
  --shutdown_after_seconds     $SHUTDOWN_AFTER"
fi

FT_FLAGS="$FT_FLAGS \
  --heartbeat_interval_ms  $HB_INTERVAL \
  --heartbeat_timeout_ms   $HB_TIMEOUT \
  --snapshot_interval_ms   $SNAPSHOT_INTERVAL"

# ---- Summary ------------------------------------------------
echo ""
echo "=== Fault-Tolerance Simulation ================================="
echo "  Ranks (--np):          $NP"
echo "  Wallclock:             ${WALLCLOCK}s"
echo "  Heartbeat interval:    ${HB_INTERVAL}ms   timeout: ${HB_TIMEOUT}ms"
echo "  Token timeout:         $((HB_TIMEOUT * 2))ms  (2 × heartbeat_timeout)"
echo "  Snapshot push:         every ${SNAPSHOT_INTERVAL}ms to ring successor (0 = disabled)"
echo ""

if [ "$ENABLE_DROPOUT" = true ]; then
echo "  [SOFT DROPOUT]  rank $DROPOUT_RANK silent at t=${DROPOUT_AFTER}s"
echo "                  recovers after ${RECOVERY_AFTER}s  (back at t=$(echo "$DROPOUT_AFTER + $RECOVERY_AFTER" | bc)s)"
echo "                  Process stays alive; token ring unaffected."
else
echo "  [SOFT DROPOUT]  disabled"
fi

if [ "$ENABLE_HARD_CRASH" = true ]; then
echo "  [HARD CRASH]    rank $HARD_CRASH_RANK crashes at t=${HARD_CRASH_AFTER}s"
echo "                  recovers after ${HARD_CRASH_RECOVERY}s  (back at t=$(echo "$HARD_CRASH_AFTER + $HARD_CRASH_RECOVERY" | bc)s)"
echo "                  All MPI suspended; token regenerated via logical ring."
else
echo "  [HARD CRASH]    disabled"
fi

if [ "$ENABLE_SHUTDOWN" = true ]; then
echo "  [GRACEFUL SHUT] rank $SHUTDOWN_RANK permanently leaves at t=${SHUTDOWN_AFTER}s"
echo "                  Broadcasts best genome to all peers before departing."
else
echo "  [GRACEFUL SHUT] disabled"
fi

echo ""
echo "  Experiment: $EXPERIMENT_NAME  /  Trial: $TRIAL_NUM"
echo "  Output: $OUTPUT_DIR"
echo "================================================================"
echo ""

# ---- Run ----------------------------------------------------
cd "$BUILD_DIR"

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 \
E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM \
FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"

mpirun -np "$NP" mpi/examm_mpi \
  --training_filenames   "$DATASET_DIR"/c172_file_[1-9].csv \
  --validation_filenames "$DATASET_DIR"/c172_file_1[0-2].csv \
  --time_offset 1 \
  --input_parameter_names  $INPUT_PARAMETERS \
  --output_parameter_names Pitch \
  --number_islands 10 \
  --island_size    10 \
  --max_wallclock_seconds "$WALLCLOCK" \
  --max_genomes 2000 \
  --bp_iterations 5 \
  --num_mutations 2 \
  --normalize min_max \
  --output_directory "$OUTPUT_DIR" \
  --possible_node_types simple UGRNN MGU GRU delta LSTM \
  --std_message_level INFO \
  --file_message_level INFO \
  $FT_FLAGS

echo ""
echo "=== Run complete. Results in: $OUTPUT_DIR"
echo ""
echo "To generate graphs, run:"
echo "  python3 $EXACT_DIR/scripts/aggregate_results.py \\"
echo "          $EXPERIMENTS_DIR"
