#!/bin/bash
# ============================================================
# p2p_fault_tolerance.sh
# Builds and runs the decentralized P2P EXAMM fault-tolerance
# simulation with configurable dropout and graceful shutdown.
#
# Usage:
#   bash p2p_fault_tolerance.sh [OPTIONS]
#
# Options (all optional - defaults shown):
#   --np                  <int>       Number of MPI ranks              (default: 4)
#   --wallclock           <int>       Max run time in seconds          (default: 600)
#   --dropout_ranks       <int...>    Ranks for unintentional failure  (default: "2")
#   --dropout_after       <float>     Seconds before dropout fires     (default: 180)
#   --recovery_after      <float>     Seconds before dropout recovers  (default: 60)
#   --shutdown_ranks      <int...>    Ranks for graceful shutdown      (default: "3")
#   --shutdown_after      <float>     Seconds before shutdown fires    (default: 480)
#   --hb_interval         <int>       Heartbeat interval in ms         (default: 1000)
#   --hb_timeout          <int>       Heartbeat timeout in ms          (default: 5000)
#   --output              <str>       Output directory name            (default: c172_fault_tolerance)
#   --no_dropout                      Disable unintentional failure
#   --no_shutdown                     Disable graceful shutdown
#   --skip_build                      Skip cmake/make step
#   -h, --help                        Show this message
#
# Examples:
#   # Clean run (no faults)
#   bash p2p_fault_tolerance.sh --no_dropout --no_shutdown --output clean
#
#   # 1 unintentional dropout at 3 min, recovers after 1 min
#   bash p2p_fault_tolerance.sh --dropout_ranks "1" --dropout_after 180 --recovery_after 60 --no_shutdown
#
#   # 3 simultaneous dropouts at 3 min, recover after 1 min
#   bash p2p_fault_tolerance.sh --dropout_ranks "1 2 3" --dropout_after 180 --recovery_after 60 --no_shutdown
#
#   # 1 graceful shutdown at 3 min
#   bash p2p_fault_tolerance.sh --no_dropout --shutdown_ranks "1" --shutdown_after 180
#
#   # 3 simultaneous graceful shutdowns at 3 min
#   bash p2p_fault_tolerance.sh --no_dropout --shutdown_ranks "1 2 3" --shutdown_after 180
# ============================================================

set -e

# ---- Defaults -----------------------------------------------
NP=4
WALLCLOCK=600
DROPOUT_RANKS="2"
DROPOUT_AFTER=180
RECOVERY_AFTER=60
SHUTDOWN_RANKS="3"
SHUTDOWN_AFTER=480
HB_INTERVAL=1000
HB_TIMEOUT=5000
OUTPUT_NAME="c172_fault_tolerance"
ENABLE_DROPOUT=true
ENABLE_SHUTDOWN=true
SKIP_BUILD=false

# ---- Parse args ---------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --np)             NP="$2";             shift 2 ;;
        --wallclock)      WALLCLOCK="$2";       shift 2 ;;
        --dropout_ranks)  DROPOUT_RANKS="$2";   shift 2 ;;
        --dropout_after)  DROPOUT_AFTER="$2";   shift 2 ;;
        --recovery_after) RECOVERY_AFTER="$2";  shift 2 ;;
        --shutdown_ranks) SHUTDOWN_RANKS="$2";  shift 2 ;;
        --shutdown_after) SHUTDOWN_AFTER="$2";  shift 2 ;;
        --hb_interval)    HB_INTERVAL="$2";     shift 2 ;;
        --hb_timeout)     HB_TIMEOUT="$2";      shift 2 ;;
        --output)         OUTPUT_NAME="$2";     shift 2 ;;
        --no_dropout)     ENABLE_DROPOUT=false; shift ;;
        --no_shutdown)    ENABLE_SHUTDOWN=false; shift ;;
        --skip_build)     SKIP_BUILD=true;      shift ;;
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
OUTPUT_DIR="$EXACT_DIR/test_output/$OUTPUT_NAME"

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
  --dropout_ranks          $DROPOUT_RANKS \
  --dropout_after_seconds  $DROPOUT_AFTER \
  --recovery_after_seconds $RECOVERY_AFTER"
fi

if [ "$ENABLE_SHUTDOWN" = true ]; then
    FT_FLAGS="$FT_FLAGS \
  --shutdown_ranks         $SHUTDOWN_RANKS \
  --shutdown_after_seconds $SHUTDOWN_AFTER"
fi

FT_FLAGS="$FT_FLAGS \
  --heartbeat_interval_ms  $HB_INTERVAL \
  --heartbeat_timeout_ms   $HB_TIMEOUT"

# ---- Summary ------------------------------------------------
echo ""
echo "=== Fault-Tolerance Simulation ================================="
echo "  Ranks (--np):          $NP"
echo "  Wallclock:             ${WALLCLOCK}s"
if [ "$ENABLE_DROPOUT" = true ]; then
    echo "  Unintentional failure: ranks [$DROPOUT_RANKS] silent at t=${DROPOUT_AFTER}s, recover after ${RECOVERY_AFTER}s"
else
    echo "  Unintentional failure: disabled"
fi
if [ "$ENABLE_SHUTDOWN" = true ]; then
    echo "  Graceful shutdown:     ranks [$SHUTDOWN_RANKS] leave at t=${SHUTDOWN_AFTER}s (permanent)"
else
    echo "  Graceful shutdown:     disabled"
fi
echo "  Heartbeat interval:    ${HB_INTERVAL}ms   timeout: ${HB_TIMEOUT}ms"
echo "  Output:                $OUTPUT_DIR"
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
