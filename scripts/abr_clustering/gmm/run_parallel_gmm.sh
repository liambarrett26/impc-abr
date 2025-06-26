#!/bin/bash

# Parallel GMM clustering pipeline
# This script orchestrates parallel execution of GMM models with different parameters

# Default parameters
DATA_PATH=""
OUTPUT_DIR="results"
SHARED_DATA_DIR="shared_data"
MIN_K=3
MAX_K=12
MIN_MUTANTS=3
MIN_CONTROLS=20
N_BOOTSTRAP=100
RANDOM_STATE=42
LOG_LEVEL="INFO"
MAX_JOBS=8
SKIP_PREPROCESSING=false
SKIP_ANALYSIS=false

# Parse command line arguments
usage() {
    echo "Usage: $0 -d DATA_PATH [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  -d, --data-path PATH         Path to IMPC ABR data file"
    echo ""
    echo "Optional:"
    echo "  -o, --output-dir DIR         Output directory (default: results)"
    echo "  -s, --shared-dir DIR         Shared data directory (default: shared_data)"
    echo "  --min-k NUM                  Minimum number of clusters (default: 3)"
    echo "  --max-k NUM                  Maximum number of clusters (default: 12)"
    echo "  --min-mutants NUM            Min mutant mice per group (default: 3)"
    echo "  --min-controls NUM           Min control mice per group (default: 20)"
    echo "  --n-bootstrap NUM            Bootstrap iterations (default: 100)"
    echo "  --random-state NUM           Random seed (default: 42)"
    echo "  --log-level LEVEL            Log level (default: INFO)"
    echo "  -j, --max-jobs NUM           Maximum parallel jobs (default: 8)"
    echo "  --skip-preprocessing         Skip preprocessing step"
    echo "  --skip-analysis              Skip analysis for individual models"
    echo "  -h, --help                   Show this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -s|--shared-dir)
            SHARED_DATA_DIR="$2"
            shift 2
            ;;
        --min-k)
            MIN_K="$2"
            shift 2
            ;;
        --max-k)
            MAX_K="$2"
            shift 2
            ;;
        --min-mutants)
            MIN_MUTANTS="$2"
            shift 2
            ;;
        --min-controls)
            MIN_CONTROLS="$2"
            shift 2
            ;;
        --n-bootstrap)
            N_BOOTSTRAP="$2"
            shift 2
            ;;
        --random-state)
            RANDOM_STATE="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        -j|--max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --skip-preprocessing)
            SKIP_PREPROCESSING=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Check required arguments
if [ -z "$DATA_PATH" ]; then
    echo "Error: Data path is required"
    usage
fi

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$SHARED_DATA_DIR"

# Log file for main process
MAIN_LOG="$OUTPUT_DIR/parallel_pipeline.log"
echo "Starting parallel GMM pipeline at $(date)" | tee "$MAIN_LOG"
echo "Configuration:" | tee -a "$MAIN_LOG"
echo "  Data path: $DATA_PATH" | tee -a "$MAIN_LOG"
echo "  Output directory: $OUTPUT_DIR" | tee -a "$MAIN_LOG"
echo "  K range: $MIN_K to $MAX_K" | tee -a "$MAIN_LOG"
echo "  Covariance types: full, tied" | tee -a "$MAIN_LOG"
echo "  Max parallel jobs: $MAX_JOBS" | tee -a "$MAIN_LOG"

# Step 1: Preprocessing (if not skipped)
if [ "$SKIP_PREPROCESSING" = false ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "Step 1: Running preprocessing..." | tee -a "$MAIN_LOG"
    
    python preprocess_data.py \
        "$DATA_PATH" \
        --output-dir "$SHARED_DATA_DIR" \
        --min-mutants "$MIN_MUTANTS" \
        --min-controls "$MIN_CONTROLS" \
        --log-level "$LOG_LEVEL" 2>&1 | tee -a "$MAIN_LOG"
    
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "Error: Preprocessing failed" | tee -a "$MAIN_LOG"
        exit 1
    fi
    
    echo "Preprocessing completed successfully" | tee -a "$MAIN_LOG"
else
    echo "Skipping preprocessing step" | tee -a "$MAIN_LOG"
fi

# Step 2: Parallel model training
echo "" | tee -a "$MAIN_LOG"
echo "Step 2: Starting parallel model training..." | tee -a "$MAIN_LOG"

# Create job tracking directory
JOB_DIR="$OUTPUT_DIR/jobs"
mkdir -p "$JOB_DIR"

# Function to run a single model
run_model() {
    local k=$1
    local cov=$2
    local job_name="gmm_k${k}_${cov}"
    local job_file="$JOB_DIR/${job_name}.job"
    
    echo "Starting job: $job_name" >> "$MAIN_LOG"
    
    # Create job script
    cat > "$job_file" << EOF
#!/bin/bash
cd "$(pwd)"
python pipeline_parallel.py $k $cov \\
    --shared-data-dir "$SHARED_DATA_DIR" \\
    --output-dir "$OUTPUT_DIR" \\
    --n-bootstrap "$N_BOOTSTRAP" \\
    --random-state "$RANDOM_STATE" \\
    --log-level "$LOG_LEVEL" \\
    $([ "$SKIP_ANALYSIS" = true ] && echo "--skip-analysis")
EOF
    
    chmod +x "$job_file"
    bash "$job_file" &
}

# Launch all jobs
job_count=0
for k in $(seq $MIN_K $MAX_K); do
    for cov in "full" "tied"; do
        # Wait if we've reached max parallel jobs
        while [ $(jobs -r | wc -l) -ge $MAX_JOBS ]; do
            sleep 1
        done
        
        run_model $k $cov
        ((job_count++))
    done
done

echo "Launched $job_count model training jobs" | tee -a "$MAIN_LOG"

# Wait for all jobs to complete
echo "Waiting for all jobs to complete..." | tee -a "$MAIN_LOG"
wait

# Check for completed models
echo "" | tee -a "$MAIN_LOG"
echo "Checking job completion status..." | tee -a "$MAIN_LOG"

completed=0
failed=0

for k in $(seq $MIN_K $MAX_K); do
    for cov in "full" "tied"; do
        model_dir="$OUTPUT_DIR/gmm_k${k}_${cov}"
        if [ -f "$model_dir/completed.txt" ]; then
            ((completed++))
            echo "  ✓ gmm_k${k}_${cov}: completed" | tee -a "$MAIN_LOG"
        elif [ -f "$model_dir/training_failed.json" ] || [ -f "$model_dir/pipeline_failed.json" ]; then
            ((failed++))
            echo "  ✗ gmm_k${k}_${cov}: failed" | tee -a "$MAIN_LOG"
        else
            ((failed++))
            echo "  ? gmm_k${k}_${cov}: unknown status" | tee -a "$MAIN_LOG"
        fi
    done
done

echo "" | tee -a "$MAIN_LOG"
echo "Model training summary: $completed completed, $failed failed" | tee -a "$MAIN_LOG"

# Step 3: Model selection and aggregation
if [ $completed -gt 0 ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "Step 3: Running model selection and aggregation..." | tee -a "$MAIN_LOG"
    
    python aggregate_results.py \
        --results-dir "$OUTPUT_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --log-level "$LOG_LEVEL" 2>&1 | tee -a "$MAIN_LOG"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "Model selection completed successfully" | tee -a "$MAIN_LOG"
    else
        echo "Warning: Model selection encountered errors" | tee -a "$MAIN_LOG"
    fi
else
    echo "Error: No models completed successfully" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "" | tee -a "$MAIN_LOG"
echo "Parallel GMM pipeline completed at $(date)" | tee -a "$MAIN_LOG"

# Generate summary report
if [ -f "$OUTPUT_DIR/model_selection_results.json" ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "Best model information:" | tee -a "$MAIN_LOG"
    python -c "
import json
with open('$OUTPUT_DIR/model_selection_results.json', 'r') as f:
    results = json.load(f)
    best = results.get('best_model', {})
    print(f\"  Model: {best.get('model_name', 'Unknown')}\")
    print(f\"  BIC: {best.get('bic', 'N/A'):.2f}\")
    print(f\"  Silhouette: {best.get('silhouette', 'N/A'):.3f}\")
    print(f\"  Stability: {best.get('stability_score', 'N/A'):.3f}\")
" | tee -a "$MAIN_LOG"
fi

echo "" | tee -a "$MAIN_LOG"
echo "Results saved to: $OUTPUT_DIR" | tee -a "$MAIN_LOG"