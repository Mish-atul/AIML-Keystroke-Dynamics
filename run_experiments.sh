#!/bin/bash
# =============================================================================
# Keystroke Dynamics - Contrastive Encoder Experiments
# =============================================================================
# Full experiment runner for training, evaluation, and deployment
#
# Usage:
#   bash run_experiments.sh           # Run full pipeline
#   bash run_experiments.sh --quick   # Quick test run
# =============================================================================

set -e

# Configuration
CSV_PATH="cmu_keystroke.csv"
ENCODER_PATH="artifacts/encoder.pt"
EPOCHS=120
BATCH_SIZE=128
SHOTS=5

# Parse arguments
QUICK_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS=5
            BATCH_SIZE=32
            shift
            ;;
        *)
            shift
            ;;
    esac
done

echo "=============================================="
echo "KEYSTROKE DYNAMICS - CONTRASTIVE ENCODER"
echo "=============================================="
echo "Quick Mode: $QUICK_MODE"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo ""

# Step 0: Prepare dataset
echo "[1/6] Preparing dataset..."
if [ ! -f "$CSV_PATH" ]; then
    if [ -f "model training/DSL-StrongPasswordData.csv" ]; then
        cp "model training/DSL-StrongPasswordData.csv" "$CSV_PATH"
        echo "  Copied dataset to $CSV_PATH"
    else
        echo "  ERROR: Dataset not found!"
        exit 1
    fi
fi

# Step 1: Run smoke tests
echo ""
echo "[2/6] Running smoke tests..."
python scripts/run_smoke_tests.py --test data
python scripts/run_smoke_tests.py --test encoder
python scripts/run_smoke_tests.py --test adapter

# Step 2: Train encoder
echo ""
echo "[3/6] Training encoder..."
python scripts/train_encoder.py \
    --csv "$CSV_PATH" \
    --out "$ENCODER_PATH" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --split user_disjoint

# Step 3: Train adapters for sample users
echo ""
echo "[4/6] Training adapters..."
python scripts/train_adapter.py \
    --encoder "$ENCODER_PATH" \
    --csv "$CSV_PATH" \
    --all \
    --shots $SHOTS

# Step 4: Evaluate
echo ""
echo "[5/6] Evaluating models..."
python scripts/evaluate.py \
    --encoder "$ENCODER_PATH" \
    --csv "$CSV_PATH" \
    --split user_disjoint \
    --out results/ \
    --baselines

# Step 5: Summary
echo ""
echo "[6/6] Experiment complete!"
echo "=============================================="
echo "Results:"
echo "  - Encoder: $ENCODER_PATH"
echo "  - Adapters: artifacts/adapters/"
echo "  - Templates: artifacts/templates/"
echo "  - Results: results/"
echo ""
echo "To run the demo:"
echo "  streamlit run streamlit_app/app.py"
echo ""
echo "To run the API:"
echo "  cd api && uvicorn app:app --reload"
echo "=============================================="
