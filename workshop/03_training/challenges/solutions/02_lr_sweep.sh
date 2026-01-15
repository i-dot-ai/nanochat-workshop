#!/bin/bash
#
# Solution: Learning Rate Sweep Experiment
#
# This script trains multiple models with different learning rates
# to demonstrate the effect of LR on training dynamics.
#
# Usage:
#     bash 02_lr_sweep.sh              # Full sweep (5 LRs, 500 steps each)
#     bash 02_lr_sweep.sh --quick      # Quick test (3 LRs, 200 steps)
#
# Expected runtime:
#     Full:  ~15 minutes on M3
#     Quick: ~5 minutes on M3
#

set -e  # Exit on error

# Configuration
LEARNING_RATES="0.001 0.01 0.02 0.1 0.5"
NUM_ITERATIONS=500
DEPTH=4

# Parse arguments
if [[ "$1" == "--quick" ]]; then
    LEARNING_RATES="0.005 0.02 0.1"
    NUM_ITERATIONS=200
    echo "Running quick sweep (3 LRs, 200 steps)"
else
    echo "Running full sweep (5 LRs, 500 steps)"
fi

echo "=============================================="
echo "Learning Rate Sweep Experiment"
echo "=============================================="
echo ""
echo "Learning rates: $LEARNING_RATES"
echo "Steps per run:  $NUM_ITERATIONS"
echo "Model depth:    $DEPTH"
echo ""

# Create results file
RESULTS_FILE="lr_sweep_results.txt"
echo "LR,Step,Loss" > "$RESULTS_FILE"

# Run sweep
for lr in $LEARNING_RATES; do
    echo "──────────────────────────────────────────────"
    echo "Training with LR = $lr"
    echo "──────────────────────────────────────────────"

    # Run training and capture output
    uv run python -m scripts.base_train \
        --depth=$DEPTH \
        --matrix_lr=$lr \
        --num_iterations=$NUM_ITERATIONS \
        --model_tag=lr_sweep_$lr \
        --eval_every=-1 \
        --sample_every=-1 2>&1 | while read line; do
            # Parse step and loss from output
            if [[ $line =~ step\ ([0-9]+)/.*loss:\ ([0-9.]+) ]]; then
                step="${BASH_REMATCH[1]}"
                loss="${BASH_REMATCH[2]}"
                # Save every 100 steps
                if (( step % 100 == 0 )); then
                    echo "$lr,$step,$loss" >> "$RESULTS_FILE"
                    printf "  Step %5s: loss = %s\n" "$step" "$loss"
                fi
            fi
        done

    echo ""
done

echo "=============================================="
echo "Results Summary"
echo "=============================================="
echo ""

# Display results table
printf "%-8s" "LR"
for step in 100 200 300 400 500; do
    if (( step <= NUM_ITERATIONS )); then
        printf "%10s" "Step $step"
    fi
done
echo ""
echo "────────────────────────────────────────────────────"

for lr in $LEARNING_RATES; do
    printf "%-8s" "$lr"
    for step in 100 200 300 400 500; do
        if (( step <= NUM_ITERATIONS )); then
            loss=$(grep "^$lr,$step," "$RESULTS_FILE" | cut -d',' -f3)
            if [[ -n "$loss" ]]; then
                printf "%10s" "$loss"
            else
                printf "%10s" "-"
            fi
        fi
    done
    echo ""
done

echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "=============================================="
echo "Expected Observations"
echo "=============================================="
echo ""
echo "LR = 0.001: Very slow learning, loss barely decreases"
echo "LR = 0.01:  Good progress, steady convergence"
echo "LR = 0.02:  Optimal - fastest stable convergence (nanochat default)"
echo "LR = 0.1:   Unstable, loss may oscillate or increase"
echo "LR = 0.5:   Diverges, loss becomes NaN or very large"
echo ""
echo "Key insight: Learning rate controls step size in gradient descent."
echo "Too small = slow, too large = unstable/divergent."
