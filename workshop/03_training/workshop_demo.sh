#!/bin/bash
set -e

# =============================================================================
# NANOCHAT WORKSHOP DEMO - Full 4-Stage Pipeline
# =============================================================================
# Module 3: Training Pipeline End-to-End
#
# TIMING BREAKDOWN (tested on MacBook M3):
#   Stage 1: Base Training (30K steps)  ~13 min
#   Stage 2: Mid Training (1000 steps)   ~4 min
#   Stage 3: SFT (200 steps)             ~2 min
#   Stage 4: RL (30 steps)              ~11 min  [optional]
#   ------------------------------------------
#   TOTAL:                              ~30 min
#
# Run as: bash workshop/03_training/workshop_demo.sh
# =============================================================================

echo ""
echo "  _            _____ "
echo " (_)     /\\   |_   _|"
echo "  _     /  \\    | |  "
echo " | |   / /\\ \\   | |  "
echo " | |_ / ____ \\ _| |_ "
echo " |_(_)_/    \\_\\_____|"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🤖 NANOCHAT WORKSHOP - Train Your Own ChatGPT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  📋 Pipeline: Base → Mid → SFT → RL"
echo "  ⏱️  Total time: ~30 minutes"
echo ""

# Environment setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
mkdir -p "$NANOCHAT_BASE_DIR"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
    uv sync --extra cpu
fi
source .venv/bin/activate

MODEL_TAG="${1:-workshop}"
echo "🏷️  Model tag: $MODEL_TAG"
echo ""

# =============================================================================
# STAGE 1: Base Training
# =============================================================================
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  📚 STAGE 1: Base Training                     ┃"
echo "┃  Learning language from raw text               ┃"
echo "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫"
echo "┃  Steps: 30,000  |  Expected: ~13 min           ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""
START=$(date +%s)

python -m scripts.base_train \
    --depth=4 \
    --max_seq_len=512 \
    --device_batch_size=1 \
    --total_batch_size=512 \
    --num_iterations=30000 \
    --eval_every=10000 \
    --eval_tokens=65536 \
    --core_metric_every=-1 \
    --sample_every=30000 \
    --model_tag="$MODEL_TAG"

END=$(date +%s)
echo ""
echo "✅ Base training complete! ($((END-START))s)"
echo ""

# =============================================================================
# STAGE 2: Mid Training
# =============================================================================
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  🔧 STAGE 2: Mid Training                      ┃"
echo "┃  Learning conversation format + tools          ┃"
echo "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫"
echo "┃  Steps: 1,000   |  Expected: ~4 min            ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""
START=$(date +%s)

python -m scripts.mid_train \
    --max_seq_len=1024 \
    --device_batch_size=1 \
    --total_batch_size=1024 \
    --num_iterations=1000 \
    --eval_every=500 \
    --eval_tokens=32768 \
    --model_tag="$MODEL_TAG"

END=$(date +%s)
echo ""
echo "✅ Mid training complete! ($((END-START))s)"
echo ""

# =============================================================================
# STAGE 3: SFT
# =============================================================================
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  💬 STAGE 3: Supervised Fine-Tuning            ┃"
echo "┃  Learning to be a helpful assistant            ┃"
echo "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫"
echo "┃  Steps: 200     |  Expected: ~2 min            ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""
START=$(date +%s)

python -m scripts.chat_sft \
    --device_batch_size=1 \
    --target_examples_per_step=2 \
    --num_iterations=200 \
    --eval_every=100 \
    --model_tag="$MODEL_TAG"

END=$(date +%s)
echo ""
echo "✅ SFT complete! ($((END-START))s)"
echo ""

# =============================================================================
# STAGE 4: RL
# =============================================================================
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  🎯 STAGE 4: Reinforcement Learning            ┃"
echo "┃  Learning to solve maths problems              ┃"
echo "┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫"
echo "┃  Steps: 30      |  Expected: ~11 min           ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""
START=$(date +%s)

python -m scripts.chat_rl \
    --model_tag="$MODEL_TAG" \
    --device_batch_size=1 \
    --examples_per_step=4 \
    --num_samples=4 \
    --max_new_tokens=128 \
    --eval_every=30 \
    --save_every=30 \
    --num_epochs=1 2>&1 | head -600

END=$(date +%s)
echo ""
echo "✅ RL training complete! ($((END-START))s)"
echo ""

# =============================================================================
# Download Full Model for Comparison
# =============================================================================
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
echo "┃  📥 Downloading Full Model                     ┃"
echo "┃  karpathy/nanochat-d32 for comparison          ┃"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
echo ""

python -c "
from huggingface_hub import snapshot_download
import os

cache_dir = os.path.expanduser('~/.cache/nanochat/huggingface')
print('  Downloading karpathy/nanochat-d32 (~7GB)...')
path = snapshot_download('karpathy/nanochat-d32', cache_dir=cache_dir)
print(f'  ✅ Downloaded to: {path}')
"
echo ""

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🎉 PIPELINE COMPLETE!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  📁 Your checkpoints:"
echo "     Base: $NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG/"
echo "     Mid:  $NANOCHAT_BASE_DIR/mid_checkpoints/$MODEL_TAG/"
echo "     SFT:  $NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG/"
echo "     RL:   $NANOCHAT_BASE_DIR/chatrl_checkpoints/$MODEL_TAG/"
echo ""
echo "  💬 Chat with your model (d4, ~20M params):"
echo "     python -m scripts.chat_cli --source=rl --model_tag=$MODEL_TAG"
echo ""
echo "  🔬 Compare with full model (d32, ~1.5B params):"
echo "     python -m scripts.chat_cli --source=hf"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
