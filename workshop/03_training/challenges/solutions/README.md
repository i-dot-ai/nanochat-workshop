# Challenge Solutions

Complete working solutions for the Module 3 challenges.

## Prerequisites

```bash
cd ~/Code/nanochat
source .venv/bin/activate
uv pip install huggingface_hub  # For model download
```

## Solutions

| File | Challenge | Model |
|------|-----------|-------|
| `01_pirate_data.py` | Pirate Personality | None |
| `02_lr_sweep.sh` | Learning Rate Archaeology | Trains new |
| `03_token_predictions.py` | Token Autopsy | HuggingFace d32 |
| `04_frankenstein.md` | Frankenstein Model | Guide |
| `05_embeddings.py` | Embedding Cartography | HuggingFace d32 |
| `06_loss_calculator.py` | Loss Detective | None |

## Quick Test

```bash
# No model needed
python workshop/03_training/challenges/solutions/01_pirate_data.py
python workshop/03_training/challenges/solutions/06_loss_calculator.py --loss 6.0

# Downloads karpathy/nanochat-d32 (~7GB) on first run
python workshop/03_training/challenges/solutions/03_token_predictions.py -p "The capital of France is"
python workshop/03_training/challenges/solutions/05_embeddings.py
```

## Using Your Workshop Model

To compare with your locally-trained model:

```bash
python workshop/03_training/challenges/solutions/03_token_predictions.py --local -p "Hello"
python workshop/03_training/challenges/solutions/05_embeddings.py --local
```

Note: Workshop d4 models (~20M params) produce noisy outputs. Use `--local` to see the difference.
