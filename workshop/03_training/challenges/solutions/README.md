# Challenge Solutions

Complete working solutions for the Module 3 challenges.

## Prerequisites

```bash
cd ~/Code/nanochat
source .venv/bin/activate
```

## Solutions

| File | Challenge | Model |
|------|-----------|-------|
| `01_pirate_data.py` | Pirate Personality | None |
| `02_lr_sweep.sh` | Learning Rate Archaeology | Trains new |
| `03_token_predictions.py` | Token Autopsy | Local model |
| `04_frankenstein.md` | Frankenstein Model | Guide |
| `05_embeddings.py` | Embedding Cartography | Local model |
| `06_loss_calculator.py` | Loss Detective | None |

## Quick Test

```bash
# No model needed
uv run python workshop/03_training/challenges/solutions/01_pirate_data.py
uv run python workshop/03_training/challenges/solutions/06_loss_calculator.py --loss 6.0

# Uses your workshop-trained model (requires training first)
uv run python workshop/03_training/challenges/solutions/03_token_predictions.py -p "Hello"
uv run python workshop/03_training/challenges/solutions/05_embeddings.py
```

## Compare with Full Model

To compare your workshop model with the full nanochat, visit:
**https://nanochat.karpathy.ai/**

Note: Workshop d4 models (~20M params) produce noisy outputs compared to the full model.
