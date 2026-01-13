# 🏗️ Model Training (13:00 - 14:30)

**Lead: Liam**

Training pipeline: Base → Mid → SFT → RL

## Topics

- Pre-training on raw text
- Continued pre-training (mid-training)
- Supervised fine-tuning (SFT)
- Reinforcement learning (RL)
- Loss curves and metrics

## Quick Start

```bash
# Run the full 4-stage pipeline (~30 min on M3)
bash workshop/03_training/workshop_demo.sh my_model

# Chat with your trained model
python -m scripts.chat_cli --source=rl --model_tag=my_model
```

## Materials

- [Participant Worksheet](./module3_participant_worksheet.md)
- [Challenges & Solutions](./challenges/)

## Challenges

See [challenges/solutions/](./challenges/solutions/) for worked examples:

| Challenge | Description |
|-----------|-------------|
| 01_pirate_data.py | Generate personality training data |
| 02_lr_sweep.sh | Learning rate experiments |
| 03_token_predictions.py | Next-token prediction analysis |
| 04_frankenstein.md | Domain shift experiments |
| 05_embeddings.py | Embedding similarity analysis |
| 06_loss_calculator.py | Loss ↔ perplexity converter |
