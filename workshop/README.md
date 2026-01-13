# nanochat Workshop

Train your own ChatGPT from scratch in a day.

## Agenda

| Time | Session | Lead |
|------|---------|------|
| 10:00 - 10:15 | 👋 Intros and Setup | - |
| 10:15 - 11:00 | ✂️ Tokenisation | Jordy |
| 11:00 - 11:15 | ☕️ Break | - |
| 11:15 - 12:15 | 🤖 Model Architecture | Mark |
| 12:15 - 13:00 | 🥙 Lunch | - |
| 13:00 - 14:30 | 🏗️ Model Training | Liam |
| 14:30 - 14:45 | ☕️ Break | - |
| 14:45 - 15:45 | 📐 Eval + Finetune | Dheeraj |
| 15:45 - 16:00 | 🏁 Wrap up | - |
| 17:00+ | 🍻 Social | - |

## Sessions

- [00_setup](./00_setup/) - Environment setup and prerequisites
- [01_tokenisation](./01_tokenisation/) - How text becomes tokens
- [02_architecture](./02_architecture/) - Transformer architecture deep dive
- [03_training](./03_training/) - Training pipeline: base → mid → SFT → RL
- [04_eval_finetune](./04_eval_finetune/) - Evaluation and fine-tuning

## Quick Start

```bash
# Clone the repo
git clone https://github.com/i-dot-ai/nanochat-workshop.git
cd nanochat-workshop

# Setup environment
uv venv && uv sync --extra gpu
source .venv/bin/activate

# Run the demo training (creates a small model)
bash workshop/03_training/workshop_demo.sh my_model
```

## Requirements

- Python 3.10+
- GPU recommended (works on CPU/MPS but slower)
- ~10GB disk space for model weights
