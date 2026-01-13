# 👋 Setup (10:00 - 10:15)

Environment setup and introductions.

## Prerequisites

- Python 3.10+
- Git
- ~10GB disk space

## Setup Steps

```bash
# Clone the repo
git clone https://github.com/i-dot-ai/nanochat-workshop.git
cd nanochat-workshop

# Create virtual environment
uv venv
uv sync --extra gpu  # or --extra cpu if no GPU
source .venv/bin/activate

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## Troubleshooting

TODO: Common issues and fixes
