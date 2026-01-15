# Finetuning nanochat-d32

This directory contains scripts to finetune the pretrained nanochat-d32 model from HuggingFace.

## Quick Start

From `workshop/04_eval_finetune/` directory:

```bash
make install   # Create venv and install dependencies
make download  # Download model from HuggingFace
make finetune  # Finetune on example data
```

## Directory Structure

All data is stored under `workshop/04_eval_finetune/`:

```
workshop/04_eval_finetune/
├── .venv/                          # Virtual environment
├── data/
│   └── example_data.jsonl          # Training data
├── models/
│   ├── pretrained/                 # Downloaded from HuggingFace
│   │   ├── tokenizer/
│   │   │   ├── token_bytes.pt
│   │   │   └── tokenizer.pkl
│   │   └── chatsft_checkpoints/
│   │       └── d32/
│   │           ├── meta_000650.json
│   │           └── model_000650.pt
│   └── finetuned/                  # Your finetuned models
│       └── <output_tag>/
│           ├── meta_*.json
│           └── model_*.pt
└── finetune/                       # Scripts
    └── ...
```

## Make Commands

| Command | Description |
|---------|-------------|
| `make install` | Create venv and install dependencies |
| `make download` | Download nanochat-d32 from HuggingFace |
| `make finetune` | Finetune on example data (single GPU) |
| `make finetune-multi` | Finetune on example data (8 GPUs) |
| `make finetune DATA=mydata.jsonl` | Finetune on custom data |
| `make chat` | Chat with the model (CLI) |
| `make chat-web` | Chat with the model (Web UI) |
| `make clean` | Remove virtual environment |

## Prepare Your Data

Create a JSONL file with conversations in this format:

```json
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "2+2 equals 4."}]}
```

See `data/example_data.jsonl` for more examples.

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_path` | (required) | Path to JSONL file with conversations |
| `--source` | `sft` | Which checkpoint to load: base, mid, sft |
| `--model_tag` | `d32` | Model tag (depth) |
| `--num_epochs` | `1` | Number of training epochs |
| `--device_batch_size` | `4` | Per-GPU batch size |
| `--target_examples_per_step` | `16` | Effective batch size |
| `--matrix_lr` | `0.01` | Learning rate for transformer layers (Muon) |
| `--embedding_lr` | `0.1` | Learning rate for embeddings (AdamW) |
| `--unembedding_lr` | `0.002` | Learning rate for output head (AdamW) |
| `--eval_every` | `50` | Evaluate validation loss every N steps |
| `--val_split` | `0.1` | Fraction of data for validation |
| `--output_tag` | `finetuned` | Name for output checkpoint |

## Data Format

Each line is a JSON object with a `messages` array. Messages alternate between `user` and `assistant` roles:

```json
{
  "messages": [
    {"role": "user", "content": "User's question or prompt"},
    {"role": "assistant", "content": "Assistant's response"},
    {"role": "user", "content": "Follow-up question"},
    {"role": "assistant", "content": "Follow-up response"}
  ]
}
```

The model learns to predict assistant responses given the conversation history.

## Tips

- **Memory**: Reduce `--device_batch_size` if you get OOM errors (try 2 or 1)
- **Learning rate**: The default LRs are conservative. You can try higher values for faster convergence
- **Epochs**: For small datasets, multiple epochs may be needed. Watch for overfitting (val loss increasing)
- **Data quality**: The model learns from your examples. High-quality, diverse examples lead to better results
