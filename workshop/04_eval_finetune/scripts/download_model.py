"""
Download nanochat-d32 pretrained model from HuggingFace.

Usage:
    python -m workshop.04_eval_finetune.scripts.download_model

This downloads the model files to workshop/04_eval_finetune/models/pretrained/
"""

import os
from huggingface_hub import hf_hub_download


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def download_nanochat_d32():
    """Download the nanochat-d32 model from HuggingFace."""
    repo_id = "karpathy/nanochat-d32"
    workshop_dir = get_workshop_dir()
    base_dir = os.path.join(workshop_dir, "models", "pretrained")

    # Create directories
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    model_dir = os.path.join(base_dir, "chatsft_checkpoints", "d32")
    os.makedirs(tokenizer_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Downloading nanochat-d32 to {base_dir}...")

    # Download tokenizer files
    print("Downloading tokenizer files...")
    for filename in ["token_bytes.pt", "tokenizer.pkl"]:
        local_path = os.path.join(tokenizer_dir, filename)
        if os.path.exists(local_path):
            print(f"  {filename} already exists, skipping")
        else:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=tokenizer_dir,
                local_dir_use_symlinks=False,
            )
            print(f"  Downloaded {filename}")

    # Download model files
    print("Downloading model files...")
    for filename in ["meta_000650.json", "model_000650.pt"]:
        local_path = os.path.join(model_dir, filename)
        if os.path.exists(local_path):
            print(f"  {filename} already exists, skipping")
        else:
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=model_dir,
                local_dir_use_symlinks=False,
            )
            print(f"  Downloaded {filename}")

    print(f"\nModel downloaded successfully!")
    print(f"Tokenizer: {tokenizer_dir}")
    print(f"Model: {model_dir}")

    return base_dir


if __name__ == "__main__":
    download_nanochat_d32()
