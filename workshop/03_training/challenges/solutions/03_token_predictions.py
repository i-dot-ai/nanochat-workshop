#!/usr/bin/env python3
"""
Solution: Examine the model's next-token predictions.

Shows exactly what the model predicts as the most likely next tokens
for any given prompt, with probabilities.

Usage:
    python 03_token_predictions.py -p "The capital of France is"
    python 03_token_predictions.py --compare
    python 03_token_predictions.py --local  # Use workshop-trained d4 model
"""
import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import torch

DEFAULT_HF_REPO = "nanochat-students/nanochat-d20"


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_hf_model(repo_id: str = DEFAULT_HF_REPO) -> str:
    """Download model from HuggingFace and return the cache path."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} from HuggingFace...")
    cache_dir = snapshot_download(repo_id=repo_id)
    return cache_dir


def load_hf_model(device: torch.device, repo_id: str = DEFAULT_HF_REPO):
    """Load model from HuggingFace repo with its own tokenizer."""
    from nanochat.gpt import GPT, GPTConfig
    from nanochat.tokenizer import RustBPETokenizer

    # Download
    hf_cache_dir = download_hf_model(repo_id)

    # Find step from downloaded files
    model_files = glob.glob(os.path.join(hf_cache_dir, "model_*.pt"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {hf_cache_dir}")
    step = int(os.path.basename(model_files[0]).split("_")[1].split(".")[0])

    # Load metadata
    meta_path = os.path.join(hf_cache_dir, f"meta_{step:06d}.json")
    with open(meta_path) as f:
        meta = json.load(f)

    # Load model weights
    model_path = os.path.join(hf_cache_dir, f"model_{step:06d}.pt")
    model_data = torch.load(model_path, map_location=device, weights_only=True)
    if device.type in {"cpu", "mps"}:
        model_data = {k: v.float() if v.dtype == torch.bfloat16 else v for k, v in model_data.items()}
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    config = GPTConfig(**meta["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Load tokenizer from HF repo (not local)
    tokenizer_path = os.path.join(hf_cache_dir, "tokenizer.pkl")
    with open(tokenizer_path, "rb") as f:
        enc = pickle.load(f)
    tokenizer = RustBPETokenizer(enc, "<|bos|>")

    print(f"Loaded {repo_id} (step {step}, {sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer


def load_local_model(source: str, device: torch.device, model_tag: str = None):
    """Load model from local checkpoint."""
    from nanochat.checkpoint_manager import load_model

    print(f"Loading local {source} model...")
    model, tokenizer, meta = load_model(
        source, device=device, phase="eval", model_tag=model_tag
    )
    print(f"Loaded (step {meta['step']}, {sum(p.numel() for p in model.parameters()):,} params)")
    return model, tokenizer


def predict_next_tokens(
    model, tokenizer, text: str, top_k: int = 5, device: torch.device = None
) -> list[tuple[str, float]]:
    """Get top-k most likely next tokens with probabilities."""
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens], device=device)

    with torch.no_grad():
        logits = model(input_ids)

    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)

    return [
        (tokenizer.decode([idx.item()]), prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]


def display_predictions(prompt: str, predictions: list[tuple[str, float]]) -> None:
    """Display predictions with visual probability bars."""
    print(f'\nPrompt: "{prompt}"')
    print("─" * 50)

    max_prob = max(p for _, p in predictions) if predictions else 1

    for i, (token, prob) in enumerate(predictions, 1):
        bar_len = int((prob / max_prob) * 20) if max_prob > 0 else 0
        bar = "█" * bar_len
        token_display = repr(token) if not token.strip() else token
        print(f"  {i}. {token_display:15} {prob*100:6.2f}%  {bar}")


def analyse_confidence(predictions: list[tuple[str, float]]) -> str:
    """Describe confidence level."""
    top_prob = predictions[0][1] if predictions else 0

    if top_prob > 0.7:
        return "Very High (>70%)"
    elif top_prob > 0.4:
        return "High (40-70%)"
    elif top_prob > 0.2:
        return "Medium (20-40%)"
    elif top_prob > 0.1:
        return "Low (10-20%)"
    else:
        return "Very Low (<10%)"


def compare_prompts(model, tokenizer, device: torch.device) -> None:
    """Compare predictions across different prompt types."""
    test_cases = [
        ("The capital of France is", "Factual - should predict 'Paris'"),
        ("2 + 2 =", "Arithmetic - should predict '4'"),
        ("Hello, how are", "Greeting - should predict 'you'"),
        ("The", "Open-ended - low confidence expected"),
        ("def fibonacci(n):", "Code - should predict 'if' or newline"),
    ]

    print("\n" + "=" * 60)
    print("Comparing Predictions Across Prompt Types")
    print("=" * 60)

    for prompt, note in test_cases:
        print(f"\n[{note}]")
        predictions = predict_next_tokens(model, tokenizer, prompt, top_k=5, device=device)
        display_predictions(prompt, predictions)
        print(f"Confidence: {analyse_confidence(predictions)}")


def main():
    parser = argparse.ArgumentParser(
        description="Examine model's next-token predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 03_token_predictions.py -p "The capital of France is"
  python 03_token_predictions.py --compare
  python 03_token_predictions.py --local -p "Hello"  # Use workshop model
        """
    )
    parser.add_argument("-p", "--prompt", type=str, help="Text prompt to analyse")
    parser.add_argument("--top_k", "-k", type=int, default=5, help="Top predictions to show")
    parser.add_argument("--compare", action="store_true", help="Compare multiple prompts")
    parser.add_argument(
        "--local", action="store_true",
        help="Use local workshop model instead of HuggingFace"
    )
    parser.add_argument("--source", choices=["base", "mid", "sft", "rl"], default="sft")
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--hf_repo", type=str, default=DEFAULT_HF_REPO)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # Load model
    if args.local:
        try:
            model, tokenizer = load_local_model(args.source, device, args.model_tag)
        except FileNotFoundError:
            print(f"\nError: No local checkpoint found for '{args.source}'")
            print("Train a model first, or remove --local to use HuggingFace model")
            return 1
    else:
        model, tokenizer = load_hf_model(device, args.hf_repo)

    # Run analysis
    if args.compare:
        compare_prompts(model, tokenizer, device)
    elif args.prompt:
        predictions = predict_next_tokens(model, tokenizer, args.prompt, args.top_k, device)
        display_predictions(args.prompt, predictions)
        print(f"\nConfidence: {analyse_confidence(predictions)}")
    else:
        # Default examples
        for prompt in ["The capital of France is", "Hello", "2 + 2 ="]:
            predictions = predict_next_tokens(model, tokenizer, prompt, 5, device)
            display_predictions(prompt, predictions)
        print("\nUse --prompt for custom text, or --compare for more examples")

    return 0


if __name__ == "__main__":
    sys.exit(main())
