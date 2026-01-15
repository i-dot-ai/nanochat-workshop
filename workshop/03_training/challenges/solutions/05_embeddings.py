#!/usr/bin/env python3
"""
Solution: Analyse token embedding relationships.

Extracts token embeddings and analyses semantic relationships
through cosine similarity and word analogies.

Usage:
    python 05_embeddings.py
    python 05_embeddings.py --words "cat dog bird fish"
    python 05_embeddings.py --analogy "man woman king"
    python 05_embeddings.py --local  # Use workshop-trained model
"""
import argparse
import glob
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

import torch
import numpy as np

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def get_token_embedding(model, tokenizer, word: str) -> np.ndarray:
    """Get embedding for a word. Mean if multi-token."""
    tokens = tokenizer.encode(word)
    embeddings = model.transformer.wte.weight[tokens].detach().cpu().numpy()
    return embeddings.mean(axis=0) if len(tokens) > 1 else embeddings[0]


def get_embeddings(model, tokenizer, words: list[str]) -> dict[str, np.ndarray]:
    """Get embeddings for a list of words."""
    return {word: get_token_embedding(model, tokenizer, word) for word in words}


def find_most_similar(
    target: np.ndarray,
    embeddings: dict[str, np.ndarray],
    exclude: set[str] = None,
    top_k: int = 5
) -> list[tuple[str, float]]:
    """Find most similar words to target vector."""
    exclude = exclude or set()
    similarities = [
        (word, cosine_similarity(target, emb))
        for word, emb in embeddings.items()
        if word not in exclude
    ]
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def test_analogy(
    embeddings: dict[str, np.ndarray],
    a: str, b: str, c: str,
    top_k: int = 3
) -> list[tuple[str, float]]:
    """Test analogy: A is to B as C is to ?"""
    if not all(w in embeddings for w in [a, b, c]):
        return []
    target = embeddings[b] - embeddings[a] + embeddings[c]
    return find_most_similar(target, embeddings, exclude={a, b, c}, top_k=top_k)


def display_similarity_matrix(embeddings: dict[str, np.ndarray], pairs: list[tuple[str, str]]) -> None:
    """Display pairwise similarities."""
    print("\nWord Pair Similarities")
    print("─" * 40)
    for word1, word2 in pairs:
        if word1 in embeddings and word2 in embeddings:
            sim = cosine_similarity(embeddings[word1], embeddings[word2])
            print(f"  {word1:10} ↔ {word2:10}  {sim:+.4f}")


def display_analogy(a: str, b: str, c: str, results: list[tuple[str, float]]) -> None:
    """Display analogy test results."""
    print(f"\nAnalogy: {a} → {b} :: {c} → ?")
    print("─" * 40)
    if results:
        for word, sim in results:
            print(f"  {word:15} (similarity: {sim:+.4f})")
    else:
        print("  (missing embeddings)")


def display_nearest_neighbours(word: str, embeddings: dict[str, np.ndarray], top_k: int = 5) -> None:
    """Display nearest neighbours for a word."""
    if word not in embeddings:
        print(f"  '{word}' not in vocabulary")
        return
    similar = find_most_similar(embeddings[word], embeddings, exclude={word}, top_k=top_k)
    print(f"\nNearest neighbours to '{word}':")
    print("─" * 30)
    for neighbour, sim in similar:
        print(f"  {neighbour:15} {sim:+.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse token embedding relationships",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 05_embeddings.py
  python 05_embeddings.py --words "happy sad angry calm"
  python 05_embeddings.py --analogy "man woman king"
  python 05_embeddings.py --local  # Use workshop model
        """
    )
    parser.add_argument("--words", type=str, help="Space-separated words to analyse")
    parser.add_argument("--analogy", type=str, help="Three words: 'a b c' for a→b :: c→?")
    parser.add_argument("--neighbours", type=str, help="Find neighbours for this word")
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
            print("Note: Small workshop models have noisy embeddings")
        except FileNotFoundError:
            print(f"\nError: No local checkpoint found for '{args.source}'")
            print("Train a model first, or remove --local to use HuggingFace model")
            return 1
    else:
        model, tokenizer = load_hf_model(device, args.hf_repo)

    # Default word set
    if args.words:
        words = args.words.split()
    else:
        words = [
            "good", "bad", "happy", "sad",
            "king", "queen", "man", "woman",
            "cat", "dog", "bird", "fish",
            "run", "walk", "jump", "swim",
            "red", "blue", "green", "yellow",
            "one", "two", "three", "four",
        ]

    print(f"\nExtracting embeddings for {len(words)} words...")
    embeddings = get_embeddings(model, tokenizer, words)

    # Handle specific modes
    if args.analogy:
        analogy_words = args.analogy.split()
        if len(analogy_words) >= 3:
            a, b, c = analogy_words[:3]
            for w in [a, b, c]:
                if w not in embeddings:
                    embeddings[w] = get_token_embedding(model, tokenizer, w)
            results = test_analogy(embeddings, a, b, c)
            display_analogy(a, b, c, results)
        return 0

    if args.neighbours:
        if args.neighbours not in embeddings:
            embeddings[args.neighbours] = get_token_embedding(model, tokenizer, args.neighbours)
        display_nearest_neighbours(args.neighbours, embeddings)
        return 0

    # Default: comprehensive analysis
    print("\n" + "=" * 50)
    print("Embedding Analysis")
    print("=" * 50)

    pairs = [
        ("good", "bad"),
        ("happy", "sad"),
        ("king", "queen"),
        ("man", "woman"),
        ("cat", "dog"),
        ("run", "walk"),
    ]
    display_similarity_matrix(embeddings, pairs)

    print("\n" + "=" * 50)
    print("Analogy Tests")
    print("=" * 50)

    analogies = [
        ("man", "woman", "king"),
        ("good", "bad", "happy"),
    ]
    for a, b, c in analogies:
        results = test_analogy(embeddings, a, b, c, top_k=3)
        display_analogy(a, b, c, results)

    print("\n" + "=" * 50)
    print("Nearest Neighbours")
    print("=" * 50)

    for word in ["king", "good", "cat"]:
        display_nearest_neighbours(word, embeddings)

    return 0


if __name__ == "__main__":
    sys.exit(main())
