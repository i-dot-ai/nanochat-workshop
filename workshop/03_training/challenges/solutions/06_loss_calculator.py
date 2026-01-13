#!/usr/bin/env python3
"""
Solution: Convert between loss, perplexity, and bits-per-byte.

This tool helps understand what loss values actually mean by converting
between different metrics and providing intuitive explanations.

Usage:
    python 06_loss_calculator.py --loss 6.0
    python 06_loss_calculator.py --perplexity 403
    python 06_loss_calculator.py --bpb 1.5
    python 06_loss_calculator.py  # Show training journey overview
"""
import argparse
import math
import sys

# nanochat defaults
VOCAB_SIZE = 65536
TOKENS_PER_BYTE = 0.30  # Approximate for BPE tokenizer


def loss_to_perplexity(loss: float) -> float:
    """Convert cross-entropy loss (nats) to perplexity."""
    return math.exp(loss)


def perplexity_to_loss(perplexity: float) -> float:
    """Convert perplexity to cross-entropy loss (nats)."""
    return math.log(perplexity)


def loss_to_bits_per_byte(loss: float, tokens_per_byte: float = TOKENS_PER_BYTE) -> float:
    """Convert loss (nats) to bits-per-byte."""
    bits_per_token = loss / math.log(2)  # Convert nats to bits
    return bits_per_token * tokens_per_byte


def bits_per_byte_to_loss(bpb: float, tokens_per_byte: float = TOKENS_PER_BYTE) -> float:
    """Convert bits-per-byte to loss (nats)."""
    bits_per_token = bpb / tokens_per_byte
    return bits_per_token * math.log(2)


def calculate_all_metrics(loss: float) -> dict:
    """Calculate all metrics from a loss value."""
    perplexity = loss_to_perplexity(loss)
    bpb = loss_to_bits_per_byte(loss)
    vocab_fraction = perplexity / VOCAB_SIZE

    return {
        "loss": loss,
        "perplexity": perplexity,
        "bits_per_byte": bpb,
        "vocab_fraction": vocab_fraction,
        "effective_choices": perplexity,
    }


def format_metrics(metrics: dict) -> str:
    """Format metrics for display."""
    lines = [
        f"Loss:           {metrics['loss']:.3f} nats",
        f"Perplexity:     {metrics['perplexity']:,.1f}",
        f"Bits per byte:  {metrics['bits_per_byte']:.3f}",
        "",
        f"Interpretation:",
        f"  Model is choosing between ~{metrics['effective_choices']:,.0f} equally likely tokens",
        f"  That's {metrics['vocab_fraction']*100:.2f}% of the {VOCAB_SIZE:,} token vocabulary",
    ]
    return "\n".join(lines)


def describe_quality(loss: float) -> str:
    """Describe what a loss value means in practical terms."""
    if loss >= 10.0:
        return "Random - model hasn't learned anything"
    elif loss >= 8.0:
        return "Very early training - basic patterns emerging"
    elif loss >= 6.5:
        return "Early training - learning language structure"
    elif loss >= 5.5:
        return "Good base model - solid language understanding"
    elif loss >= 4.5:
        return "Strong model - good for chat/tasks"
    elif loss >= 3.5:
        return "Very strong - high quality responses"
    elif loss >= 2.5:
        return "Excellent - near state-of-the-art quality"
    else:
        return "Exceptional - possibly overfitting"


def show_training_journey():
    """Display loss progression through training stages."""
    print("Training Journey: How Loss Decreases")
    print("=" * 60)
    print()

    stages = [
        (11.1, "Untrained", "Random guessing over vocabulary"),
        (9.0, "Step ~100", "Learning token frequencies"),
        (8.0, "Step ~500", "Learning common patterns"),
        (7.0, "Step ~2000", "Understanding basic grammar"),
        (6.0, "Base done", "Solid language model"),
        (5.5, "Mid done", "Adapted to chat format"),
        (5.0, "SFT done", "Helpful assistant behaviour"),
        (4.5, "RL done", "Task-specific optimisation"),
    ]

    # Header
    print(f"{'Stage':<15} {'Loss':>7} {'Perplexity':>12} {'% Vocab':>10} {'Bits/B':>8}")
    print("─" * 60)

    for loss, stage, description in stages:
        metrics = calculate_all_metrics(loss)
        print(
            f"{stage:<15} "
            f"{loss:>7.1f} "
            f"{metrics['perplexity']:>12,.0f} "
            f"{metrics['vocab_fraction']*100:>9.1f}% "
            f"{metrics['bits_per_byte']:>8.2f}"
        )

    print()
    print("Key Insights:")
    print("─" * 60)
    print("• Loss = -log(P(correct_token)) - measures 'surprise'")
    print("• Perplexity = e^loss - 'how many tokens seem equally likely'")
    print("• Bits/byte - compression efficiency (lower = better understanding)")
    print()
    print("• Untrained: perplexity ≈ vocab_size (random guessing)")
    print("• Trained: perplexity drops as model learns patterns")
    print("• Perfect: perplexity = 1 (impossible for natural language)")


def show_comparison_table():
    """Show comparison of different loss values."""
    print("\nLoss Value Reference")
    print("=" * 70)
    print()

    losses = [11.1, 8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]

    print(f"{'Loss':>6} │ {'Perplexity':>12} │ {'Bits/Byte':>10} │ {'Quality':<25}")
    print("─" * 70)

    for loss in losses:
        metrics = calculate_all_metrics(loss)
        quality = describe_quality(loss)
        print(
            f"{loss:>6.1f} │ "
            f"{metrics['perplexity']:>12,.0f} │ "
            f"{metrics['bits_per_byte']:>10.2f} │ "
            f"{quality:<25}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Convert between loss, perplexity, and bits-per-byte",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python 06_loss_calculator.py --loss 6.0
  python 06_loss_calculator.py --perplexity 403
  python 06_loss_calculator.py --bpb 2.6
  python 06_loss_calculator.py --compare

The three metrics are related:
  loss (nats) = log(perplexity)
  bits/byte ≈ loss × 0.43 × tokens_per_byte
        """
    )
    parser.add_argument(
        "--loss", "-l",
        type=float,
        help="Loss value in nats"
    )
    parser.add_argument(
        "--perplexity", "-p",
        type=float,
        help="Perplexity value"
    )
    parser.add_argument(
        "--bpb", "-b",
        type=float,
        help="Bits-per-byte value"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Show comparison table of different loss values"
    )
    args = parser.parse_args()

    # Calculate from whichever input was provided
    if args.loss is not None:
        loss = args.loss
    elif args.perplexity is not None:
        loss = perplexity_to_loss(args.perplexity)
    elif args.bpb is not None:
        loss = bits_per_byte_to_loss(args.bpb)
    elif args.compare:
        show_comparison_table()
        return 0
    else:
        # Default: show training journey
        show_training_journey()
        print()
        show_comparison_table()
        print()
        print("Usage: --loss 6.0, --perplexity 403, or --bpb 2.6")
        return 0

    # Display results
    metrics = calculate_all_metrics(loss)
    print()
    print(format_metrics(metrics))
    print()
    print(f"Quality: {describe_quality(loss)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
