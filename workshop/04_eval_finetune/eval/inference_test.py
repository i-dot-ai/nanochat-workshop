"""
Inference tests for validating model behavior in CLI and web UI scenarios.

This script runs a set of test prompts through the model to validate:
- Basic response generation
- Instruction following
- Multi-turn conversation handling
- Edge cases and error handling

Usage:
    # Test pretrained model
    python -m workshop.04_eval_finetune.eval.inference_test --source sft

    # Test with specific prompts
    python -m workshop.04_eval_finetune.eval.inference_test --source sft --test-file tests.json

    # Quick smoke test
    python -m workshop.04_eval_finetune.eval.inference_test --source sft --quick
"""

import argparse
import json
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime

import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class TestCase:
    """A single test case for inference testing."""
    name: str
    messages: List[Dict[str, str]]
    expected_contains: List[str] = field(default_factory=list)
    expected_not_contains: List[str] = field(default_factory=list)
    max_tokens: int = 256
    temperature: float = 0.0
    category: str = "general"


# Default test cases
DEFAULT_TESTS = [
    # Basic functionality
    TestCase(
        name="simple_greeting",
        messages=[{"role": "user", "content": "Hello!"}],
        expected_contains=[],  # Just check it responds
        category="basic",
    ),
    TestCase(
        name="simple_question",
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        expected_contains=["4"],
        category="basic",
    ),
    TestCase(
        name="factual_question",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        expected_contains=["Paris"],
        category="basic",
    ),

    # Instruction following
    TestCase(
        name="list_format",
        messages=[{"role": "user", "content": "List 3 colors."}],
        expected_contains=[],  # Should produce a list
        category="instruction",
    ),
    TestCase(
        name="word_count",
        messages=[{"role": "user", "content": "Respond with exactly one word: yes or no. Is the sky blue?"}],
        max_tokens=10,
        category="instruction",
    ),
    TestCase(
        name="code_generation",
        messages=[{"role": "user", "content": "Write a Python function that adds two numbers."}],
        expected_contains=["def", "return"],
        category="instruction",
    ),

    # Math reasoning
    TestCase(
        name="simple_math",
        messages=[{"role": "user", "content": "What is 15 * 7?"}],
        expected_contains=["105"],
        category="math",
    ),
    TestCase(
        name="word_problem",
        messages=[{"role": "user", "content": "If I have 5 apples and buy 3 more, how many do I have?"}],
        expected_contains=["8"],
        category="math",
    ),

    # Multi-turn conversation
    TestCase(
        name="multi_turn_context",
        messages=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ],
        expected_contains=["Alice"],
        category="multi_turn",
    ),
    TestCase(
        name="multi_turn_math",
        messages=[
            {"role": "user", "content": "I have 10 apples."},
            {"role": "assistant", "content": "Okay, you have 10 apples."},
            {"role": "user", "content": "I eat 3. How many do I have left?"},
        ],
        expected_contains=["7"],
        category="multi_turn",
    ),

    # Edge cases
    TestCase(
        name="empty_response_handling",
        messages=[{"role": "user", "content": "Say nothing."}],
        max_tokens=50,
        category="edge_case",
    ),
    TestCase(
        name="long_input",
        messages=[{"role": "user", "content": "Summarize this: " + "The quick brown fox jumps over the lazy dog. " * 20}],
        max_tokens=100,
        category="edge_case",
    ),
    TestCase(
        name="special_characters",
        messages=[{"role": "user", "content": "What does @#$%^& mean?"}],
        category="edge_case",
    ),
]

QUICK_TESTS = [t for t in DEFAULT_TESTS if t.category == "basic"]


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    response: str
    elapsed_seconds: float
    expected_contains: List[str]
    found_contains: List[str]
    missing_contains: List[str]
    unexpected_contains: List[str]
    error: Optional[str] = None


def run_inference(model, tokenizer, engine, messages: List[Dict[str, str]],
                  max_tokens: int = 256, temperature: float = 0.0,
                  top_k: int = 50, autocast_ctx=None) -> str:
    """Run inference and return the response text."""

    # Build conversation tokens
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<|user_start|>")
    user_end = tokenizer.encode_special("<|user_end|>")
    assistant_start = tokenizer.encode_special("<|assistant_start|>")
    assistant_end = tokenizer.encode_special("<|assistant_end|>")

    conversation_tokens = [bos]
    for message in messages:
        if message["role"] == "user":
            conversation_tokens.append(user_start)
            conversation_tokens.extend(tokenizer.encode(message["content"]))
            conversation_tokens.append(user_end)
        elif message["role"] == "assistant":
            conversation_tokens.append(assistant_start)
            conversation_tokens.extend(tokenizer.encode(message["content"]))
            conversation_tokens.append(assistant_end)

    conversation_tokens.append(assistant_start)

    # Generate response
    ctx = autocast_ctx if autocast_ctx else nullcontext()
    response_tokens = []

    with ctx:
        for token_column, token_masks in engine.generate(
            conversation_tokens,
            num_samples=1,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        ):
            token = token_column[0]
            if token == assistant_end or token == bos:
                break
            response_tokens.append(token)

    response = tokenizer.decode(response_tokens)
    return response


def run_test(test: TestCase, model, tokenizer, engine, autocast_ctx) -> TestResult:
    """Run a single test case."""

    start_time = time.time()
    error = None
    response = ""

    try:
        response = run_inference(
            model, tokenizer, engine,
            test.messages,
            max_tokens=test.max_tokens,
            temperature=test.temperature,
            autocast_ctx=autocast_ctx,
        )
    except Exception as e:
        error = str(e)

    elapsed = time.time() - start_time

    # Check expected content
    response_lower = response.lower()
    found = []
    missing = []
    unexpected = []

    for expected in test.expected_contains:
        if expected.lower() in response_lower:
            found.append(expected)
        else:
            missing.append(expected)

    for not_expected in test.expected_not_contains:
        if not_expected.lower() in response_lower:
            unexpected.append(not_expected)

    # Determine pass/fail
    passed = (
        error is None and
        len(missing) == 0 and
        len(unexpected) == 0 and
        len(response.strip()) > 0  # Must have some response
    )

    return TestResult(
        name=test.name,
        passed=passed,
        response=response,
        elapsed_seconds=elapsed,
        expected_contains=test.expected_contains,
        found_contains=found,
        missing_contains=missing,
        unexpected_contains=unexpected,
        error=error,
    )


def run_all_tests(tests: List[TestCase], model, tokenizer, engine, autocast_ctx) -> List[TestResult]:
    """Run all test cases and return results."""

    results = []
    for i, test in enumerate(tests):
        print(f"\rRunning test {i+1}/{len(tests)}: {test.name}...", end="", flush=True)
        result = run_test(test, model, tokenizer, engine, autocast_ctx)
        results.append(result)
        status = "✓" if result.passed else "✗"
        print(f"\r{status} Test {i+1}/{len(tests)}: {test.name} ({result.elapsed_seconds:.2f}s)")

    return results


def generate_test_report(results: List[TestResult], source: str, output_path: Optional[str] = None) -> str:
    """Generate a test report."""

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = passed / total if total > 0 else 0

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Inference Test Report",
        f"",
        f"**Source**: {source}",
        f"**Timestamp**: {timestamp}",
        f"**Pass Rate**: {passed}/{total} ({pass_rate*100:.1f}%)",
        f"",
        f"## Summary",
        f"",
        f"| Status | Count |",
        f"|--------|-------|",
        f"| ✓ Passed | {passed} |",
        f"| ✗ Failed | {total - passed} |",
        f"| Total | {total} |",
        f"",
        f"## Detailed Results",
        f"",
    ]

    # Group by category
    categories = {}
    for result in results:
        test = next((t for t in DEFAULT_TESTS if t.name == result.name), None)
        category = test.category if test else "unknown"
        if category not in categories:
            categories[category] = []
        categories[category].append(result)

    for category, cat_results in categories.items():
        cat_passed = sum(1 for r in cat_results if r.passed)
        lines.append(f"### {category.replace('_', ' ').title()} ({cat_passed}/{len(cat_results)})")
        lines.append("")

        for result in cat_results:
            status = "✓" if result.passed else "✗"
            lines.append(f"#### {status} {result.name}")
            lines.append("")

            if result.error:
                lines.append(f"**Error**: {result.error}")
            else:
                lines.append(f"**Response** ({result.elapsed_seconds:.2f}s):")
                lines.append("```")
                lines.append(result.response[:500] + ("..." if len(result.response) > 500 else ""))
                lines.append("```")

                if result.missing_contains:
                    lines.append(f"**Missing expected**: {result.missing_contains}")
                if result.unexpected_contains:
                    lines.append(f"**Unexpected found**: {result.unexpected_contains}")

            lines.append("")

    # Failure summary
    failures = [r for r in results if not r.passed]
    if failures:
        lines.append("## Failures Summary")
        lines.append("")
        lines.append("| Test | Issue |")
        lines.append("|------|-------|")
        for f in failures:
            if f.error:
                issue = f"Error: {f.error[:50]}"
            elif f.missing_contains:
                issue = f"Missing: {f.missing_contains}"
            elif f.unexpected_contains:
                issue = f"Unexpected: {f.unexpected_contains}"
            else:
                issue = "Empty response"
            lines.append(f"| {f.name} | {issue} |")
        lines.append("")

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Run inference tests')
    parser.add_argument('--source', type=str, default='sft',
                        help='Model source: base|mid|sft')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Custom test file (JSON)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick smoke tests only')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path')
    parser.add_argument('-g', '--model-tag', type=str, default=None)
    parser.add_argument('-s', '--step', type=int, default=None)
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16',
                        choices=['float32', 'bfloat16'])
    parser.add_argument('--device-type', type=str, default='',
                        choices=['cuda', 'cpu', 'mps'])
    args = parser.parse_args()

    # Initialize
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    # Load model
    print(f"Loading {args.source} model...")
    model, tokenizer, meta = load_model(args.source, device, phase="eval",
                                        model_tag=args.model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    # Get tests
    if args.test_file:
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
        tests = [TestCase(**t) for t in test_data]
    elif args.quick:
        tests = QUICK_TESTS
    else:
        tests = DEFAULT_TESTS

    print(f"\nRunning {len(tests)} tests on {args.source} model...\n")

    # Run tests
    results = run_all_tests(tests, model, tokenizer, engine, autocast_ctx)

    # Generate report
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(get_workshop_dir(), "results", f"inference_test_{args.source}.md")

    report = generate_test_report(results, args.source, output_path)

    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    print(f"\n{'='*50}")
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*50}")

    compute_cleanup()

    return results


if __name__ == "__main__":
    main()
