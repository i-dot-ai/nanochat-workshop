"""
Evaluation suite for comparing pretrained and finetuned checkpoints.

This script runs a comprehensive evaluation across multiple tasks and generates
detailed results for comparison.

Usage:
    # Evaluate pretrained model
    python -m workshop.04_eval_finetune.eval.run_eval --source pretrained

    # Evaluate finetuned model
    python -m workshop.04_eval_finetune.eval.run_eval --source finetuned/my_model

    # Evaluate all available models
    python -m workshop.04_eval_finetune.eval.run_eval --all

    # Quick evaluation (subset of problems)
    python -m workshop.04_eval_finetune.eval.run_eval --all --quick

    # Evaluate specific tasks
    python -m workshop.04_eval_finetune.eval.run_eval --source pretrained --tasks "ARC-Easy|MMLU"
"""

import argparse
import json
import os
import re
import sys
import time
import glob as glob_module
from datetime import datetime
from functools import partial
from contextlib import nullcontext
from typing import Dict, List, Optional, Any

import torch
import torch.distributed as dist

from nanochat.common import compute_init, compute_cleanup, get_dist_info, print0, autodetect_device_type
from nanochat.checkpoint_manager import load_checkpoint
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.engine import Engine

from tasks.humaneval import HumanEval
from tasks.mmlu import MMLU
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.spellingbee import SpellingBee


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_models_dir():
    """Get the models directory for this workshop."""
    return os.path.join(get_workshop_dir(), "models")


def get_results_dir():
    """Get/create the results directory."""
    results_dir = os.path.join(get_workshop_dir(), "results")
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def list_available_models():
    """List all available models in the workshop."""
    models = []
    models_dir = get_models_dir()

    # Check pretrained
    pretrained_path = os.path.join(models_dir, "pretrained", "chatsft_checkpoints")
    if os.path.exists(pretrained_path):
        models.append("pretrained")

    # Check finetuned
    finetuned_dir = os.path.join(models_dir, "finetuned")
    if os.path.exists(finetuned_dir):
        for tag in os.listdir(finetuned_dir):
            tag_path = os.path.join(finetuned_dir, tag)
            if os.path.isdir(tag_path):
                # Check if it has model files
                if glob_module.glob(os.path.join(tag_path, "model_*.pt")):
                    models.append(f"finetuned/{tag}")

    return models


def load_model_local(model_type: str, device, phase="eval"):
    """
    Load model from local workshop directory.

    model_type: "pretrained" or "finetuned/<tag>"
    """
    models_dir = get_models_dir()

    if model_type == "pretrained":
        checkpoints_dir = os.path.join(models_dir, "pretrained", "chatsft_checkpoints")
        tokenizer_dir = os.path.join(models_dir, "pretrained", "tokenizer")
    elif model_type.startswith("finetuned/"):
        tag = model_type.split("/", 1)[1]
        checkpoints_dir = os.path.join(models_dir, "finetuned", tag)
        tokenizer_dir = os.path.join(models_dir, "pretrained", "tokenizer")
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'pretrained' or 'finetuned/<tag>'")

    # Find model tag (for pretrained, it's d32)
    if model_type == "pretrained":
        if not os.path.exists(checkpoints_dir):
            raise FileNotFoundError(f"Pretrained model not found at {checkpoints_dir}. Run 'make download' first.")
        model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
        if not model_tags:
            raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
        # Pick largest model
        candidates = []
        for tag in model_tags:
            match = re.match(r"d(\d+)", tag)
            if match:
                candidates.append((int(match.group(1)), tag))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            model_tag = candidates[0][1]
        else:
            model_tag = model_tags[0]
        checkpoint_dir = os.path.join(checkpoints_dir, model_tag)
    else:
        # For finetuned models, checkpoints_dir is already the checkpoint dir
        checkpoint_dir = checkpoints_dir
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(f"Finetuned model not found at {checkpoint_dir}")

    # Find latest step
    checkpoint_files = glob_module.glob(os.path.join(checkpoint_dir, "model_*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))

    print0(f"Loading model from {checkpoint_dir} step {step}...")

    # Load checkpoint
    model_data, _, meta_data = load_checkpoint(checkpoint_dir, step, device, load_optimizer=False)

    # Convert bfloat16 to float for CPU/MPS
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: v.float() if v.dtype == torch.bfloat16 else v
            for k, v in model_data.items()
        }

    # Fix torch compile prefix
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}

    # Build model
    model_config_kwargs = meta_data["model_config"]
    model_config = GPTConfig(**model_config_kwargs)
    with torch.device("meta"):
        model = GPT(model_config)
    model.to_empty(device=device)
    model.init_weights()
    model.load_state_dict(model_data, strict=True, assign=True)
    model.eval()

    # Load tokenizer
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

    return model, tokenizer, meta_data


# Task configuration
ALL_TASKS = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
QUICK_TASKS = ['ARC-Easy', 'MMLU']  # Subset for quick evaluation

BASELINE_ACCURACIES = {
    'ARC-Easy': 0.25,      # multiple choice 1 of 4 => 25%
    'ARC-Challenge': 0.25,  # multiple choice 1 of 4 => 25%
    'MMLU': 0.25,           # multiple choice 1 of 4 => 25%
    'GSM8K': 0.0,           # open-ended => 0%
    'HumanEval': 0.0,       # open-ended => 0%
    'SpellingBee': 0.0,     # open-ended => 0%
}

TASK_MODULES = {
    'HumanEval': HumanEval,
    'MMLU': partial(MMLU, subset="all", split="test"),
    'ARC-Easy': partial(ARC, subset="ARC-Easy", split="test"),
    'ARC-Challenge': partial(ARC, subset="ARC-Challenge", split="test"),
    'GSM8K': partial(GSM8K, subset="main", split="test"),
    'SpellingBee': partial(SpellingBee, size=256, split="test"),
}


def run_generative_eval(task_object, tokenizer, model, engine, num_samples, max_new_tokens,
                        temperature, top_k, max_problems=None, collect_failures=False):
    """Run generative evaluation (sampling-based)."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)

    num_passed, total = 0, 0
    failures = []

    for i in range(ddp_rank, num_problems, ddp_world_size):
        conversation = task_object[i]
        encoded_prompt = tokenizer.render_for_completion(conversation)

        results, _ = engine.generate_batch(
            encoded_prompt,
            num_samples=num_samples,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        prefix_length = len(encoded_prompt)
        completions = [tokenizer.decode(result_tokens[prefix_length:]) for result_tokens in results]
        outcomes = [task_object.evaluate(conversation, completion) for completion in completions]
        passed = any(outcomes)

        total += 1
        num_passed += int(passed)

        # Collect failures for analysis
        if collect_failures and not passed:
            failures.append({
                'index': i,
                'conversation': conversation,
                'completions': completions,
                'expected': conversation.get('answer', conversation.get('expected', 'N/A')),
            })

        print(f"\r\033[KRank {ddp_rank} | {num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

    # Aggregate across ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    accuracy = num_passed / total if total > 0 else 0.0
    return accuracy, failures


def run_categorical_eval(task_object, tokenizer, model, batch_size, max_problems=None, collect_failures=False):
    """Run categorical evaluation (logit-based)."""
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    device = model.get_device()
    bos = tokenizer.get_bos_token_id()

    num_problems = len(task_object) if max_problems is None else min(len(task_object), max_problems)
    ceil_div = lambda x, y: -(-x // y)
    num_batches = ceil_div(num_problems, batch_size)

    letter_to_id_cache = {}
    num_passed, total = 0, 0
    failures = []

    for i in range(ddp_rank, num_batches, ddp_world_size):
        i0, i1 = i * batch_size, min((i + 1) * batch_size, num_problems)

        conversations = [task_object[ii] for ii in range(i0, i1)]
        prompt_ids = [tokenizer.render_for_completion(conversation) for conversation in conversations]
        max_length = max(len(ids) for ids in prompt_ids)
        answer_time_positions = [len(ids) - 1 for ids in prompt_ids]
        padded_prompt_ids = [ids + [bos] * (max_length - len(ids)) for ids in prompt_ids]
        prompt_ids = torch.tensor(padded_prompt_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(prompt_ids)

        for idx, conversation in enumerate(conversations):
            letters = conversation['letters']
            letter_ids = []
            for letter in letters:
                if letter not in letter_to_id_cache:
                    encoded_letter = tokenizer.encode(letter)
                    assert len(encoded_letter) == 1, "Each letter must be a single token"
                    letter_to_id_cache[letter] = encoded_letter[0]
                letter_ids.append(letter_to_id_cache[letter])

            answer_pos = answer_time_positions[idx]
            focus_logits = logits[idx, answer_pos, letter_ids]
            argmax_letter_id = focus_logits.argmax(dim=-1).item()
            predicted_letter = letters[argmax_letter_id]
            outcome = task_object.evaluate(conversation, predicted_letter)

            num_passed += int(outcome)
            total += 1

            # Collect failures
            if collect_failures and not outcome:
                failures.append({
                    'index': i0 + idx,
                    'question': conversation.get('question', str(conversation.get('messages', []))[:200]),
                    'predicted': predicted_letter,
                    'expected': conversation.get('answer', 'N/A'),
                    'choices': conversation.get('choices', []),
                })

        # Progress update
        if total > 0:
            print(f"\r\033[K{num_passed}/{total} ({100*num_passed/total:.2f}%)", end='', flush=True)

    print()

    # Aggregate across ranks
    if ddp:
        num_passed_tensor = torch.tensor([num_passed], dtype=torch.long, device=device)
        total_tensor = torch.tensor([total], dtype=torch.long, device=device)
        dist.all_reduce(num_passed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        num_passed = num_passed_tensor.item()
        total = total_tensor.item()

    accuracy = num_passed / total if total > 0 else 0.0
    return accuracy, failures


def evaluate_task(task_name: str, model, tokenizer, engine,
                  batch_size: int = 8, num_samples: int = 1,
                  max_new_tokens: int = 512, temperature: float = 0.0,
                  top_k: int = 50, max_problems: Optional[int] = None,
                  collect_failures: bool = False) -> Dict[str, Any]:
    """Evaluate a single task and return results."""

    task_module = TASK_MODULES.get(task_name)
    if task_module is None:
        raise ValueError(f"Unknown task: {task_name}")

    task_object = task_module()
    start_time = time.time()

    if task_object.eval_type == 'generative':
        accuracy, failures = run_generative_eval(
            task_object, tokenizer, model, engine,
            num_samples, max_new_tokens, temperature, top_k,
            max_problems=max_problems, collect_failures=collect_failures
        )
    elif task_object.eval_type == 'categorical':
        accuracy, failures = run_categorical_eval(
            task_object, tokenizer, model, batch_size,
            max_problems=max_problems, collect_failures=collect_failures
        )
    else:
        raise ValueError(f"Unsupported eval type: {task_object.eval_type}")

    elapsed = time.time() - start_time
    baseline = BASELINE_ACCURACIES.get(task_name, 0.0)
    centered_acc = (accuracy - baseline) / (1.0 - baseline) if baseline < 1.0 else 0.0

    return {
        'task': task_name,
        'accuracy': accuracy,
        'accuracy_pct': accuracy * 100,
        'baseline': baseline,
        'centered_accuracy': centered_acc,
        'num_problems': max_problems or len(task_object),
        'elapsed_seconds': elapsed,
        'eval_type': task_object.eval_type,
        'failures': failures if collect_failures else [],
    }


def evaluate_checkpoint(source: str, task_names: List[str], device,
                        autocast_ctx, args, collect_failures: bool = False) -> Dict[str, Any]:
    """Evaluate a single checkpoint on multiple tasks."""

    print0(f"\n{'='*60}")
    print0(f"Evaluating: {source}")
    print0(f"{'='*60}")

    try:
        model, tokenizer, meta = load_model_local(source, device, phase="eval")
        engine = Engine(model, tokenizer)
    except Exception as e:
        print0(f"Failed to load {source}: {e}")
        return {'source': source, 'error': str(e), 'results': {}}

    results = {}
    total_start = time.time()

    for task_name in task_names:
        print0(f"\n--- {task_name} ---")
        try:
            with autocast_ctx:
                task_result = evaluate_task(
                    task_name, model, tokenizer, engine,
                    batch_size=args.batch_size,
                    num_samples=args.num_samples,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    max_problems=args.max_problems,
                    collect_failures=collect_failures,
                )
                results[task_name] = task_result
                print0(f"{task_name}: {task_result['accuracy_pct']:.2f}% (centered: {task_result['centered_accuracy']:.3f})")
        except Exception as e:
            print0(f"Error evaluating {task_name}: {e}")
            import traceback
            traceback.print_exc()
            results[task_name] = {'task': task_name, 'error': str(e)}

    # Calculate aggregate metrics
    valid_results = [r for r in results.values() if 'accuracy' in r]
    if valid_results:
        mean_accuracy = sum(r['accuracy'] for r in valid_results) / len(valid_results)
        mean_centered = sum(r['centered_accuracy'] for r in valid_results) / len(valid_results)
    else:
        mean_accuracy = 0.0
        mean_centered = 0.0

    total_elapsed = time.time() - total_start

    return {
        'source': source,
        'results': results,
        'mean_accuracy': mean_accuracy,
        'mean_centered_accuracy': mean_centered,
        'chatcore': mean_centered,  # ChatCORE is mean centered accuracy
        'total_elapsed_seconds': total_elapsed,
        'timestamp': datetime.now().isoformat(),
    }


def save_results(all_results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print0(f"\nResults saved to: {output_path}")


def print_comparison_table(all_results: Dict[str, Dict]):
    """Print a comparison table of results."""

    sources = list(all_results.keys())
    if not sources:
        return

    # Get all tasks
    all_tasks = set()
    for source_result in all_results.values():
        if 'results' in source_result:
            all_tasks.update(source_result['results'].keys())
    all_tasks = sorted(all_tasks)

    # Print header
    print0("\n" + "=" * 80)
    print0("EVALUATION RESULTS")
    print0("=" * 80)

    # Header row
    header = f"{'Task':<20}"
    for source in sources:
        header += f"{source:>20}"
    print0(header)
    print0("-" * 80)

    # Task rows
    for task in all_tasks:
        row = f"{task:<20}"
        for source in sources:
            source_result = all_results.get(source, {})
            task_result = source_result.get('results', {}).get(task, {})
            if 'accuracy_pct' in task_result:
                row += f"{task_result['accuracy_pct']:>19.2f}%"
            elif 'error' in task_result:
                row += f"{'ERROR':>20}"
            else:
                row += f"{'N/A':>20}"
        print0(row)

    # Summary row
    print0("-" * 80)
    row = f"{'ChatCORE':<20}"
    for source in sources:
        source_result = all_results.get(source, {})
        chatcore = source_result.get('chatcore', 0.0)
        row += f"{chatcore:>20.4f}"
    print0(row)

    row = f"{'Mean Accuracy':<20}"
    for source in sources:
        source_result = all_results.get(source, {})
        mean_acc = source_result.get('mean_accuracy', 0.0) * 100
        row += f"{mean_acc:>19.2f}%"
    print0(row)

    print0("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate nanochat checkpoints')

    # Source selection
    parser.add_argument('--source', type=str, default=None,
                        help="Model to evaluate: pretrained or finetuned/<tag>")
    parser.add_argument('--all', action='store_true',
                        help="Evaluate all available models")

    # Task selection
    parser.add_argument('--tasks', type=str, default=None,
                        help="Tasks to evaluate (pipe-separated). Default: all tasks")
    parser.add_argument('--quick', action='store_true',
                        help="Quick evaluation with subset of tasks and problems")

    # Evaluation parameters
    parser.add_argument('-d', '--dtype', type=str, default='bfloat16',
                        choices=['float32', 'bfloat16'])
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-x', '--max-problems', type=int, default=None,
                        help='Max problems per task (for quick testing)')

    # Output
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file path')
    parser.add_argument('--collect-failures', action='store_true',
                        help='Collect failure cases for analysis')

    # Device
    parser.add_argument('--device-type', type=str, default='',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device type (empty=autodetect)')

    args = parser.parse_args()

    # List available models
    available_models = list_available_models()
    print0(f"Available models: {available_models}")

    if not available_models:
        print0("No models found. Run 'make download' to download the pretrained model.")
        sys.exit(1)

    # Determine sources to evaluate
    if args.all:
        sources = available_models
    elif args.source:
        if args.source not in available_models:
            print0(f"Model '{args.source}' not found. Available: {available_models}")
            sys.exit(1)
        sources = [args.source]
    else:
        # Default to pretrained if available
        if 'pretrained' in available_models:
            sources = ['pretrained']
        else:
            sources = available_models[:1]
        print0(f"No --source specified, using: {sources}")

    # Determine tasks
    if args.tasks:
        task_names = args.tasks.split('|')
    elif args.quick:
        task_names = QUICK_TASKS
        if args.max_problems is None:
            args.max_problems = 50  # Quick mode: limit problems
    else:
        task_names = ALL_TASKS

    # Initialize
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    ptdtype = torch.float32 if args.dtype == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    print0(f"\nEvaluation Configuration:")
    print0(f"  Models: {sources}")
    print0(f"  Tasks: {task_names}")
    print0(f"  Max problems: {args.max_problems or 'all'}")
    print0(f"  Device: {device}")

    # Run evaluations
    all_results = {}
    for source in sources:
        result = evaluate_checkpoint(
            source, task_names, device, autocast_ctx, args,
            collect_failures=args.collect_failures
        )
        all_results[source] = result

    # Print comparison
    print_comparison_table(all_results)

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(get_results_dir(), f"eval_results_{timestamp}.json")

    save_results(all_results, output_path)

    compute_cleanup()

    return all_results


if __name__ == "__main__":
    main()
