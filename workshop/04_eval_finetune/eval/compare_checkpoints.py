"""
Compare evaluation results across BASE, MID, and SFT checkpoints.

This script loads previous evaluation results and generates comparison reports.

Usage:
    # Compare from JSON results file
    python -m workshop.04_eval_finetune.eval.compare_checkpoints --results results/eval_results.json

    # Compare multiple result files
    python -m workshop.04_eval_finetune.eval.compare_checkpoints --results file1.json file2.json
"""

import argparse
import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def format_percentage(value: float, precision: int = 2) -> str:
    """Format a decimal as percentage string."""
    return f"{value * 100:.{precision}f}%"


def format_delta(current: float, baseline: float, precision: int = 2) -> str:
    """Format delta between two values."""
    delta = (current - baseline) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{precision}f}%"


def generate_comparison_table(results: Dict[str, Dict], task_order: Optional[List[str]] = None) -> str:
    """Generate a markdown comparison table."""

    sources = list(results.keys())
    if not sources:
        return "No results to compare."

    # Collect all tasks
    all_tasks = set()
    for source_result in results.values():
        if 'results' in source_result:
            all_tasks.update(source_result['results'].keys())

    if task_order:
        all_tasks = [t for t in task_order if t in all_tasks]
    else:
        all_tasks = sorted(all_tasks)

    lines = []

    # Header
    header = "| Task |"
    separator = "|------|"
    for source in sources:
        header += f" {source.upper()} |"
        separator += "------:|"

    # Add delta columns if we have base to compare against
    if 'base' in sources and len(sources) > 1:
        for source in sources:
            if source != 'base':
                header += f" Δ vs BASE |"
                separator += "----------:|"

    lines.append(header)
    lines.append(separator)

    # Task rows
    for task in all_tasks:
        row = f"| {task} |"
        base_acc = None

        # Accuracy columns
        for source in sources:
            source_result = results.get(source, {})
            task_result = source_result.get('results', {}).get(task, {})

            if 'accuracy' in task_result:
                acc = task_result['accuracy']
                row += f" {format_percentage(acc)} |"
                if source == 'base':
                    base_acc = acc
            elif 'error' in task_result:
                row += " ERROR |"
            else:
                row += " N/A |"

        # Delta columns
        if 'base' in sources and len(sources) > 1 and base_acc is not None:
            for source in sources:
                if source != 'base':
                    source_result = results.get(source, {})
                    task_result = source_result.get('results', {}).get(task, {})
                    if 'accuracy' in task_result:
                        row += f" {format_delta(task_result['accuracy'], base_acc)} |"
                    else:
                        row += " - |"

        lines.append(row)

    # Summary rows
    lines.append("|" + "-" * 6 + "|" + "-" * 7 * len(sources) + "|")

    # Mean accuracy
    row = "| **Mean Acc** |"
    base_mean = None
    for source in sources:
        source_result = results.get(source, {})
        mean_acc = source_result.get('mean_accuracy', 0.0)
        row += f" **{format_percentage(mean_acc)}** |"
        if source == 'base':
            base_mean = mean_acc

    if 'base' in sources and len(sources) > 1 and base_mean is not None:
        for source in sources:
            if source != 'base':
                source_result = results.get(source, {})
                mean_acc = source_result.get('mean_accuracy', 0.0)
                row += f" {format_delta(mean_acc, base_mean)} |"
    lines.append(row)

    # ChatCORE
    row = "| **ChatCORE** |"
    base_core = None
    for source in sources:
        source_result = results.get(source, {})
        chatcore = source_result.get('chatcore', 0.0)
        row += f" **{chatcore:.4f}** |"
        if source == 'base':
            base_core = chatcore

    if 'base' in sources and len(sources) > 1 and base_core is not None:
        for source in sources:
            if source != 'base':
                source_result = results.get(source, {})
                chatcore = source_result.get('chatcore', 0.0)
                delta = chatcore - base_core
                sign = "+" if delta >= 0 else ""
                row += f" {sign}{delta:.4f} |"
    lines.append(row)

    return "\n".join(lines)


def generate_task_analysis(results: Dict[str, Dict]) -> str:
    """Generate per-task analysis."""

    sources = list(results.keys())
    if not sources:
        return ""

    # Collect all tasks
    all_tasks = set()
    for source_result in results.values():
        if 'results' in source_result:
            all_tasks.update(source_result['results'].keys())
    all_tasks = sorted(all_tasks)

    lines = []

    for task in all_tasks:
        lines.append(f"\n### {task}\n")

        # Collect accuracies across sources
        task_results = []
        for source in sources:
            source_result = results.get(source, {})
            task_result = source_result.get('results', {}).get(task, {})
            if 'accuracy' in task_result:
                task_results.append({
                    'source': source,
                    'accuracy': task_result['accuracy'],
                    'centered': task_result.get('centered_accuracy', 0),
                    'eval_type': task_result.get('eval_type', 'unknown'),
                    'elapsed': task_result.get('elapsed_seconds', 0),
                })

        if not task_results:
            lines.append("No results available for this task.")
            continue

        # Find best performer
        best = max(task_results, key=lambda x: x['accuracy'])
        worst = min(task_results, key=lambda x: x['accuracy'])

        lines.append(f"- **Evaluation type**: {task_results[0]['eval_type']}")
        lines.append(f"- **Best performer**: {best['source'].upper()} ({format_percentage(best['accuracy'])})")

        if len(task_results) > 1:
            lines.append(f"- **Improvement range**: {format_percentage(worst['accuracy'])} → {format_percentage(best['accuracy'])}")

        # Performance progression
        if len(task_results) > 1:
            lines.append("\n**Progression**:")
            for tr in task_results:
                bar_length = int(tr['accuracy'] * 40)
                bar = "█" * bar_length + "░" * (40 - bar_length)
                lines.append(f"- {tr['source'].upper():>5}: {bar} {format_percentage(tr['accuracy'])}")

    return "\n".join(lines)


def generate_insights(results: Dict[str, Dict]) -> str:
    """Generate insights from the comparison."""

    sources = list(results.keys())
    if len(sources) < 2:
        return "Need at least 2 checkpoints to generate insights."

    lines = []

    # Overall improvement
    if 'base' in sources and 'sft' in sources:
        base_core = results['base'].get('chatcore', 0)
        sft_core = results['sft'].get('chatcore', 0)
        improvement = sft_core - base_core

        lines.append("## Key Insights\n")

        if improvement > 0:
            lines.append(f"1. **Overall improvement**: SFT shows +{improvement:.4f} ChatCORE improvement over BASE")
        else:
            lines.append(f"1. **Performance note**: SFT ChatCORE is {improvement:.4f} compared to BASE")

    # Task-specific insights
    if 'base' in results and 'sft' in results:
        base_results = results['base'].get('results', {})
        sft_results = results['sft'].get('results', {})

        improvements = []
        regressions = []

        for task in base_results:
            if task in sft_results:
                base_acc = base_results[task].get('accuracy', 0)
                sft_acc = sft_results[task].get('accuracy', 0)
                delta = sft_acc - base_acc

                if delta > 0.05:  # >5% improvement
                    improvements.append((task, delta))
                elif delta < -0.05:  # >5% regression
                    regressions.append((task, delta))

        if improvements:
            improvements.sort(key=lambda x: x[1], reverse=True)
            lines.append(f"\n2. **Biggest improvements**:")
            for task, delta in improvements[:3]:
                lines.append(f"   - {task}: {format_delta(delta, 0)}")

        if regressions:
            regressions.sort(key=lambda x: x[1])
            lines.append(f"\n3. **Areas needing attention**:")
            for task, delta in regressions[:3]:
                lines.append(f"   - {task}: {format_delta(delta, 0)}")

    # MID phase analysis
    if 'mid' in sources:
        mid_core = results['mid'].get('chatcore', 0)
        if 'base' in sources:
            base_core = results['base'].get('chatcore', 0)
            mid_improvement = mid_core - base_core
            lines.append(f"\n4. **MID phase contribution**: +{mid_improvement:.4f} ChatCORE from midtraining")

    return "\n".join(lines)


def generate_report(results: Dict[str, Dict], output_path: Optional[str] = None) -> str:
    """Generate a complete comparison report in markdown format."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# NanoChat Checkpoint Comparison Report

Generated: {timestamp}

## Summary

This report compares evaluation results across different training phases:
- **BASE**: Initial pretrained model
- **MID**: Midtraining phase (continued training on task mixtures)
- **SFT**: Supervised fine-tuning (instruction tuning)

## Results Comparison

{generate_comparison_table(results)}

{generate_insights(results)}

## Per-Task Analysis

{generate_task_analysis(results)}

## Methodology

- **Categorical tasks** (MMLU, ARC): Evaluated using logit comparison across answer choices
- **Generative tasks** (GSM8K, HumanEval, SpellingBee): Evaluated using sampling and answer extraction
- **ChatCORE metric**: Mean centered accuracy across all tasks, normalized by baseline (random) performance
- **Centered accuracy**: `(accuracy - baseline) / (1 - baseline)` where baseline is random chance

## Notes

- Multiple choice tasks have 25% random baseline
- Open-ended tasks have 0% baseline
- Higher ChatCORE indicates better overall performance relative to random baseline
"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Compare checkpoint evaluation results')
    parser.add_argument('--results', type=str, nargs='+', required=True,
                        help='JSON result file(s) to compare')
    parser.add_argument('--output', type=str, default=None,
                        help='Output markdown report path')
    args = parser.parse_args()

    # Load and merge results
    all_results = {}
    for filepath in args.results:
        results = load_results(filepath)
        all_results.update(results)

    # Generate report
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(get_workshop_dir(), "results", "comparison_report.md")

    report = generate_report(all_results, output_path)
    print(report)


if __name__ == "__main__":
    main()
