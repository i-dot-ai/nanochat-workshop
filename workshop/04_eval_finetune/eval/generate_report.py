"""
Generate comprehensive evaluation report in markdown format.

This script creates a detailed report.md with:
- Evaluation results tables
- Performance comparisons
- Visualizations (ASCII charts)
- Interpretation of results

Usage:
    # Generate from evaluation results
    python -m workshop.04_eval_finetune.eval.generate_report --results results/eval_results.json

    # Generate with custom output path
    python -m workshop.04_eval_finetune.eval.generate_report --results results.json --output report.md
"""

import argparse
import json
import os
import platform
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def get_system_info() -> Dict[str, str]:
    """Collect system information."""
    import torch

    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
    }

    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpu_name'] = torch.cuda.get_device_name(0)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        info['device'] = 'Apple Silicon (MPS)'
    else:
        info['device'] = 'CPU'

    return info


def format_pct(value: float, precision: int = 2) -> str:
    """Format value as percentage."""
    return f"{value * 100:.{precision}f}%"


def create_bar_chart(values: Dict[str, float], max_width: int = 40, title: str = "") -> str:
    """Create an ASCII bar chart."""
    if not values:
        return ""

    lines = []
    if title:
        lines.append(f"\n**{title}**\n")
        lines.append("```")

    max_val = max(values.values()) if values else 1.0
    max_label_len = max(len(k) for k in values.keys())

    for label, value in values.items():
        bar_len = int((value / max_val) * max_width) if max_val > 0 else 0
        bar = "█" * bar_len + "░" * (max_width - bar_len)
        lines.append(f"{label:>{max_label_len}} │ {bar} {format_pct(value)}")

    lines.append("```")
    return "\n".join(lines)


def create_delta_indicator(delta: float) -> str:
    """Create visual delta indicator."""
    if delta > 0.1:
        return "🟢 Strong improvement"
    elif delta > 0.05:
        return "🟡 Moderate improvement"
    elif delta > 0:
        return "⬆️ Slight improvement"
    elif delta > -0.05:
        return "➡️ Similar"
    elif delta > -0.1:
        return "🟡 Slight regression"
    else:
        return "🔴 Significant regression"


def generate_results_table(results: Dict[str, Dict], format_type: str = "markdown") -> str:
    """Generate results table in specified format."""

    sources = list(results.keys())
    if not sources:
        return "No results available."

    # Collect all tasks
    all_tasks = set()
    for source_result in results.values():
        if 'results' in source_result:
            all_tasks.update(source_result['results'].keys())
    all_tasks = sorted(all_tasks)

    lines = []

    # Header
    header = "| Task | Type |"
    sep = "|------|------|"
    for source in sources:
        header += f" {source.upper()} |"
        sep += "------:|"
    lines.append(header)
    lines.append(sep)

    # Task rows
    for task in all_tasks:
        eval_type = "?"
        row = f"| {task} |"

        for source in sources:
            task_result = results.get(source, {}).get('results', {}).get(task, {})
            if 'eval_type' in task_result:
                eval_type = "MC" if task_result['eval_type'] == 'categorical' else "Gen"

        row = f"| {task} | {eval_type} |"

        for source in sources:
            task_result = results.get(source, {}).get('results', {}).get(task, {})
            if 'accuracy' in task_result:
                row += f" {format_pct(task_result['accuracy'])} |"
            else:
                row += " - |"
        lines.append(row)

    # Summary
    lines.append("|------|------|" + "------:|" * len(sources))

    # Mean accuracy row
    row = "| **Mean** | - |"
    for source in sources:
        mean_acc = results.get(source, {}).get('mean_accuracy', 0)
        row += f" **{format_pct(mean_acc)}** |"
    lines.append(row)

    # ChatCORE row
    row = "| **ChatCORE** | - |"
    for source in sources:
        chatcore = results.get(source, {}).get('chatcore', 0)
        row += f" **{chatcore:.4f}** |"
    lines.append(row)

    return "\n".join(lines)


def generate_interpretation(results: Dict[str, Dict]) -> str:
    """Generate human-readable interpretation of results."""

    sources = list(results.keys())
    lines = []

    lines.append("## Results Interpretation\n")

    # Overall assessment
    if 'base' in results and 'sft' in results:
        base_core = results['base'].get('chatcore', 0)
        sft_core = results['sft'].get('chatcore', 0)
        improvement = sft_core - base_core

        lines.append("### Overall Assessment\n")

        if improvement > 0.1:
            lines.append("The SFT model shows **substantial improvement** over the BASE model. "
                        "The training pipeline is working effectively to improve task performance.")
        elif improvement > 0.05:
            lines.append("The SFT model shows **moderate improvement** over the BASE model. "
                        "The fine-tuning process is having a positive impact.")
        elif improvement > 0:
            lines.append("The SFT model shows **slight improvement** over the BASE model. "
                        "Consider adjusting training hyperparameters or adding more data.")
        else:
            lines.append("The SFT model shows **similar or lower** performance compared to BASE. "
                        "This may indicate overfitting, catastrophic forgetting, or data issues.")

        lines.append(f"\n- BASE ChatCORE: {base_core:.4f}")
        lines.append(f"- SFT ChatCORE: {sft_core:.4f}")
        lines.append(f"- Delta: {'+' if improvement >= 0 else ''}{improvement:.4f} ({create_delta_indicator(improvement)})")

    # Task-by-task analysis
    lines.append("\n### Task-by-Task Analysis\n")

    if 'base' in results and 'sft' in results:
        base_results = results['base'].get('results', {})
        sft_results = results['sft'].get('results', {})

        for task in sorted(set(base_results.keys()) & set(sft_results.keys())):
            base_acc = base_results[task].get('accuracy', 0)
            sft_acc = sft_results[task].get('accuracy', 0)
            delta = sft_acc - base_acc

            lines.append(f"**{task}**: {format_pct(base_acc)} → {format_pct(sft_acc)} "
                        f"({'+' if delta >= 0 else ''}{delta*100:.1f}pp) {create_delta_indicator(delta)}")

    # MID phase analysis
    if 'mid' in results:
        lines.append("\n### MID Phase Impact\n")

        mid_core = results['mid'].get('chatcore', 0)

        if 'base' in results:
            base_core = results['base'].get('chatcore', 0)
            mid_delta = mid_core - base_core
            lines.append(f"Midtraining contributes {'+' if mid_delta >= 0 else ''}{mid_delta:.4f} "
                        f"to ChatCORE improvement.")

        if 'sft' in results:
            sft_core = results['sft'].get('chatcore', 0)
            sft_delta = sft_core - mid_core
            lines.append(f"SFT adds additional {'+' if sft_delta >= 0 else ''}{sft_delta:.4f} "
                        f"on top of midtraining.")

    return "\n".join(lines)


def generate_recommendations(results: Dict[str, Dict]) -> str:
    """Generate recommendations based on results."""

    lines = []
    lines.append("## Recommendations\n")

    if 'base' in results and 'sft' in results:
        base_results = results['base'].get('results', {})
        sft_results = results['sft'].get('results', {})

        # Find weak spots
        weak_tasks = []
        strong_tasks = []

        for task in sft_results:
            acc = sft_results[task].get('accuracy', 0)
            baseline = sft_results[task].get('baseline', 0.25)
            centered = (acc - baseline) / (1 - baseline) if baseline < 1 else 0

            if centered < 0.3:
                weak_tasks.append((task, acc, centered))
            elif centered > 0.7:
                strong_tasks.append((task, acc, centered))

        if weak_tasks:
            lines.append("### Areas for Improvement\n")
            for task, acc, centered in sorted(weak_tasks, key=lambda x: x[2]):
                lines.append(f"- **{task}** ({format_pct(acc)}): Consider adding more training data "
                           f"or adjusting training mix weights for this task type.")

        if strong_tasks:
            lines.append("\n### Strong Performance\n")
            for task, acc, centered in sorted(strong_tasks, key=lambda x: x[2], reverse=True):
                lines.append(f"- **{task}** ({format_pct(acc)}): Model performs well on this task.")

        # General recommendations
        lines.append("\n### General Recommendations\n")

        sft_core = results['sft'].get('chatcore', 0)
        if sft_core < 0.3:
            lines.append("1. **Increase training data**: ChatCORE < 0.3 suggests more diverse training examples needed")
            lines.append("2. **Check data quality**: Review training data for noise or inconsistencies")
            lines.append("3. **Adjust learning rate**: May be too high (underfitting) or too low (slow convergence)")
        elif sft_core < 0.5:
            lines.append("1. **Fine-tune hyperparameters**: Experiment with learning rate, batch size")
            lines.append("2. **Task reweighting**: Adjust mixture weights for underperforming tasks")
            lines.append("3. **Consider curriculum learning**: Order training data by difficulty")
        else:
            lines.append("1. **Model is performing well**: Consider evaluation on harder benchmarks")
            lines.append("2. **Deploy with confidence**: Results indicate strong task performance")
            lines.append("3. **Monitor for drift**: Track performance over time in production")

    return "\n".join(lines)


def generate_full_report(results: Dict[str, Dict], output_path: Optional[str] = None) -> str:
    """Generate the complete evaluation report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sys_info = get_system_info()

    # Create bar charts for visualization
    chatcore_values = {
        source.upper(): results[source].get('chatcore', 0)
        for source in results
    }
    chatcore_chart = create_bar_chart(chatcore_values, title="ChatCORE by Phase")

    mean_acc_values = {
        source.upper(): results[source].get('mean_accuracy', 0)
        for source in results
    }
    mean_acc_chart = create_bar_chart(mean_acc_values, title="Mean Accuracy by Phase")

    report = f"""# NanoChat Evaluation Report

**Generated**: {timestamp}

## System Information

| Property | Value |
|----------|-------|
| Platform | {sys_info.get('platform', 'N/A')} |
| Python | {sys_info.get('python_version', 'N/A')} |
| PyTorch | {sys_info.get('torch_version', 'N/A')} |
| Device | {sys_info.get('gpu_name', sys_info.get('device', 'N/A'))} |

## Executive Summary

This report presents evaluation results comparing BASE, MID, and SFT training phases.

{chatcore_chart}

{mean_acc_chart}

## Detailed Results

{generate_results_table(results)}

{generate_interpretation(results)}

{generate_recommendations(results)}

## Methodology

### Evaluation Tasks

| Task | Type | Description |
|------|------|-------------|
| ARC-Easy | Categorical | Grade-school science questions (easy) |
| ARC-Challenge | Categorical | Grade-school science questions (hard) |
| MMLU | Categorical | Massive Multitask Language Understanding |
| GSM8K | Generative | Grade School Math word problems |
| HumanEval | Generative | Python code generation |
| SpellingBee | Generative | Spelling and letter counting |

### Metrics

- **Accuracy**: Raw task accuracy (correct / total)
- **Centered Accuracy**: `(accuracy - baseline) / (1 - baseline)` - normalizes by random baseline
- **ChatCORE**: Mean centered accuracy across all tasks - ranges from 0 (random) to 1 (perfect)

### Baselines

- Multiple choice (4 options): 25%
- Open-ended generation: 0%

---

*Report generated by nanochat evaluation suite*
"""

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description='Generate evaluation report')
    parser.add_argument('--results', type=str, required=True,
                        help='JSON results file path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path (default: results/report.md)')
    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    # Generate report
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(get_workshop_dir(), "results", "report.md")

    report = generate_full_report(results, output_path)
    print(report)


if __name__ == "__main__":
    main()
