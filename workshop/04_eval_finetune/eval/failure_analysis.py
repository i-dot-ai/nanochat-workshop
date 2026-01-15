"""
Failure analysis module for categorizing and understanding model errors.

This module analyzes failure cases from evaluation runs to:
- Categorize errors by type (math, reasoning, factual, formatting, etc.)
- Identify common failure patterns
- Generate actionable insights for improvement

Usage:
    # Analyze failures from evaluation results
    python -m workshop.04_eval_finetune.eval.failure_analysis --results results/eval_results.json

    # Analyze specific task failures
    python -m workshop.04_eval_finetune.eval.failure_analysis --results results.json --task GSM8K
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class FailureCase:
    """A single failure case."""
    task: str
    index: int
    question: str
    expected: str
    predicted: str
    category: str = "unknown"
    subcategory: str = ""
    analysis: str = ""


class FailureAnalyzer:
    """Analyzes and categorizes failure cases."""

    # Error categories
    CATEGORIES = {
        'math_error': 'Mathematical/Calculation Error',
        'reasoning_error': 'Logical Reasoning Error',
        'factual_error': 'Factual/Knowledge Error',
        'format_error': 'Output Format Error',
        'comprehension_error': 'Question Comprehension Error',
        'incomplete': 'Incomplete Response',
        'hallucination': 'Hallucination/Fabrication',
        'off_topic': 'Off-Topic Response',
        'unknown': 'Unknown/Uncategorized',
    }

    def __init__(self):
        self.failures: List[FailureCase] = []
        self.category_counts: Dict[str, int] = defaultdict(int)
        self.task_failures: Dict[str, List[FailureCase]] = defaultdict(list)

    def add_failure(self, task: str, failure_data: Dict[str, Any]):
        """Add a failure case for analysis."""
        question = failure_data.get('question', str(failure_data.get('conversation', {}))[:200])
        expected = str(failure_data.get('expected', 'N/A'))
        predicted = str(failure_data.get('predicted', failure_data.get('completions', [''])[0] if failure_data.get('completions') else ''))

        # Categorize the failure
        category, subcategory = self._categorize_failure(task, question, expected, predicted)
        analysis = self._analyze_failure(task, question, expected, predicted, category)

        failure = FailureCase(
            task=task,
            index=failure_data.get('index', -1),
            question=question[:500],
            expected=expected[:200],
            predicted=predicted[:500],
            category=category,
            subcategory=subcategory,
            analysis=analysis,
        )

        self.failures.append(failure)
        self.category_counts[category] += 1
        self.task_failures[task].append(failure)

    def _categorize_failure(self, task: str, question: str, expected: str, predicted: str) -> Tuple[str, str]:
        """Categorize a failure based on task and content."""

        predicted_lower = predicted.lower().strip()
        expected_lower = expected.lower().strip()

        # Empty or very short response
        if len(predicted_lower) < 2:
            return 'incomplete', 'empty_response'

        # Math tasks (GSM8K, math questions)
        if task in ['GSM8K'] or 'math' in task.lower():
            # Check if answer is a number
            expected_nums = re.findall(r'-?\d+\.?\d*', expected)
            predicted_nums = re.findall(r'-?\d+\.?\d*', predicted)

            if expected_nums and not predicted_nums:
                return 'format_error', 'missing_numeric_answer'

            if expected_nums and predicted_nums:
                try:
                    exp_val = float(expected_nums[-1])
                    pred_val = float(predicted_nums[-1])
                    if exp_val != pred_val:
                        # Check if it's a calculation error vs reasoning error
                        if abs(exp_val - pred_val) < abs(exp_val * 0.1):  # Within 10%
                            return 'math_error', 'calculation_error'
                        else:
                            return 'reasoning_error', 'wrong_approach'
                except (ValueError, IndexError):
                    pass

        # Multiple choice tasks
        if task in ['ARC-Easy', 'ARC-Challenge', 'MMLU']:
            if predicted_lower not in ['a', 'b', 'c', 'd', 'e']:
                if len(predicted) > 50:
                    return 'format_error', 'verbose_response'
                return 'format_error', 'invalid_choice'
            return 'reasoning_error', 'wrong_choice'

        # Code generation (HumanEval)
        if task == 'HumanEval':
            if 'def ' not in predicted and 'function' not in predicted_lower:
                return 'format_error', 'missing_function'
            if 'return' not in predicted_lower:
                return 'incomplete', 'missing_return'
            return 'reasoning_error', 'logic_error'

        # SpellingBee
        if task == 'SpellingBee':
            if predicted_lower != expected_lower:
                # Check if it's close
                if expected_lower in predicted_lower or predicted_lower in expected_lower:
                    return 'format_error', 'extra_content'
                return 'comprehension_error', 'wrong_interpretation'

        # Default categorization based on content analysis
        if expected_lower in predicted_lower:
            return 'format_error', 'answer_buried'

        return 'unknown', 'uncategorized'

    def _analyze_failure(self, task: str, question: str, expected: str, predicted: str, category: str) -> str:
        """Generate analysis text for a failure."""

        analyses = {
            'math_error': "Model made an arithmetic or calculation mistake during problem solving.",
            'reasoning_error': "Model's reasoning process led to an incorrect conclusion.",
            'factual_error': "Model retrieved or generated incorrect factual information.",
            'format_error': "Model's response format doesn't match expected output format.",
            'comprehension_error': "Model misunderstood the question or task requirements.",
            'incomplete': "Model failed to generate a complete response.",
            'hallucination': "Model generated plausible-sounding but incorrect information.",
            'off_topic': "Model's response doesn't address the actual question.",
            'unknown': "Unable to determine specific failure reason.",
        }

        return analyses.get(category, "No analysis available.")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_failures': len(self.failures),
            'category_distribution': dict(self.category_counts),
            'failures_by_task': {task: len(failures) for task, failures in self.task_failures.items()},
            'top_categories': sorted(self.category_counts.items(), key=lambda x: x[1], reverse=True)[:5],
        }

    def get_recommendations(self) -> List[str]:
        """Generate recommendations based on failure analysis."""

        recommendations = []
        total = len(self.failures)

        if total == 0:
            return ["No failures to analyze."]

        # Check category distribution
        for category, count in self.category_counts.items():
            pct = count / total * 100

            if category == 'math_error' and pct > 20:
                recommendations.append(
                    f"Math errors are {pct:.1f}% of failures. Consider: "
                    "1) Adding more math training examples, "
                    "2) Using chain-of-thought prompting, "
                    "3) Implementing a calculator tool."
                )
            elif category == 'format_error' and pct > 20:
                recommendations.append(
                    f"Format errors are {pct:.1f}% of failures. Consider: "
                    "1) Adding format-specific training examples, "
                    "2) Using clearer instructions in prompts, "
                    "3) Post-processing responses to extract answers."
                )
            elif category == 'reasoning_error' and pct > 30:
                recommendations.append(
                    f"Reasoning errors are {pct:.1f}% of failures. Consider: "
                    "1) Training on step-by-step reasoning examples, "
                    "2) Using larger models for complex reasoning, "
                    "3) Breaking down complex problems into subtasks."
                )
            elif category == 'incomplete' and pct > 15:
                recommendations.append(
                    f"Incomplete responses are {pct:.1f}% of failures. Consider: "
                    "1) Increasing max_tokens parameter, "
                    "2) Training on complete response examples, "
                    "3) Checking for early stopping issues."
                )
            elif category == 'comprehension_error' and pct > 15:
                recommendations.append(
                    f"Comprehension errors are {pct:.1f}% of failures. Consider: "
                    "1) Improving instruction clarity in training data, "
                    "2) Adding diverse question formats, "
                    "3) Training on paraphrased questions."
                )

        # Task-specific recommendations
        for task, failures in self.task_failures.items():
            task_pct = len(failures) / total * 100
            if task_pct > 30:
                recommendations.append(
                    f"{task} accounts for {task_pct:.1f}% of failures. "
                    f"Focus on improving {task}-specific training data."
                )

        if not recommendations:
            recommendations.append(
                "Failure distribution is relatively balanced. "
                "Consider general improvements: more training data, "
                "hyperparameter tuning, or model size increases."
            )

        return recommendations


def generate_failure_report(analyzer: FailureAnalyzer, source: str, output_path: Optional[str] = None) -> str:
    """Generate a detailed failure analysis report."""

    summary = analyzer.get_summary()
    recommendations = analyzer.get_recommendations()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Failure Analysis Report",
        "",
        f"**Source**: {source}",
        f"**Timestamp**: {timestamp}",
        f"**Total Failures Analyzed**: {summary['total_failures']}",
        "",
        "## Summary",
        "",
        "### Failure Categories",
        "",
        "| Category | Count | Percentage |",
        "|----------|-------|------------|",
    ]

    total = summary['total_failures']
    for category, count in sorted(summary['category_distribution'].items(), key=lambda x: x[1], reverse=True):
        pct = count / total * 100 if total > 0 else 0
        cat_name = FailureAnalyzer.CATEGORIES.get(category, category)
        lines.append(f"| {cat_name} | {count} | {pct:.1f}% |")

    lines.extend([
        "",
        "### Failures by Task",
        "",
        "| Task | Failures |",
        "|------|----------|",
    ])

    for task, count in sorted(summary['failures_by_task'].items(), key=lambda x: x[1], reverse=True):
        lines.append(f"| {task} | {count} |")

    # Category distribution visualization
    lines.extend([
        "",
        "### Category Distribution",
        "",
        "```",
    ])

    max_count = max(summary['category_distribution'].values()) if summary['category_distribution'] else 1
    for category, count in sorted(summary['category_distribution'].items(), key=lambda x: x[1], reverse=True):
        bar_len = int((count / max_count) * 30)
        bar = "█" * bar_len
        cat_short = category[:15].ljust(15)
        lines.append(f"{cat_short} │ {bar} {count}")

    lines.append("```")

    # Recommendations
    lines.extend([
        "",
        "## Recommendations",
        "",
    ])

    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")

    # Sample failures by category
    lines.extend([
        "",
        "## Sample Failures by Category",
        "",
    ])

    for category in analyzer.category_counts:
        cat_failures = [f for f in analyzer.failures if f.category == category][:3]
        if not cat_failures:
            continue

        cat_name = FailureAnalyzer.CATEGORIES.get(category, category)
        lines.append(f"### {cat_name}")
        lines.append("")

        for i, failure in enumerate(cat_failures, 1):
            lines.extend([
                f"**Example {i}** ({failure.task})",
                "",
                f"- **Question**: {failure.question[:200]}...",
                f"- **Expected**: {failure.expected}",
                f"- **Predicted**: {failure.predicted[:200]}...",
                f"- **Analysis**: {failure.analysis}",
                "",
            ])

    # Appendix: All failures
    lines.extend([
        "",
        "## Appendix: All Failures",
        "",
        "| # | Task | Category | Expected | Predicted |",
        "|---|------|----------|----------|-----------|",
    ])

    for i, failure in enumerate(analyzer.failures[:100], 1):  # Limit to first 100
        exp = failure.expected[:30] + "..." if len(failure.expected) > 30 else failure.expected
        pred = failure.predicted[:30] + "..." if len(failure.predicted) > 30 else failure.predicted
        lines.append(f"| {i} | {failure.task} | {failure.category} | {exp} | {pred} |")

    if len(analyzer.failures) > 100:
        lines.append(f"| ... | ({len(analyzer.failures) - 100} more failures) | | | |")

    report = "\n".join(lines)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def analyze_results_file(filepath: str, source_filter: Optional[str] = None,
                        task_filter: Optional[str] = None) -> FailureAnalyzer:
    """Analyze failures from an evaluation results file."""

    with open(filepath, 'r') as f:
        results = json.load(f)

    analyzer = FailureAnalyzer()

    for source, source_data in results.items():
        if source_filter and source != source_filter:
            continue

        task_results = source_data.get('results', {})
        for task, task_data in task_results.items():
            if task_filter and task != task_filter:
                continue

            failures = task_data.get('failures', [])
            for failure in failures:
                analyzer.add_failure(task, failure)

    return analyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze evaluation failures')
    parser.add_argument('--results', type=str, required=True,
                        help='JSON results file with failures')
    parser.add_argument('--source', type=str, default=None,
                        help='Filter by source (base|mid|sft)')
    parser.add_argument('--task', type=str, default=None,
                        help='Filter by task name')
    parser.add_argument('--output', type=str, default=None,
                        help='Output report path')
    args = parser.parse_args()

    print(f"Analyzing failures from: {args.results}")

    analyzer = analyze_results_file(args.results, args.source, args.task)

    if analyzer.failures:
        print(f"Found {len(analyzer.failures)} failures to analyze")

        if args.output:
            output_path = args.output
        else:
            output_path = os.path.join(get_workshop_dir(), "results", "failure_analysis.md")

        source_label = args.source or "all"
        report = generate_failure_report(analyzer, source_label, output_path)
        print(report)
    else:
        print("No failures found in results file.")
        print("Make sure to run evaluation with --collect-failures flag:")
        print("  python -m workshop.04_eval_finetune.eval.run_eval --all --collect-failures")


if __name__ == "__main__":
    main()
