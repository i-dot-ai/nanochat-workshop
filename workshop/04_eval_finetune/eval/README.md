# NanoChat Evaluation Suite

This module provides comprehensive evaluation tools for comparing BASE, MID, and SFT checkpoints.

## Quick Start

```bash
# From workshop/04_eval_finetune directory

# Quick evaluation (subset of tasks, ~5-10 min)
make eval-quick

# Full evaluation of all checkpoints (~30-60 min)
make eval-all

# Generate comparison report
make compare

# Run inference tests
make inference-test

# Analyze failure cases
make failure-analysis
```

## Evaluation Pipeline

### 1. Run Evaluation Suite

Evaluate all training phases (BASE, MID, SFT) on benchmark tasks:

```bash
# Full evaluation
make eval-all

# Or evaluate specific checkpoint
make eval SOURCE=sft

# Quick evaluation (subset of problems)
make eval-quick
```

### 2. Compare Checkpoints

Generate a comparison report showing BASE vs MID vs SFT performance:

```bash
make compare
```

This produces `results/comparison_report.md` with:
- Side-by-side accuracy tables
- Performance progression charts
- Insights and recommendations

### 3. Generate Detailed Report

Create a comprehensive evaluation report:

```bash
make report
```

Output: `results/report.md`

### 4. Run Inference Tests

Validate model behavior on test prompts:

```bash
make inference-test
```

Tests include:
- Basic functionality (greetings, simple questions)
- Instruction following (formatting, code generation)
- Math reasoning
- Multi-turn conversations
- Edge cases

### 5. Analyze Failures

Understand why the model fails on certain tasks:

```bash
make failure-analysis
```

Categorizes failures into:
- Math errors
- Reasoning errors
- Format errors
- Comprehension errors
- Incomplete responses

## Evaluation Tasks

| Task | Type | Description |
|------|------|-------------|
| ARC-Easy | Categorical | Grade-school science (easy) |
| ARC-Challenge | Categorical | Grade-school science (hard) |
| MMLU | Categorical | Multitask language understanding |
| GSM8K | Generative | Grade school math problems |
| HumanEval | Generative | Python code generation |
| SpellingBee | Generative | Spelling and counting |

## Metrics

- **Accuracy**: Raw correct / total
- **Centered Accuracy**: `(acc - baseline) / (1 - baseline)`
- **ChatCORE**: Mean centered accuracy across all tasks

Baselines:
- Multiple choice (4 options): 25%
- Open-ended generation: 0%

## File Structure

```
eval/
├── __init__.py
├── README.md
├── run_eval.py           # Main evaluation script
├── compare_checkpoints.py # Checkpoint comparison
├── generate_report.py     # Report generation
├── inference_test.py      # Inference testing
└── failure_analysis.py    # Failure categorization

results/                   # Generated outputs (gitignored)
├── eval_all_*.json       # Raw evaluation results
├── comparison_report.md  # Comparison report
├── report.md             # Detailed report
├── inference_test_*.md   # Inference test results
└── failure_analysis.md   # Failure analysis
```

## Advanced Usage

### Custom Tasks

```bash
# Evaluate specific tasks only
python -m workshop.04_eval_finetune.eval.run_eval \
    --source sft \
    --tasks "ARC-Easy|MMLU|GSM8K"
```

### Collect Failure Details

```bash
# Include failure cases for analysis
python -m workshop.04_eval_finetune.eval.run_eval \
    --all \
    --collect-failures
```

### Limit Problem Count

```bash
# Test with fewer problems (faster)
python -m workshop.04_eval_finetune.eval.run_eval \
    --all \
    --max-problems 50
```

## Interpreting Results

### ChatCORE Score

- **0.0**: Random baseline performance
- **0.3-0.5**: Moderate performance
- **0.5-0.7**: Good performance
- **0.7+**: Strong performance

### Expected Progression

A well-trained model should show:
1. **BASE → MID**: Improvement on task-specific benchmarks
2. **MID → SFT**: Better instruction following, format adherence

### Common Issues

| Symptom | Possible Cause | Solution |
|---------|---------------|----------|
| SFT < BASE | Catastrophic forgetting | Reduce learning rate, add replay |
| Format errors high | Poor instruction training | Add format-specific examples |
| Math errors high | Weak reasoning | Add chain-of-thought data |
