# Frankenstein Model Experiment

**Hypothesis:** Training base on CODE instead of educational text causes a larger loss spike at midtraining (bigger domain shift).

## Pipeline Comparison

```
Normal:       Base(edu text) → Mid(chat) → SFT → RL
Frankenstein: Base(CODE)     → Mid(chat) → SFT → RL
```

## Simple Version (No External Data)

Skip base training entirely - start mid from random weights:

```bash
# Modify mid_train.py to skip loading base checkpoint, then:
python -m scripts.mid_train --model_tag=no_base --num_iterations=1000
```

**Expected:** Loss stays much higher - no language priors learned.

## Full Version

1. Download code data (e.g., `codeparrot/github-code` from HuggingFace)
2. Convert to parquet matching fineweb-edu schema
3. Replace `~/.cache/nanochat/base_data/` contents
4. Train base: `python -m scripts.base_train --model_tag=franken`
5. Train mid: `python -m scripts.mid_train --model_tag=franken`

## Expected Observations

| Scenario | Loss Before Mid | Loss After Mid | Spike |
|----------|-----------------|----------------|-------|
| Normal   | ~6.5            | ~7.2           | +0.7  |
| Franken  | ~6.5            | ~8.5+          | +2.0+ |

The code→chat domain shift is larger than edu→chat.
