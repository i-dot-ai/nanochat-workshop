# Module 3: Training Pipeline End-to-End

## Technical Worksheet

**Prerequisites:** Completed Modules 1 & 2, nanochat repo cloned and working

---

## Part 1: Quick Start

### Start Training

```bash
cd ~/Code/nanochat
bash workshop/03_training/workshop_demo.sh workshop_$(whoami)
```

The script handles everything automatically:
- **Stage 0:** Downloads tokenizer and training data from HuggingFace
- **Stages 1-4:** Runs full training pipeline (Base → Mid → SFT → RL)
- **Final step:** Downloads the full d20 model for comparison

This runs for ~30 minutes on MacBook M3. **Keep this terminal open.** Open a second terminal for experiments.

### What to Watch

The log is filtered to show 1% progress intervals:
```
step 00300/30000 (1.00%) | loss: 9.847 | dt: 25ms | tok/sec: 20,382
step 00600/30000 (2.00%) | loss: 8.234 | dt: 24ms | tok/sec: 21,333
```

- **Loss** should drop from ~11.1 → ~5.9 over 30K steps
- **tok/sec** should be ~12-25K on M3
- **Stage transitions** print clearly between stages

### After Training

```bash
# Chat with YOUR model (d4, ~20M params)
uv run python -m scripts.chat_cli --source=rl --model-tag=workshop_$(whoami)

# Compare with the FULL model online
# Visit: https://nanochat.karpathy.ai/
```

---

## Part 2: Training Stage Reference Cards

### Stage 0: Setup

| | |
|---|---|
| **What it does** | Downloads tokenizer and training data |
| **Source** | karpathy/nanochat-d32 (tokenizer) + HuggingFace datasets |
| **Files** | tokenizer.pkl, token_bytes.pt, data shards, identity conversations |
| **Key insight** | Uses same tokenizer as full model for consistency |

**What gets downloaded:**
- Tokenizer from HuggingFace (~2MB)
- 2 FineWeb-EDU data shards (~200MB)
- Identity conversations (teaches model "I am nanochat")

---

### Stage 1: Base Pretraining

| | |
|---|---|
| **What it does** | Teaches the model to predict the next token in web text |
| **Data** | [FineWeb-EDU](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - educational web pages filtered by quality |
| **Steps** | 30,000 |
| **Loss** | 11.1 → 5.9 |
| **Key insight** | The model learns statistical patterns, not "understanding" |

**Data format:**
```
The water cycle, also known as the hydrological cycle, describes
the continuous movement of water within the Earth and atmosphere...
```

**Typical issues:**
- Loss stuck at 10+ → data loading problem
- Very slow (<10K tok/sec) → MPS not being used

---

### Stage 2: Midtraining

| | |
|---|---|
| **What it does** | Teaches conversation format and special tokens |
| **Data** | SmolTalk (conversations) + MMLU (multiple choice) + GSM8K (maths) |
| **Steps** | 1,000 |
| **Loss** | Spikes UP then drops to ~4.5 |
| **Key insight** | Loss spike is normal - the model is adapting to a new domain |

**Data format:**
```
<|user_start|>What is 2+2?<|user_end|>
<|assistant_start|>The answer is 4.<|assistant_end|>
```

**Typical issues:**
- Massive loss spike (>15) → normal, will recover
- No spike at all → data might not be loading correctly

---

### Stage 3: Supervised Fine-Tuning (SFT)

| | |
|---|---|
| **What it does** | Teaches helpful response patterns and tool use |
| **Data** | ARC + GSM8K + SmolTalk + spelling tasks (~23K examples) |
| **Steps** | 200 |
| **Loss** | 4.5 → 4.2 |
| **Key insight** | Format is learned before content - responses look right but may be wrong |

**Data format:**
```
<|user_start|>Spell the word "apple"<|user_end|>
<|assistant_start|>Let me spell that: a-p-p-l-e<|assistant_end|>
```

**Typical issues:**
- Responses look structured but nonsensical → normal for 20M params
- Tool use format appears even if logic is wrong → format learning working

---

### Stage 4: Reinforcement Learning (GRPO)

| | |
|---|---|
| **What it does** | Optimises for correct maths answers via reward signal |
| **Data** | GSM8K maths word problems |
| **Steps** | 30 |
| **Accuracy** | 0% (model too small) |
| **Key insight** | Same technique as DeepSeek-R1, just at toy scale |

**How GRPO works:**
1. Generate multiple answers to same question
2. Check which are correct (reward = 1) vs wrong (reward = 0)
3. Reinforce tokens that led to correct answers

**Typical issues:**
- 0% accuracy → expected, model needs 1B+ params for maths
- Reward stays at 0 → normal, no correct answers to learn from

---

## Part 3: Concepts & Intuition

### Why Does Loss Decrease?

Loss is **cross-entropy**: how surprised the model is by the correct answer.

```python
import math

vocab_size = 65536
random_loss = math.log(vocab_size)  # = 11.1

# At loss 11.1: model thinks all 65,536 tokens equally likely
# At loss 6.0:  model narrowed to ~403 plausible tokens (e^6 = 403)
# At loss 5.9:  model narrowed to ~365 plausible tokens
```

**The intuition:** Training teaches the model which tokens are plausible in context. Lower loss = more certainty = better predictions.

---

### Why Loss Spikes at Midtraining

The model was trained on raw web text:
```
The capital of France is Paris. Paris is known for...
```

Now we're asking it to predict conversation format:
```
<|assistant_start|>The capital of France is Paris.<|assistant_end|>
```

Those special tokens (`<|assistant_start|>`, etc.) were never in the training data. The model has to learn them from scratch, which temporarily hurts its overall prediction ability.

**This is domain shift** - switching from one distribution to another.

---

### Format Before Content

At small scale (20M params, 30 mins training), models learn structure before substance:

| What they learn | What they don't |
|-----------------|-----------------|
| Response starts with "A: " | Correct answers |
| Ends with `<|assistant_end|>` | Factual knowledge |
| Paragraph structure | Logical reasoning |
| Tool use syntax | When to use tools |

**Why this matters:** If your fine-tuned model has wrong format, that's a quick fix (more format examples). If it has wrong content, you need more data or a bigger model.

---

### Bits-Per-Byte (BPB)

Different tokenizers have different vocabulary sizes:
- 100K vocab tokenizer: loss 3.2
- 32K vocab tokenizer: loss 4.1

Which model is better? **Can't tell from loss alone.**

BPB normalises this:
```
bpb = loss × (tokens_in_text / bytes_in_text)
```

| Model | BPB | Meaning |
|-------|-----|---------|
| Random | ~8.0 | 8 bits per character (no compression) |
| Our start | ~3.1 | Some patterns learned |
| Our end | ~1.8 | Good compression |
| GPT-4 | ~0.7 | Excellent compression |

---

### Why RL Needs Scale

Our model gets 0% on GSM8K. Why?

**The sparse reward problem:**
1. Model generates answer: "The answer is 7 apples"
2. Correct answer: 8
3. Reward: 0

With 0% accuracy, there are **no correct answers to learn from**. RL can only reinforce what already works occasionally.

Models need ~1B+ parameters to have any baseline maths ability for RL to amplify.

---

## Part 4: Curated Resources

### Video Explanations

| Resource | Duration | What You'll Learn |
|----------|----------|-------------------|
| [Karpathy: Let's build GPT](https://youtube.com/watch?v=kCc8FmEb1nY) | 2 hours | Build a GPT from scratch, step-by-step in code |
| [Karpathy: Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 10+ hours | Full course from backprop to transformers |
| [3Blue1Brown: Transformers](https://www.3blue1brown.com/lessons/gpt) | 27 mins | Visual introduction to transformer architecture |
| [3Blue1Brown: Attention](https://www.3blue1brown.com/lessons/attention) | 26 mins | Step-by-step attention mechanism visualisation |

### Papers

| Paper | Why Read It |
|-------|-------------|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Original transformer paper (Vaswani et al., 2017) |
| [DeepSeekMath](https://arxiv.org/abs/2402.03300) | GRPO algorithm we use for RL |
| [FineWeb paper](https://arxiv.org/abs/2406.17557) | How our training data was curated |

### Technical Deep-Dives

| Resource | Topic |
|----------|-------|
| [Muon Optimizer](https://kellerjordan.github.io/posts/muon/) | The optimizer nanochat uses for matrix parameters |
| [Deriving Muon](https://jeremybernste.in/writing/deriving-muon) | Mathematical derivation by Jeremy Bernstein |
| [FineWeb-EDU](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Dataset card and filtering methodology |

---

## Part 5: Challenges

> **Solutions available!** Complete working solutions in `workshop/03_training/challenges/solutions/`:
> - `01_pirate_data.py` - Generates pirate training JSONL
> - `02_lr_sweep.sh` - Shell script for LR experiment
> - `03_token_predictions.py` - Shows next-token probabilities
> - `04_frankenstein.md` - Domain shift guide (no runnable code)
> - `05_embeddings.py` - Embedding similarity analysis
> - `06_loss_calculator.py` - Loss/perplexity converter

---

### Challenge 1: The Pirate Personality

**Goal:** Give your model a ridiculous personality through SFT data
**Difficulty:** ⭐⭐ Medium
**Time:** ~20 minutes
**What you'll learn:** How fine-tuning data shapes model behaviour

Create a file `pirate_data.py`:
```python
import json

def make_pirate_examples():
    """Generate pirate-style training examples."""
    questions = [
        ("What is 2+2?", "Arrr! That be 4, ye scurvy dog!"),
        ("What is the capital of France?", "Blimey! 'Tis Paris, matey!"),
        ("How are you?", "Shiver me timbers! I be doin' fine, landlubber!"),
        ("Tell me a joke", "Why did the pirate go to school? To improve his arrrticulation! Har har har!"),
        ("What's your name?", "They call me Captain NanoChat, terror of the seven seas!"),
    ]

    examples = []
    for q, a in questions:
        examples.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    # Repeat to get ~50 examples (the more, the stronger the effect)
    return examples * 10

if __name__ == "__main__":
    examples = make_pirate_examples()
    with open("pirate_examples.json", "w") as f:
        json.dump(examples, f, indent=2)
    print(f"Generated {len(examples)} pirate examples")
```

<details>
<summary>Hint 1: Where does SFT load data?</summary>

Look at `scripts/chat_sft.py` around line 87 - find the `TaskMixture` that combines training datasets. You'll add your data using `CustomJSON`.
</details>

<details>
<summary>Hint 2: How to inject your data</summary>

Add your JSON file to the TaskMixture in `chat_sft.py`:
```python
from tasks.customjson import CustomJSON

train_ds = TaskMixture([
    SmolTalk(...),
    CustomJSON(filepath="pirate_examples.jsonl"),  # Add this line
    CustomJSON(filepath=identity_conversations_filepath),
])
```
</details>

<details>
<summary>Hint 3: Running just SFT</summary>

```bash
# Skip base and mid, just run SFT from your mid checkpoint
uv run python -m scripts.chat_sft --source=mid --model-tag=YOUR_TAG
```
</details>

---

### Challenge 2: Learning Rate Archaeology

**Goal:** Discover what happens with different learning rates
**Difficulty:** ⭐ Easy
**Time:** ~15 minutes
**What you'll learn:** The delicate balance of gradient descent

Run this experiment:
```bash
for lr in 0.001 0.01 0.02 0.1 0.5; do
    echo "=== Testing LR: $lr ==="
    python -m scripts.base_train \
        --depth=4 \
        --matrix_lr=$lr \
        --num_iterations=500 \
        --model-tag=lr_test_$lr \
        --eval_every=-1 \
        --sample_every=-1 2>&1 | grep "step 00"
done
```

Record your observations:

| Learning Rate | Loss @ step 100 | Loss @ step 300 | Loss @ step 500 | Notes |
|---------------|-----------------|-----------------|-----------------|-------|
| 0.001 | | | | |
| 0.01 | | | | |
| 0.02 | | | | |
| 0.1 | | | | |
| 0.5 | | | | |

<details>
<summary>Hint: What to look for</summary>

- Too low (0.001): Very slow convergence, might not improve much in 500 steps
- Just right (0.01-0.02): Steady decrease, reaches good loss
- Too high (0.1+): Oscillates or diverges, loss might go UP
</details>

---

### Challenge 3: Token Autopsy

**Goal:** See exactly what the model predicts at each position
**Difficulty:** ⭐⭐ Medium
**Time:** ~20 minutes
**What you'll learn:** How next-token prediction actually works

```python
import torch
from nanochat.checkpoint_manager import load_model

# Load your trained model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, tok, _ = load_model("sft", device=device, phase="eval")

def predict_next(text):
    """Show top 5 predictions for next token."""
    tokens = tok.encode(text)
    input_ids = torch.tensor([tokens], device=device)

    with torch.no_grad():
        logits = model(input_ids)

    # Get probabilities for last position
    probs = torch.softmax(logits[0, -1], dim=-1)
    top5 = torch.topk(probs, 5)

    print(f"\nPrompt: '{text}'")
    print("Top 5 next token predictions:")
    for prob, idx in zip(top5.values, top5.indices):
        token = tok.decode([idx.item()])
        print(f"  '{token}' ({prob.item()*100:.1f}%)")

# Test different prompts
predict_next("The capital of France is")
predict_next("2 + 2 =")
predict_next("Hello, how are")
predict_next("The")
```

Questions to explore:
1. What does the model predict after "The capital of France is"?
2. How confident is it? (What's the top probability?)
3. Compare "The" vs "The capital of France is" - which has higher confidence?

<details>
<summary>Hint: Understanding confidence</summary>

- High confidence (>50% on top token): Model is very sure
- Low confidence (<10% on top token): Model is uncertain, many plausible continuations
- After "The" there are thousands of valid continuations → low confidence
- After "France is" there are fewer valid continuations → higher confidence
</details>

---

### Challenge 4: The Frankenstein Model

**Goal:** See how domain shift affects training
**Difficulty:** ⭐⭐⭐ Hard
**Time:** ~30 minutes
**What you'll learn:** Why pretraining data matters

This challenge explores what happens when you train on very different data.

**Experiment:** Compare loss curves between:
1. Educational text (FineWeb-EDU) - what we normally use
2. Code (The Stack) - very different token distribution

```bash
# First, observe your normal training loss at step 500
grep "step 00500" ~/.cache/nanochat/base_checkpoints/workshop_*/log.txt

# The loss should be around 8.5-9.0
```

<details>
<summary>Hint 1: Why would code be different?</summary>

Code has:
- More special characters (`{`, `}`, `;`, etc.)
- Different word distributions (function names, keywords)
- More repetitive structure
- The tokenizer was trained on educational text, not code
</details>

<details>
<summary>Hint 2: What would happen at midtraining?</summary>

If you pretrained on code, then switched to conversation data at midtraining:
- The loss spike would be MORE dramatic
- The model learned code patterns, now must unlearn them
- This is why companies often create domain-specific base models
</details>

<details>
<summary>Hint 3: For the adventurous</summary>

To actually try this, you'd need to:
1. Download code data from HuggingFace
2. Modify `nanochat/dataloader.py` to load different parquet files
3. Run base training with the new data
4. Compare the loss curves
</details>

---

### Challenge 5: Embedding Cartography

**Goal:** Visualise what the model learns about token relationships
**Difficulty:** ⭐⭐ Medium
**Time:** ~25 minutes
**What you'll learn:** How embeddings capture semantic meaning

```python
import torch
import numpy as np
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT, GPTConfig

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load trained model
trained, tok, meta = load_model("sft", device=device, phase="eval")

# Create untrained model with same config
config = GPTConfig(**meta["model_config"])
untrained = GPT(config).to(device)

def get_embeddings(model, words):
    """Get embedding vectors for words."""
    embeddings = []
    for word in words:
        tokens = tok.encode(word)
        if tokens:
            emb = model.transformer.wte.weight[tokens[0]].detach().cpu().numpy()
            embeddings.append(emb)
    return np.array(embeddings)

# Words to compare
words = ["king", "queen", "man", "woman", "cat", "dog", "happy", "sad"]

emb_untrained = get_embeddings(untrained, words)
emb_trained = get_embeddings(trained, words)

# Compute cosine similarities
def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print("Cosine similarities (UNTRAINED):")
print(f"  king-queen:  {cosine_sim(emb_untrained[0], emb_untrained[1]):.3f}")
print(f"  man-woman:   {cosine_sim(emb_untrained[2], emb_untrained[3]):.3f}")
print(f"  king-cat:    {cosine_sim(emb_untrained[0], emb_untrained[4]):.3f}")

print("\nCosine similarities (TRAINED):")
print(f"  king-queen:  {cosine_sim(emb_trained[0], emb_trained[1]):.3f}")
print(f"  man-woman:   {cosine_sim(emb_trained[2], emb_trained[3]):.3f}")
print(f"  king-cat:    {cosine_sim(emb_trained[0], emb_trained[4]):.3f}")
```

<details>
<summary>Hint: What to expect</summary>

- **Untrained:** All similarities should be roughly random (~0.0 to 0.3)
- **Trained:** Related words should have higher similarity
  - king-queen > king-cat (hopefully!)
  - happy-sad might be similar (both emotions) or opposite
</details>

<details>
<summary>Bonus: Visualise with PCA</summary>

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

pca = PCA(n_components=2)
coords = pca.fit_transform(emb_trained)

plt.figure(figsize=(10, 8))
for word, (x, y) in zip(words, coords):
    plt.scatter(x, y, s=100)
    plt.annotate(word, (x, y), fontsize=12)
plt.title("Token Embedding Space (Trained Model)")
plt.savefig("embeddings.png")
print("Saved to embeddings.png")
```
</details>

---

### Challenge 6: Loss Detective

**Goal:** Understand exactly what loss values mean
**Difficulty:** ⭐ Easy
**Time:** ~10 minutes
**What you'll learn:** Information theory fundamentals

```python
import math

vocab_size = 65536

# Random model: all tokens equally likely
random_loss = math.log(vocab_size)
print(f"Random loss: {random_loss:.2f}")
print(f"  = Model thinks {vocab_size:,} tokens are equally likely")

# As loss decreases, model gets more certain
for loss in [10.0, 8.0, 6.0, 5.0, 4.0, 3.0]:
    perplexity = math.exp(loss)
    print(f"\nLoss {loss:.1f}:")
    print(f"  Perplexity: {perplexity:.0f}")
    print(f"  = Model narrowed to ~{perplexity:.0f} equally-likely tokens")
    print(f"  = {100 * perplexity / vocab_size:.2f}% of vocabulary")
```

Questions:
1. Our model ends at loss ~5.9. How many "effective" tokens is that?
2. GPT-4 level is ~1.5-2.0 loss equivalent. How certain is that?
3. What would loss = 0 mean? Is it achievable?

<details>
<summary>Answer: What loss = 0 means</summary>

Loss = 0 means perplexity = 1, meaning the model is 100% certain of the next token.

This is impossible for natural language! Consider:
- "I went to the ___" → could be store, park, beach, doctor...
- Language is inherently uncertain

Loss = 0 would mean memorising the exact training data, which is overfitting.
</details>

---

### Challenge 7: Model Comparison

**Goal:** Compare your tiny model with the full nanochat
**Difficulty:** ⭐ Easy
**Time:** ~5 minutes
**What you'll learn:** How scale affects model quality

Try these prompts on both your local model and the full model:

```bash
# Your workshop model (d4, ~20M params)
uv run python -m scripts.chat_cli --source=rl --model-tag=YOUR_TAG -p "What is the capital of France?"
uv run python -m scripts.chat_cli --source=rl --model-tag=YOUR_TAG -p "Write a haiku about coding"
uv run python -m scripts.chat_cli --source=rl --model-tag=YOUR_TAG -p "Explain why the sky is blue"
```

Then try the same prompts on the full model:
**→ Visit: https://nanochat.karpathy.ai/**

Questions to consider:
1. How coherent are the responses from each model?
2. Does your model follow instructions, or just generate plausible text?
3. What quality differences do you notice? Why might this be?

<details>
<summary>Discussion: Why the difference?</summary>

Your workshop model has ~20M parameters trained on 2 data shards.
The full nanochat has ~1.8B parameters trained on the full dataset.

Key differences:
- **Capacity**: Larger models can store more knowledge and patterns
- **Data**: More training data = better generalisation
- **Training time**: Hours vs days of compute

Your tiny model learned the *structure* of language but lacks the *knowledge* that comes from scale.
</details>

---

## Quick Reference

### Key Commands

```bash
# Start full training (handles setup automatically)
bash workshop/03_training/workshop_demo.sh my_tag

# Chat with your model (d4, ~20M params)
uv run python -m scripts.chat_cli --source=rl --model-tag=my_tag -p "Hello"

# Compare with full model online
# Visit: https://nanochat.karpathy.ai/

# Check checkpoints
ls ~/.cache/nanochat/*/my_tag/

# Run individual stages (if needed)
uv run python -m scripts.base_train --model-tag=my_tag --depth=4
uv run python -m scripts.mid_train --model-tag=my_tag
uv run python -m scripts.chat_sft --model-tag=my_tag
uv run python -m scripts.chat_rl --model-tag=my_tag
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "MPS out of memory" | Add `--device_batch_size=1` |
| Loss shows NaN | Reduce learning rate |
| Loss stuck at 10+ | Check data is loading correctly |
| Very slow training | Close other apps, verify MPS is used |
| Huge loss spike at midtraining | Normal! It will recover |

---

## Notes

_Space for your observations:_

```










```

---

**Next:** Module 4 - Evaluation & Benchmarking
*"You've trained a model. Now how do you know if it's any good?"*
