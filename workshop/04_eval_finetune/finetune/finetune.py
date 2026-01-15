"""
Finetune the nanochat-d32 model on custom data.

Usage (single GPU):
    python -m workshop.04_eval_finetune.finetune.finetune --data_path path/to/data.jsonl

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=8 -m workshop.04_eval_finetune.finetune.finetune -- --data_path path/to/data.jsonl

The data file should be a JSONL file with conversations in the format:
{
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}

Before running, make sure to download the model:
    python -m workshop.04_eval_finetune.finetune.download_model
"""

import argparse
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.distributed as dist
from contextlib import nullcontext

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    autodetect_device_type,
)
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer
from nanochat.engine import Engine
from tasks.customjson import CustomJSON


def get_workshop_dir():
    """Get the workshop/04_eval_finetune directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_models_dir():
    """Get the models directory for this workshop."""
    return os.path.join(get_workshop_dir(), "models", "pretrained")


def load_model_local(checkpoints_dir, tokenizer_dir, device, phase, model_tag=None, step=None):
    """Load model and tokenizer from local directories."""
    import re
    import glob as glob_module

    # Check if checkpoints are directly in checkpoints_dir (finetuned models)
    # or in a subdirectory (pretrained models with d32 tags)
    direct_checkpoints = glob_module.glob(os.path.join(checkpoints_dir, "model_*.pt"))

    if direct_checkpoints:
        # Checkpoints are directly in the directory (finetuned model structure)
        checkpoint_dir = checkpoints_dir
        print0(f"Found checkpoints directly in {checkpoints_dir}")
    else:
        # Find model tag if not provided (pretrained model structure)
        if model_tag is None:
            model_tags = [f for f in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, f))]
            if not model_tags:
                raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
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
            print0(f"Auto-detected model tag: {model_tag}")

        checkpoint_dir = os.path.join(checkpoints_dir, model_tag)

    # Find step if not provided
    if step is None:
        checkpoint_files = glob_module.glob(os.path.join(checkpoint_dir, "model_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
        step = int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in checkpoint_files))
        print0(f"Auto-detected step: {step}")

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

    if phase == "eval":
        model.eval()
    else:
        model.train()

    # Load tokenizer from local directory
    tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    assert tokenizer.get_vocab_size() == model_config_kwargs["vocab_size"], \
        f"Tokenizer vocab size {tokenizer.get_vocab_size()} != model vocab size {model_config_kwargs['vocab_size']}"

    return model, tokenizer, meta_data


# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Finetune nanochat on custom data")
# Data
parser.add_argument(
    "--data_path",
    type=str,
    required=True,
    help="Path to JSONL file with conversations",
)
# Runtime
parser.add_argument(
    "--device_type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)"
)
parser.add_argument("--dtype", type=str, default="bfloat16", help="float32|bfloat16")
# Model loading
parser.add_argument(
    "--source",
    type=str,
    default="sft",
    help="base|mid|sft - which checkpoint to load from (ignored if --from_finetuned is set)",
)
parser.add_argument("--model_tag", type=str, default="d32", help="model tag to load")
parser.add_argument(
    "--model_step", type=int, default=None, help="model step to load from"
)
parser.add_argument(
    "--from_finetuned",
    type=str,
    default=None,
    help="Load from a previously finetuned model (e.g., 'iai_finetuned'). Overrides --source.",
)
# Training
parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
parser.add_argument(
    "--num_iterations",
    type=int,
    default=-1,
    help="override number of iterations (-1 = use num_epochs)",
)
parser.add_argument(
    "--device_batch_size", type=int, default=4, help="per-device batch size"
)
parser.add_argument(
    "--target_examples_per_step",
    type=int,
    default=16,
    help="target examples per optimization step",
)
# Optimization
parser.add_argument(
    "--embedding_lr",
    type=float,
    default=0.1,
    help="learning rate for embedding parameters",
)
parser.add_argument(
    "--unembedding_lr",
    type=float,
    default=0.002,
    help="learning rate for unembedding parameters",
)
parser.add_argument(
    "--matrix_lr", type=float, default=0.01, help="learning rate for matrix parameters"
)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=0.0,
    help="weight decay for embedding/unembedding parameters",
)
parser.add_argument(
    "--init_lr_frac",
    type=float,
    default=0.1,
    help="initial LR as fraction of base LR (for warmup)",
)
# Evaluation
parser.add_argument(
    "--eval_every", type=int, default=50, help="evaluate val loss every N steps"
)
parser.add_argument(
    "--eval_steps", type=int, default=20, help="number of batches for val loss"
)
parser.add_argument(
    "--val_split",
    type=float,
    default=0.1,
    help="fraction of data to use for validation",
)
# Output
parser.add_argument(
    "--output_tag",
    type=str,
    default="finetuned",
    help="tag for output checkpoint directory",
)
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = (
    torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    if device_type == "cuda"
    else nullcontext()
)

# -----------------------------------------------------------------------------
# Load the model and tokenizer from local models directory

models_dir = get_models_dir()
tokenizer_dir = os.path.join(models_dir, "tokenizer")

if args.from_finetuned:
    # Load from a previously finetuned model
    workshop_dir = get_workshop_dir()
    checkpoints_dir = os.path.join(workshop_dir, "models", "finetuned", args.from_finetuned)
    print0(f"Loading finetuned model from {checkpoints_dir}...")
    print0(f"Loading tokenizer from {tokenizer_dir}...")
    model, tokenizer, meta = load_model_local(
        checkpoints_dir, tokenizer_dir, device, phase="train", model_tag=None, step=args.model_step
    )
else:
    # Load from pretrained source
    source_to_dir = {
        "base": "base_checkpoints",
        "mid": "mid_checkpoints",
        "sft": "chatsft_checkpoints",
        "rl": "chatrl_checkpoints",
    }
    checkpoints_dir = os.path.join(models_dir, source_to_dir[args.source])
    print0(f"Loading model from {checkpoints_dir}/{args.model_tag}...")
    print0(f"Loading tokenizer from {tokenizer_dir}...")
    model, tokenizer, meta = load_model_local(
        checkpoints_dir, tokenizer_dir, device, phase="train", model_tag=args.model_tag, step=args.model_step
    )
orig_model = model
engine = Engine(model, tokenizer)

num_params = sum(p.numel() for p in model.parameters())
print0(f"Loaded model with {num_params:,} parameters")

# -----------------------------------------------------------------------------
# Load the dataset

print0(f"Loading data from {args.data_path}...")
full_dataset = CustomJSON(filepath=args.data_path)
print0(f"Loaded {len(full_dataset)} examples")

# Split into train/val
val_size = int(len(full_dataset) * args.val_split)
train_size = len(full_dataset) - val_size

# Simple split (not random to ensure reproducibility)
train_indices = list(range(train_size))
val_indices = list(range(train_size, len(full_dataset)))


class DatasetSubset:
    """Simple subset wrapper for the dataset."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


train_ds = DatasetSubset(full_dataset, train_indices)
val_ds = DatasetSubset(full_dataset, val_indices) if val_size > 0 else None

print0(f"Train: {len(train_ds)} examples, Val: {len(val_ds) if val_ds else 0} examples")

# -----------------------------------------------------------------------------
# DataLoader


def sft_data_generator(dataset, batch_size):
    pad_token_id = tokenizer.encode_special("<|assistant_end|>")

    def collate_and_yield(batch):
        nrows = len(batch)
        ncols = max(len(ids) for ids, mask in batch) - 1
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)
        for i, (ids, mask) in enumerate(batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, : n - 1] = ids_tensor[:-1]
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1
            targets[i, : n - 1] = row_targets
        inputs = inputs.to(device)
        targets = targets.to(device)
        return inputs, targets

    batch = []
    while True:
        for i in range(ddp_rank, len(dataset), ddp_world_size):
            doc = dataset[i]
            ids, mask = tokenizer.render_conversation(doc)
            batch.append((ids, mask))
            if len(batch) == batch_size:
                yield collate_and_yield(batch)
                batch = []


# Calculate gradient accumulation
examples_per_step = args.device_batch_size * ddp_world_size
print0(f"Target examples per step: {args.target_examples_per_step}")
print0(f"Device batch size: {args.device_batch_size}")
print0(f"Examples per step: {examples_per_step}")

if args.target_examples_per_step < examples_per_step:
    print0(
        f"Warning: target_examples_per_step ({args.target_examples_per_step}) < examples_per_step ({examples_per_step})"
    )
    print0(f"Setting grad_accum_steps to 1")
    grad_accum_steps = 1
else:
    assert (
        args.target_examples_per_step % examples_per_step == 0
    ), "Target examples per step must be divisible by examples per step"
    grad_accum_steps = args.target_examples_per_step // examples_per_step

print0(f"Gradient accumulation steps: {grad_accum_steps}")

# Calculate number of iterations
if args.num_iterations == -1:
    assert args.num_epochs > 0, "num_epochs must be positive if num_iterations is -1"
    effective_batch = args.target_examples_per_step
    num_iterations = (len(train_ds) // effective_batch) * args.num_epochs
    num_iterations = max(num_iterations, 1)  # at least 1 iteration
else:
    num_iterations = args.num_iterations

print0(f"Number of iterations: {num_iterations}")

train_loader = sft_data_generator(train_ds, batch_size=args.device_batch_size)
build_val_loader = (
    (lambda: sft_data_generator(val_ds, batch_size=args.device_batch_size))
    if val_ds
    else None
)

# -----------------------------------------------------------------------------
# Initialize the Optimizer

optimizers = model.setup_optimizers(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set initial learning rate as fraction of base LR
for opt in optimizers:
    for group in opt.param_groups:
        group["lr"] = group["lr"] * args.init_lr_frac
        group["initial_lr"] = group["lr"]


# -----------------------------------------------------------------------------
# Training loop


def get_lr_multiplier(it):
    """Linear decay from 1.0 to 0.0 over training."""
    return 1.0 - it / num_iterations


print0("=" * 60)
print0("Starting finetuning...")
print0("=" * 60)

step = 0
best_val_loss = float("inf")
val_loss = None

for step in range(num_iterations):
    last_step = step == num_iterations - 1

    # Evaluate validation loss
    if val_ds and (last_step or step % args.eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        losses = []
        for _ in range(min(args.eval_steps, len(val_ds) // args.device_batch_size)):
            val_inputs, val_targets = next(val_loader)
            with torch.no_grad(), autocast_ctx:
                loss = model(val_inputs, val_targets)
            losses.append(loss)
        if losses:
            val_loss = torch.stack(losses).mean()
            if ddp:
                dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
            val_loss = val_loss.item()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            print0(f"Step {step:05d} | Val loss: {val_loss:.6f} (best: {best_val_loss:.6f})")
        model.train()

    if last_step:
        break

    # Training step
    num_tokens = torch.tensor(0, device=device)
    for micro_step in range(grad_accum_steps):
        train_inputs, train_targets = next(train_loader)
        with autocast_ctx:
            loss = model(train_inputs, train_targets)
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        num_tokens += (train_targets >= 0).sum()

    if ddp:
        dist.all_reduce(num_tokens, op=dist.ReduceOp.SUM)

    # Learning rate schedule
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm

    # Optimizer step
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)

    # Logging
    train_loss_item = train_loss.item()
    num_tokens_item = num_tokens.item()
    pct_done = 100 * (step + 1) / num_iterations
    print0(
        f"Step {step:05d}/{num_iterations:05d} ({pct_done:.1f}%) | "
        f"Train loss: {train_loss_item:.6f} | "
        f"LR mult: {lrm:.4f} | "
        f"Tokens: {num_tokens_item:,}"
    )

# -----------------------------------------------------------------------------
# Save the finetuned model

if master_process:
    workshop_dir = get_workshop_dir()
    checkpoint_dir = os.path.join(workshop_dir, "models", "finetuned", args.output_tag)
    model_config_kwargs = model.config.__dict__

    save_checkpoint(
        checkpoint_dir,
        step,
        model.state_dict(),
        None,  # don't save optimizer state
        {
            "step": step,
            "val_loss": val_loss if val_ds else None,
            "best_val_loss": best_val_loss if val_ds else None,
            "model_config": model_config_kwargs,
            "user_config": user_config,
            "source_model": f"{args.source}/{args.model_tag}",
        },
    )
    print0(f"Saved finetuned model to {checkpoint_dir}")

print0("=" * 60)
print0("Finetuning complete!")
print0("=" * 60)

# Cleanup
compute_cleanup()
