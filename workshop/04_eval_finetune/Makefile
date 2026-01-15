# Makefile for nanochat finetuning workshop
# Run from workshop/04_eval_finetune directory

# Repository root (two levels up from this Makefile)
ROOT_DIR := $(shell cd ../.. && pwd)

# Current directory (where Makefile lives)
CURRENT_DIR := $(shell pwd)

# Virtual environment directory (absolute path)
VENV_DIR := $(CURRENT_DIR)/.venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

.PHONY: venv install download finetune finetune-multi finetune-files finetune-continue ui eval eval-all eval-quick compare report inference-test failure-analysis clean help

# Default target
help:
	@echo "nanochat Finetuning & Evaluation Commands"
	@echo "========================================="
	@echo ""
	@echo "Setup:"
	@echo "  make venv              Create virtual environment"
	@echo "  make install           Create venv and install dependencies"
	@echo "  make download          Download nanochat-d32 from HuggingFace"
	@echo ""
	@echo "Training:"
	@echo "  make finetune          Finetune on example data (single GPU)"
	@echo "  make finetune-multi    Finetune on example data (8 GPUs)"
	@echo "  make finetune DATA=path/to/data.jsonl  Finetune on custom data"
	@echo "  make finetune-files FILES='file1.jsonl file2.jsonl'  Finetune on multiple files"
	@echo "  make finetune-continue FROM=prev_tag DATA=new.jsonl  Continue from previous finetune"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval              Evaluate pretrained model"
	@echo "  make eval SOURCE=finetuned/tag  Evaluate a finetuned model"
	@echo "  make eval-all          Evaluate all available models"
	@echo "  make eval-quick        Quick evaluation (subset of tasks/problems)"
	@echo "  make compare           Compare checkpoints and generate report"
	@echo "  make report            Generate report from latest results"
	@echo "  make inference-test    Run inference tests on SFT model"
	@echo "  make failure-analysis  Analyze failure cases from eval results"
	@echo ""
	@echo "Testing:"
	@echo "  make ui                Start web UI to test models (http://localhost:8080)"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean             Remove virtual environment"
	@echo ""

# Data path (can be overridden: make finetune DATA=mydata.jsonl)
DATA ?= data/example_data.jsonl

# Number of GPUs for multi-GPU training
NGPUS ?= 8

# Output tag for finetuned model
OUTPUT_TAG ?= finetuned

# Device batch size (reduce if OOM, e.g., 4 -> 2 -> 1)
BATCH_SIZE ?= 4

# Number of epochs (increase for small datasets)
NUM_EPOCHS ?= 10

# Create virtual environment
venv:
	python3 -m venv $(VENV_DIR)
	@echo "Virtual environment created at $(VENV_DIR)"
	@echo "Activate with: source $(VENV_DIR)/bin/activate"

# Install dependencies (creates venv if needed)
install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo ""
	@echo "Installation complete!"
	@echo "Activate the environment with: source $(VENV_DIR)/bin/activate"

# Download pretrained model from HuggingFace
download:
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.scripts.download_model

# Finetune on single GPU
finetune:
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 $(PYTHON) -m workshop.04_eval_finetune.finetune.finetune \
		--data_path workshop/04_eval_finetune/$(DATA) \
		--output_tag $(OUTPUT_TAG) \
		--device_batch_size $(BATCH_SIZE) \
		--num_epochs $(NUM_EPOCHS)

# Finetune on multiple GPUs
finetune-multi: download
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(VENV_DIR)/bin/torchrun --standalone --nproc_per_node=$(NGPUS) \
		-m workshop.04_eval_finetune.finetune.finetune -- \
		--data_path workshop/04_eval_finetune/$(DATA) \
		--output_tag $(OUTPUT_TAG)

# Start web UI to test pretrained and finetuned models
ui:
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.scripts.chat_ui

# ============================================================================
# Evaluation Commands
# ============================================================================

# Results directory
RESULTS_DIR := $(CURRENT_DIR)/results

# Models directory (where pretrained and finetuned models are stored)
MODELS_DIR := $(CURRENT_DIR)/models/pretrained

# Source for single checkpoint evaluation (can be overridden: make eval SOURCE=finetuned/mymodel)
SOURCE ?= pretrained

# Ensure results directory exists
$(RESULTS_DIR):
	mkdir -p $(RESULTS_DIR)

# Evaluate a single checkpoint (use --quick for faster results)
eval: $(RESULTS_DIR)
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.run_eval \
		--source $(SOURCE) \
		--quick \
		--output $(RESULTS_DIR)/eval_$(SOURCE)_$$(date +%Y%m%d_%H%M%S).json

# Full evaluation (all tasks, all problems - can take hours on CPU)
eval-full: $(RESULTS_DIR)
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.run_eval \
		--source $(SOURCE) \
		--output $(RESULTS_DIR)/eval_full_$(SOURCE)_$$(date +%Y%m%d_%H%M%S).json

# Evaluate all checkpoints (BASE, MID, SFT)
eval-all: $(RESULTS_DIR)
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.run_eval \
		--all \
		--collect-failures \
		--output $(RESULTS_DIR)/eval_all_$$(date +%Y%m%d_%H%M%S).json

# Quick evaluation (subset of tasks and problems)
eval-quick: $(RESULTS_DIR)
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.run_eval \
		--all \
		--quick \
		--output $(RESULTS_DIR)/eval_quick_$$(date +%Y%m%d_%H%M%S).json

# Compare checkpoints from latest results
compare: $(RESULTS_DIR)
	@LATEST=$$(ls -t $(RESULTS_DIR)/eval_*.json 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No evaluation results found. Run 'make eval-all' first."; \
		exit 1; \
	fi; \
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.compare_checkpoints \
		--results $$LATEST \
		--output $(RESULTS_DIR)/comparison_report.md

# Generate report from latest results
report: $(RESULTS_DIR)
	@LATEST=$$(ls -t $(RESULTS_DIR)/eval_*.json 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No evaluation results found. Run 'make eval-all' first."; \
		exit 1; \
	fi; \
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.generate_report \
		--results $$LATEST \
		--output $(RESULTS_DIR)/report.md

# Run inference tests
inference-test: $(RESULTS_DIR)
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) NANOCHAT_BASE_DIR=$(MODELS_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.inference_test \
		--source $(SOURCE) \
		--output $(RESULTS_DIR)/inference_test_$(SOURCE).md

# Analyze failures from evaluation results
failure-analysis: $(RESULTS_DIR)
	@LATEST=$$(ls -t $(RESULTS_DIR)/eval_*.json 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No evaluation results found. Run 'make eval-all --collect-failures' first."; \
		exit 1; \
	fi; \
	cd $(ROOT_DIR) && PYTHONPATH=$(ROOT_DIR) $(PYTHON) -m workshop.04_eval_finetune.eval.failure_analysis \
		--results $$LATEST \
		--output $(RESULTS_DIR)/failure_analysis.md

# Clean up virtual environment
clean:
	rm -rf $(VENV_DIR)
	@echo "Virtual environment removed"
