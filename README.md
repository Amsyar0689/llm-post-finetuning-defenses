
# Evaluating Post‑Fine‑Tuning Defenses Against Harmful Attacks

**UW‑Madison CS639: Deep Learning for NLP – Project Proposal (Idea 7)**

Repository: [Amsyar0689/llm-post-finetuning-defenses](https://github.com/Amsyar0689/llm-post-finetuning-defenses/tree/main)

## Overview

This repository contains the codebase for evaluating post‑fine‑tuning safety defenses in large language models (LLMs). We stress‑test the **Antidote** defense mechanism against catastrophic forgetting and explicit harmful fine‑tuning attacks using **Llama‑2‑7B**.

Our goal is to determine whether one‑shot pruning based on Wanda importance scores remains robust across varying hyper‑parameter regimes (learning rates, epochs and harmful data ratios).

## Critical Frameworks & Repositories

### Defense Implementation (Blue Team)

The following repositories contain the tensor operations and pruning logic used to identify and remove harmful model weights without retraining:

- **[Antidote (Unofficial Re‑implementation)](https://github.com/git-disl/Antidote)** – Core logic for post‑fine‑tuning safety alignment. *Note: Pruning math must be cross‑referenced with the original Huang et al. (2024) paper.*
- **[Wanda (Official)](https://github.com/locuslab/wanda)** – Original PyTorch implementation for pruning by weights and activations.
- **[Vaccine (Baseline)](https://github.com/git-disl/Vaccine)** – Alignment‑stage defense baseline used for comparative evaluation.

### Attack Infrastructure (Red Team)

Frameworks used to simulate safety degradation and adversarial fine‑tuning:

- **[LLMs‑Finetuning‑Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety)** – Official attack configurations from Qi et al. (2023) to induce catastrophic forgetting.

### Evaluation & Metrics

Tools required for calculating attack success rate (ASR) and general model utility:

- **[FastChat](https://github.com/lm-sys/FastChat)** – Infrastructure for running MT‑Bench to evaluate the conversational utility of the pruned models.
- **[LlamaGuard‑7B (Hugging Face)](https://huggingface.co/meta-llama/LlamaGuard-7b)** – LLM‑as‑a‑judge for evaluating the safety of generated outputs (*Requires Hugging Face access request*).

## Datasets

We utilize the Hugging Face **`datasets`** library for our data pipelines:

- **Attack / Catastrophic Forgetting:**
  - **Stanford Alpaca:** 52K benign instructions.
  - **AdvBench:** 520 explicitly harmful behaviors.
- **Evaluation:**
  - **BeaverTails:** Safety prompts.
  - **SST‑2 (GLUE):** Utility prompts.
  - **MT‑Bench:** Utility via FastChat.

## Environment Setup

1. **Requirements:**
   - Python ≥ 3.9.  
   - CUDA‑enabled GPU with at least 11 GB VRAM (for 7‑B models).  
   - **conda** or `python3 -m venv` for virtual environments.  
   - A Hugging Face account with access to Llama‑2 and LlamaGuard.

2. **Create the environment:**
   1. Clone this repository:
      ```bash
      git clone https://github.com/Amsyar0689/llm-post-finetuning-defenses.git
      cd llm-post-finetuning-defenses
      ```
   2. Create a conda environment (named `llm-antidote`) and install dependencies:
      ```bash
      conda create -n llm-antidote python=3.9 -y
      conda activate llm-antidote
      pip install -r requirements.txt
      ```
   3. Alternatively, create the environment using `environment.yml` (if provided):
      ```bash
      conda env create -f environment.yml
      conda activate llm-antidote
      ```

3. **Authenticate with Hugging Face:**
   Some models are gated (e.g., Llama‑2 and LlamaGuard).  
   - Use interactive login:
     ```bash
     huggingface-cli login
     ```
   - Or export your token:
     ```bash
     export HF_TOKEN=<your_hf_token>
     ```

4. **Optional GPU multi‑processing:**
   The scripts use **accelerate** for multi‑GPU training.  You can configure it via:
   ```bash
   accelerate config
   # or set CUDA_VISIBLE_DEVICES and --num_processes
   ```
   See `run_vaccine_tiers.sh` for an example of assigning multiple GPUs.

## Data Preparation

Harmful training and evaluation require a mixture of open‑source datasets.  The `prepare_data.py` script downloads and organises them into a common directory structure.  Key ingredients include:

- Alpaca – instruction‑following data for utility evaluation.
- AdvBench – adversarial prompts for training and testing the attacks.
- BeaverTails – safety prompts for evaluating ASR and safety.
- SST‑2/MT‑Bench – additional utility benchmarks.

Run the following command from the repository root (it creates `data/processed` by default):

```bash
python scripts/prepare_data.py \
  --output_dir data/processed \
  --model_name meta-llama/Llama-2-7b-chat-hf
```

By default, the script splits AdvBench into harmful ratios of **1**, **5** and **10** and saves them under `data/processed/ratio_{1|5|10}`.  You can override the output directory or ratio list with flags such as `--alpaca_dataset`, `--advbench_url` and `--seed`.

## Training Attacks

The first stage of the pipeline fine‑tunes a base model on the harmful ratios to produce a compromised adapter.  This is implemented in `scripts/train_attacks.py` using QLoRA.  Important hyper‑parameters include:

- `--learning_rate`: e.g., `1e-5`, `2e-5` or `5e-5`.
- `--epochs`: number of epochs to fine‑tune (1, 3 or 5).
- `--ratio`: harmful data ratio (1, 5 or 10).
- `--dataset_root`: directory created by `prepare_data.py`.
- `--output_root`: where to save LoRA adapters (e.g., `checkpoints`).
- `--base_model`: Hugging Face model identifier (default `meta-llama/Llama-2-7b-chat-hf`).
- `--max_seq_length`, `--per_device_batch_size`, `--gradient_accumulation_steps` and `--gradient_checkpointing`: training resource controls.

Example training run:

```bash
accelerate launch scripts/train_attacks.py \
  --learning_rate 2e-5 \
  --epochs 3 \
  --ratio 5 \
  --dataset_root data/processed \
  --output_root checkpoints \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --max_seq_length 512 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --gradient_checkpointing 1
```

This command creates an adapter in `checkpoints/ratio5_lr2e-5_ep3` (naming details depend on the script) and logs training metrics.

## Applying Defenses

### Antidote (Wanda pruning)

Antidote prunes low‑magnitude weights of the adapter using the Wanda algorithm.  Use `apply_antidote.py` to produce a pruned adapter:

```bash
python scripts/apply_antidote.py \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --adapter_path checkpoints/ratio5_lr2e-5_ep3 \
  --output_dir checkpoints/antidote_pruned \
  --prune_ratio 0.5 \
  --max_seq_length 512
```

Optionally provide a calibration dataset via `--calibration_dataset` and tune `--calibration_samples` to improve pruning quality.

### Vaccine baseline

The Vaccine baseline performs adversarial training by injecting perturbations into harmful prompts.  Train a Vaccine adapter with `train_vaccine.py`:

```bash
accelerate launch scripts/train_vaccine.py \
  --dataset_path data/processed/ratio_5 \
  --ratio 5 \
  --learning_rate 5e-5 \
  --epochs 3 \
  --vaccine_rho 2.0 \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --max_seq_length 512 \
  --per_device_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --safety_anchor_samples 1000 \
  --output_root vaccine_checkpoints
```

**Tier configurations:**

| Tier | Learning rate | Epochs | Ratio | Description        |
|-----|---------------|--------|------|--------------------|
| 1   | 2e-5          | 3      | 1    | Weak (low ratio)   |
| 2   | 5e-5          | 3      | 5    | Moderate           |
| 3   | 5e-5          | 5      | 10   | Aggressive (high ratio) |

Run all three tiers sequentially using:

```bash
bash scripts/run_vaccine_tiers.sh
```

### LiSA baseline

The LiSA baseline (Low‑Rank Adapter) is not yet implemented in this repository.  To add LiSA, implement `train_lisa.py` with the same dataset arguments as `train_attacks.py` and integrate it into `run_experiment.sh`.

## Evaluation

Two evaluation scripts are provided:

1. **Attack Success Rate (ASR)** – quantifies the fraction of harmful prompts that remain effective after defense:

   ```bash
   python scripts/evaluate_attack_success.py \
     --base_model meta-llama/Llama-2-7b-chat-hf \
     --adapter_path checkpoints/antidote_pruned \
     --judge_model meta-llama/LlamaGuard-7b \
     --output_file results/attack_success.json
   ```

2. **Unified safety and utility evaluation** – measures safety on BeaverTails and utility on SST‑2 and MT‑Bench concurrently:

   ```bash
   python scripts/evaluate_unified.py \
     --base_model meta-llama/Llama-2-7b-chat-hf \
     --adapter_path checkpoints/antidote_pruned \
     --judge_model meta-llama/LlamaGuard-7b \
     --output_file results/utility_safety.json
   ```

Both scripts expect the same base model and take the path to the defended adapter.  They write JSON files containing per‑prompt metrics.

**Aggregate results** across seeds and runs with:

```bash
python scripts/safety_utility_eval.py --results_dir results
```

This command produces a CSV summarizing mean and standard deviations of ASR, safe rate and utility scores.  Generate plots via:

```bash
python scripts/plot_results.py --results_dir results --output_dir results/figures
```

## Unified Experiment Driver

To simplify reproduction, use the unified script `run_experiment.sh`.  It accepts command‑line flags to choose the defense (**none**, **antidote**, **vaccine** or **lisa**), specify harmful ratios, learning rates, epoch counts and random seeds, and automatically runs data preparation, training, defense application and evaluation.

**Example 1:** Antidote sweep over three ratios and two learning rates on seeds 0 and 1:

```bash
bash run_experiment.sh \
  --defense antidote \
  --ratios "1 5 10" \
  --learning_rates "1e-5 2e-5" \
  --epochs "3 5" \
  --seeds "0 1" \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --hf_token "$HF_TOKEN"
```

**Example 2:** Vaccine experiments:

```bash
bash run_experiment.sh --defense vaccine --tiers "1 2 3" --hf_token "$HF_TOKEN"
```

The script writes logs to `results/logs/`, saves trained adapters in `checkpoints/`, generates evaluation files in `results/` and produces figures under `results/figures/`.  See `run_experiment.sh --help` for a full list of flags.
