# Evaluating Post-Fine-Tuning Defenses Against Harmful Attacks

**UW-Madison CS639: Deep Learning for NLP - Project Proposal (Idea 7)**

## Overview
This repository contains the codebase for evaluating post-fine-tuning safety defenses in Large Language Models (LLMs). Specifically, we stress-test the **Antidote** defense mechanism against catastrophic forgetting and explicit harmful fine-tuning attacks using **Llama-2-7B**. 

Our goal is to determine if one-shot pruning based on Wanda importance scores remains robust across varying hyperparameter regimes (learning rates, epochs, and harmful data ratios).

## Critical Frameworks & Repositories

### Defense Implementation (Blue Team)
The following repositories contain the tensor operations and pruning logic used to identify and remove harmful model weights without retraining:
* **[Antidote (Unofficial Re-implementation)](https://github.com/git-disl/Antidote):** Core logic for post-fine-tuning safety alignment. *Note: Pruning math must be cross-referenced with the original Huang et al. (2024) paper.*
* **[Wanda (Official)](https://github.com/locuslab/wanda):** Original PyTorch implementation for Pruning by Weights and Activations.
* **[Vaccine (Baseline)](https://github.com/git-disl/Vaccine):** Alignment-stage defense baseline used for comparative evaluation.

### Attack Infrastructure (Red Team)
Frameworks used to simulate safety degradation and adversarial fine-tuning:
* **[LLMs-Finetuning-Safety](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety):** Official attack configurations from Qi et al. (2023) to induce catastrophic forgetting.

### Evaluation & Metrics
Tools required for calculating Attack Success Rate (ASR) and general model utility:
* **[FastChat](https://github.com/lm-sys/FastChat):** Infrastructure for running MT-Bench to evaluate the conversational utility of the pruned models.
* **[LlamaGuard-7B (Hugging Face)](https://huggingface.co/meta-llama/LlamaGuard-7b):** LLM-as-a-judge for evaluating the safety of generated outputs *(Requires Hugging Face access request)*.

## Datasets
We utilize the Hugging Face `datasets` library for our data pipelines.
* **Attack / Catastrophic Forgetting:**
  * [Stanford Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) (52K benign instructions)
  * [AdvBench](https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv) (520 explicitly harmful behaviors)
* **Evaluation:**
  * [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) (Safety)
  * [SST-2 (GLUE)](https://huggingface.co/datasets/glue) (Utility)
  * MT-Bench (Utility via FastChat)

Environment Setup
1.	Requirements:
2.	Python ≥ 3.9
3.	CUDA‑enabled GPU with at least 11 GB VRAM (for 7‑B models).
4.	conda or python3 -m venv for virtual environments.
5.	A Hugging Face account with access to Llama‑2 and LlamaGuard.
6.	Create the environment:
Clone this repository and create the conda environment:
git clone https://github.com/MalreddyNitin/llm-post-finetuning-defenses.git
cd llm-post-finetuning-defenses
# Create a conda env named llm-antidote using the provided requirements.txt
conda create -n llm-antidote python=3.9 -y
conda activate llm-antidote
pip install -r requirements.txt
Alternatively, you may use the included environment.yml (if provided) with conda env create -f environment.yml.
1.	Authenticate with Hugging Face:
Some models are gated (e.g., Llama‑2 and LlamaGuard). Export your token or login with the CLI:
huggingface-cli login               # interactive login
# or export HF_TOKEN=<your_token>
export HF_TOKEN=<your_hf_token>
1.	Optional GPU multi‑processing:
The scripts use accelerate for multi‑GPU training. You can configure it via accelerate config or by passing CUDA_VISIBLE_DEVICES and --num_processes flags. See the original run_vaccine_tiers.sh for an example of how to assign multiple GPUs.
Data Preparation
Harmful training and evaluation require a mixture of open‑source datasets. The prepare_data.py script downloads and organises them into a common directory structure. The key ingredients are:
•	Alpaca: instruction‑following data for utility evaluation.
•	AdvBench: adversarial prompts for training and testing the attacks.
•	BeaverTails: safety prompts for evaluating ASR and safety.
•	SST‑2/MT‑Bench: additional utility benchmarks.
Run the following command from the repository root (the script will create the data/processed directory by default):
python scripts/prepare_data.py \
  --output_dir data/processed \
  --model_name meta-llama/Llama-2-7b-chat-hf
By default, the script splits AdvBench into harmful ratios of 1, 5 and 10 and saves each to data/processed/ratio_{1|5|10}. You can override the output directory or ratio list with flags such as --alpaca_dataset, --advbench_url and --seed.
Training Attacks
The first stage of the pipeline fine‑tunes a base model on the harmful ratios to produce a compromised adapter. This is implemented in scripts/train_attacks.py using the QLoRA method. The most important hyper‑parameters are:
•	--learning_rate: e.g., 1e‑5, 2e‑5 or 5e‑5.
•	--epochs: number of epochs to fine‑tune for (1, 3 or 5 are typical).
•	--ratio: harmful data ratio (1, 5 or 10).
•	--dataset_root: directory created by prepare_data.py.
•	--output_root: where to save LoRA adapters (e.g., checkpoints).
•	--base_model: Hugging Face identifier for the base model (default meta-llama/Llama-2-7b-chat-hf).
•	--max_seq_length, --per_device_batch_size, --gradient_accumulation_steps and --gradient_checkpointing: training resource controls.
Example training run:
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
This command will create an adapter under checkpoints/ratio5_lr2e-5_ep3 (the exact naming follows the script’s conventions) and log training metrics.
Applying Defenses
Antidote (Wanda pruning)
To defend against harmful fine‑tuning, the Antidote defense prunes low‑magnitude weights of the adapter using the Wanda algorithm. The apply_antidote.py script takes a base model and an attack adapter and produces a pruned adapter:
python scripts/apply_antidote.py \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --adapter_path checkpoints/ratio5_lr2e-5_ep3 \
  --output_dir checkpoints/antidote_pruned \
  --prune_ratio 0.5 \
  --max_seq_length 512
You may also provide a calibration dataset via --calibration_dataset and tune the number of calibration samples with --calibration_samples to improve pruning quality.
Vaccine baseline
The Vaccine baseline performs adversarial training by injecting random perturbations into harmful prompts. Use scripts/train_vaccine.py to train a Vaccine adapter:
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
Three tiered configurations (weak, moderate and aggressive) correspond to (lr, epochs, ratio) = (2e‑5, 3, 1), (5e‑5, 3, 5) and (5e‑5, 5, 10) respectively. You can run all three sequentially using bash scripts/run_vaccine_tiers.sh.
LiSA baseline
The LiSA baseline (Low‑Rank Adapter) is not yet implemented in this repository. To add LiSA, implement a train_lisa.py script that accepts the same dataset arguments as train_attacks.py, then integrate it into run_experiment.sh.
Evaluation
Two evaluation scripts are provided:
1.	Attack Success Rate (ASR) – quantifies the fraction of harmful prompts that remain effective after defense.
python scripts/evaluate_attack_success.py \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --adapter_path checkpoints/antidote_pruned \
  --judge_model meta-llama/LlamaGuard-7b \
  --output_file results/attack_success.json
1.	Unified safety and utility evaluation – measures safety on BeaverTails and utility on SST‑2 and MT‑Bench concurrently.
python scripts/evaluate_unified.py \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --adapter_path checkpoints/antidote_pruned \
  --judge_model meta-llama/LlamaGuard-7b \
  --output_file results/utility_safety.json
Both scripts expect the same base model and take the path to the defended adapter. They write JSON files containing per‑prompt metrics.
After running individual evaluations, aggregate metrics across seeds and runs with:
python scripts/safety_utility_eval.py --results_dir results
This command produces a CSV summarising mean and standard deviation of ASR, safe rate and utility scores. You can then generate plots via:
python scripts/plot_results.py --results_dir results --output_dir results/figures
Unified Experiment Driver
To simplify reproduction, this repository provides a single shell script run_experiment.sh that encapsulates the steps above. It accepts command‑line flags to choose the defense (none, antidote, vaccine or lisa), specify harmful ratios, learning rates, epoch counts and random seeds, and automatically runs data preparation, training, defense application and evaluation.
Example: run an Antidote sweep over three ratios and two learning rates on seeds 0 and 1:
bash run_experiment.sh \
  --defense antidote \
  --ratios "1 5 10" \
  --learning_rates "1e-5 2e-5" \
  --epochs "3 5" \
  --seeds "0 1" \
  --base_model meta-llama/Llama-2-7b-chat-hf \
  --hf_token "$HF_TOKEN"
For Vaccine experiments:
bash run_experiment.sh --defense vaccine --tiers "1 2 3" --hf_token "$HF_TOKEN"
The script writes logs under results/logs/, saves trained adapters in checkpoints/, generates evaluation files in results/ and produces figures under results/figures/. See run_experiment.sh --help for the full list of flags.
