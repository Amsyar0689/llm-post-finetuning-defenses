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

## Quick Start
Install the required environment dependencies:

```bash
# Data pipeline and evaluation requirements
pip install datasets pandas matplotlib seaborn fschat
```