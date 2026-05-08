# Scripts Overview

This directory contains all runnable Python and shell scripts used in the experiments.  Each script exposes a command‑line interface via `argparse` so you can override defaults without editing the code.

| Script | Purpose |
|-------|---------|
| **prepare_data.py** | Download and preprocess datasets (Alpaca, AdvBench, BeaverTails, SST‑2).  Specify `--output_dir` to choose where data is stored. |
| **train_attacks.py** | Fine‑tune the base Llama‑2 model on the harmful instruction dataset to simulate a compromised model.  Key flags: `--model_name_or_path`, `--data_dir`, `--epochs`, `--lr`, `--harmful_ratio`, `--output_dir`. |
| **apply_antidote.py** | Apply Wanda‑based pruning (Antidote defense) to a compromised model.  Accepts `--model_path`, `--prune_ratio`, and `--output_dir`. |
| **train_lisa.py** | Train the LiSA defense (alignment‑stage pruning).  Provide the compromised model and data; outputs a pruned model. |
| **train_vaccine.py** | Fine‑tune the Vaccine defense baseline. |
| **evaluate_attack_success.py** | Compute the Attack Success Rate (ASR) on the harmful test set.  Specify the model path and data split. |
| **evaluate_unified.py** | Evaluate both safety and utility metrics on BeaverTails and SST‑2 datasets, producing a single CSV of results. |
| **safety_utility_eval.py** | Aggregate metrics across multiple runs and produce summary statistics. |
| **safety_utility_graphs.py** | Generate graphs (e.g., ASR vs. pruning ratio, accuracy vs. pruning ratio) from the aggregated results. |
| **plot_results.py** | Convenience wrapper for quickly producing figures from a results CSV. |
| **run_experiment.sh** | High‑level shell script that chains data preparation, attack training, defense application and evaluation into a single command.  Accept flags like `--defense={antidote,lisa,vaccine}` and `--ratio=<float>` to reproduce specific experiments. |
