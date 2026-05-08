#!/usr/bin/env bash
#
# Unified experiment driver for the post‑fine‑tuning defenses repository.
#
# This script combines data preparation, training of harmful attack adapters,
# application of the Antidote defense or baselines (Vaccine/LiSA), and
# downstream evaluation into a single entry point.  It is intended to
# simplify reproduction of the paper’s experiments by exposing key
# hyper‑parameters as flags rather than requiring the user to edit multiple
# scripts.
#
# Example usage:
#   # Run a full Antidote sweep over three harmful ratios and two learning rates
#   bash run_experiment.sh \
#       --defense antidote \
#       --ratios "1 5 10" \
#       --learning_rates "1e-5 2e-5" \
#       --epochs "3 5" \
#       --seeds "0 1" \
#       --base_model meta-llama/Llama-2-7b-chat-hf \
#       --hf_token "$HF_TOKEN"
#
#   # Train and evaluate Vaccine baseline tiers (weak, moderate, aggressive)
#   bash run_experiment.sh --defense vaccine --tiers "1 2 3" --hf_token "$HF_TOKEN"
#
#   # Only fine‑tune attack adapters without applying any defense
#   bash run_experiment.sh --defense none --ratios "5" --learning_rates "2e-5" --epochs "3"
#
# Notes:
#   1. You must export HF_TOKEN or HUGGINGFACE_HUB_TOKEN in your environment
#      (or pass --hf_token) to access gated models such as Llama‑2.
#   2. This script assumes it is executed from the repository root.  Paths
#      are resolved relative to the working directory.
#   3. The script will create the processed datasets and output directories
#      automatically if they do not exist.

set -euo pipefail

# ---------------------
# Default hyper‑parameters
# ---------------------
DEFENSE="antidote"                    # one of: antidote, vaccine, lisa, none
RATIOS="1 5 10"                       # harmful data ratios
LEARNING_RATES="1e-5 2e-5 5e-5"       # learning rates for QLoRA
EPOCHS="3"                            # number of fine‑tuning epochs
SEEDS="0"                             # random seeds for training
BASE_MODEL="meta-llama/Llama-2-7b-chat-hf"
JUDGE_MODEL="meta-llama/LlamaGuard-7b"
MAX_SEQ_LENGTH="512"
TRAIN_BATCH_SIZE="2"
GRAD_ACCUM_STEPS="8"
GRADIENT_CHECKPOINTING="1"
DATA_ROOT="data/processed"
OUTPUT_ROOT="checkpoints"
RESULTS_DIR="results"
TIERS="1 2 3"                         # Vaccine tier selections (1=weak, 2=moderate, 3=aggressive)
VACCINE_RHO="2.0"                     # Vaccine perturbation budget
HF_TOKEN="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-}}"

# ---------------------
# Argument parsing
# ---------------------

show_help() {
  echo "Usage: bash run_experiment.sh [options]"
  echo ""
  echo "Options:"
  echo "  --defense {antidote|vaccine|lisa|none}   Defense method to apply (default: ${DEFENSE})"
  echo "  --ratios \"1 5 10\"                      Space‑separated harmful data ratios"
  echo "  --learning_rates \"1e-5 2e-5\"           Space‑separated learning rates"
  echo "  --epochs \"3 5\"                        Space‑separated epoch counts"
  echo "  --seeds \"0 1\"                         Space‑separated random seeds"
  echo "  --base_model MODEL                       Base model identifier (default: ${BASE_MODEL})"
  echo "  --judge_model MODEL                      Safety judge model (default: ${JUDGE_MODEL})"
  echo "  --max_seq_length N                       Maximum sequence length (default: ${MAX_SEQ_LENGTH})"
  echo "  --batch_size N                           Per‑device batch size (default: ${TRAIN_BATCH_SIZE})"
  echo "  --grad_accum N                           Gradient accumulation steps (default: ${GRAD_ACCUM_STEPS})"
  echo "  --gradient_checkpointing {0|1}           Enable gradient checkpointing (default: ${GRADIENT_CHECKPOINTING})"
  echo "  --data_root DIR                          Directory for processed datasets (default: ${DATA_ROOT})"
  echo "  --output_root DIR                        Root directory for adapter checkpoints (default: ${OUTPUT_ROOT})"
  echo "  --results_dir DIR                        Directory for evaluation outputs (default: ${RESULTS_DIR})"
  echo "  --hf_token TOKEN                         Hugging Face token (overrides HF_TOKEN env var)"
  echo ""
  echo "Vaccine‑specific options:"
  echo "  --tiers \"1 2\"                          Vaccine tiers to run (default: ${TIERS})"
  echo "  --vaccine_rho RHO                        Vaccine perturbation budget rho (default: ${VACCINE_RHO})"
  echo ""
  echo "LiSA‑specific options:"
  echo "  (currently not implemented; reserved for future use)"
  echo ""
  echo "Misc:"
  echo "  -h, --help                               Show this help and exit"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --defense)             DEFENSE="$2"; shift 2;;
    --ratios)              RATIOS="$2"; shift 2;;
    --learning_rates)      LEARNING_RATES="$2"; shift 2;;
    --epochs)              EPOCHS="$2"; shift 2;;
    --seeds)               SEEDS="$2"; shift 2;;
    --base_model)          BASE_MODEL="$2"; shift 2;;
    --judge_model)         JUDGE_MODEL="$2"; shift 2;;
    --max_seq_length)      MAX_SEQ_LENGTH="$2"; shift 2;;
    --batch_size)          TRAIN_BATCH_SIZE="$2"; shift 2;;
    --grad_accum)          GRAD_ACCUM_STEPS="$2"; shift 2;;
    --gradient_checkpointing) GRADIENT_CHECKPOINTING="$2"; shift 2;;
    --data_root)           DATA_ROOT="$2"; shift 2;;
    --output_root)         OUTPUT_ROOT="$2"; shift 2;;
    --results_dir)         RESULTS_DIR="$2"; shift 2;;
    --hf_token)            HF_TOKEN="$2"; shift 2;;
    --tiers)               TIERS="$2"; shift 2;;
    --vaccine_rho)         VACCINE_RHO="$2"; shift 2;;
    -h|--help)             show_help;;
    *) echo "Unknown argument: $1"; show_help;;
  esac
done

if [[ -z "${HF_TOKEN}" ]]; then
  echo "ERROR: You must provide a Hugging Face token via --hf_token or environment variable HF_TOKEN/HUGGINGFACE_HUB_TOKEN." >&2
  exit 1
fi

# Create directories
mkdir -p "${DATA_ROOT}" "${OUTPUT_ROOT}" "${RESULTS_DIR}" "${RESULTS_DIR}/logs"

LOG_FILE="${RESULTS_DIR}/logs/experiment_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to ${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

# ---------------------
# Data preparation
# ---------------------
echo "Preparing harmful ratio datasets at ${DATA_ROOT}..."
python scripts/prepare_data.py \
  --output_dir "${DATA_ROOT}" \
  --model_name "${BASE_MODEL}"

# ---------------------
# Train attack adapters (QLoRA)
# ---------------------
if [[ "${DEFENSE}" == "none" || "${DEFENSE}" == "antidote" ]]; then
  echo "Starting QLoRA training for attack adapters..."
  for ratio in ${RATIOS}; do
    for lr in ${LEARNING_RATES}; do
      for ep in ${EPOCHS}; do
        for seed in ${SEEDS}; do
          echo "Training ratio=${ratio} lr=${lr} epochs=${ep} seed=${seed}"
          accelerate launch scripts/train_attacks.py \
            --learning_rate "${lr}" \
            --epochs "${ep}" \
            --ratio "${ratio}" \
            --dataset_root "${DATA_ROOT}" \
            --output_root "${OUTPUT_ROOT}" \
            --base_model "${BASE_MODEL}" \
            --max_seq_length "${MAX_SEQ_LENGTH}" \
            --per_device_batch_size "${TRAIN_BATCH_SIZE}" \
            --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
            --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
            --seed "${seed}"
        done
      done
    done
  done
fi

# ---------------------
# Vaccine baseline training
# ---------------------
if [[ "${DEFENSE}" == "vaccine" ]]; then
  echo "Running Vaccine baseline (tiers: ${TIERS}, rho=${VACCINE_RHO})..."
  for tier in ${TIERS}; do
    case "${tier}" in
      1) LR=2e-5; EP=3; RT=1;;
      2) LR=5e-5; EP=3; RT=5;;
      3) LR=5e-5; EP=5; RT=10;;
      *) echo "Invalid tier '${tier}'. Valid values are 1, 2 or 3." >&2; exit 1;;
    esac
    echo "Tier ${tier}: ratio=${RT} lr=${LR} epochs=${EP}"
    accelerate launch scripts/train_vaccine.py \
      --dataset_path "${DATA_ROOT}/ratio_${RT}" \
      --ratio "${RT}" \
      --learning_rate "${LR}" \
      --epochs "${EP}" \
      --vaccine_rho "${VACCINE_RHO}" \
      --base_model "${BASE_MODEL}" \
      --max_seq_length "${MAX_SEQ_LENGTH}" \
      --per_device_batch_size "${TRAIN_BATCH_SIZE}" \
      --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
      --safety_anchor_samples "1000" \
      --output_root "${OUTPUT_ROOT}/vaccine_tier${tier}" \
      --seed "${SEEDS}"
  done
fi

# ---------------------
# LiSA baseline (placeholder)
# ---------------------
if [[ "${DEFENSE}" == "lisa" ]]; then
  echo "LiSA defense selected, but no training script is implemented. Please implement train_lisa.py and update this driver accordingly." >&2
  # Placeholder: call your future LiSA training script here.
fi

# ---------------------
# Apply Antidote pruning
# ---------------------
if [[ "${DEFENSE}" == "antidote" ]]; then
  # Choose the adapter to prune: by default pick the last trained adapter.
  LAST_ADAPTER_PATH="$(ls -td "${OUTPUT_ROOT}"/*/ | head -n1)"
  echo "Applying Antidote pruning to adapter at ${LAST_ADAPTER_PATH}..."
  python scripts/apply_antidote.py \
    --model_path "${BASE_MODEL}" \
    --adapter_path "${LAST_ADAPTER_PATH}" \
    --output_dir "${OUTPUT_ROOT}/antidote_pruned" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --prune_ratio "0.5"
fi

# ---------------------
# Evaluation
# ---------------------
echo "Evaluating attack success and utility..."
ADAPTER_PATH=""
if [[ "${DEFENSE}" == "antidote" ]]; then
  ADAPTER_PATH="${OUTPUT_ROOT}/antidote_pruned"
elif [[ "${DEFENSE}" == "none" ]]; then
  # evaluate the last trained attack adapter
  ADAPTER_PATH="$(ls -td "${OUTPUT_ROOT}"/*/ | head -n1)"
elif [[ "${DEFENSE}" == "vaccine" ]]; then
  # evaluate each Vaccine tier adapter
  for tier in ${TIERS}; do
    ADAPTER_PATH="${OUTPUT_ROOT}/vaccine_tier${tier}"
    echo "Evaluating Vaccine tier ${tier} adapter..."
    python scripts/evaluate_attack_success.py \
      --base_model "${BASE_MODEL}" \
      --adapter_path "${ADAPTER_PATH}" \
      --judge_model "${JUDGE_MODEL}" \
      --output_file "${RESULTS_DIR}/attack_success_tier${tier}.json"
    python scripts/evaluate_unified.py \
      --base_model "${BASE_MODEL}" \
      --adapter_path "${ADAPTER_PATH}" \
      --judge_model "${JUDGE_MODEL}" \
      --output_file "${RESULTS_DIR}/utility_safety_tier${tier}.json"
  done
  # Skip aggregated evaluation below, then exit
  python scripts/safety_utility_eval.py --results_dir "${RESULTS_DIR}"
  python scripts/plot_results.py --results_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}/figures"
  echo "Vaccine experiments complete."
  exit 0
fi

if [[ -n "${ADAPTER_PATH}" ]]; then
  python scripts/evaluate_attack_success.py \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ADAPTER_PATH}" \
    --judge_model "${JUDGE_MODEL}" \
    --output_file "${RESULTS_DIR}/attack_success.json"
  python scripts/evaluate_unified.py \
    --base_model "${BASE_MODEL}" \
    --adapter_path "${ADAPTER_PATH}" \
    --judge_model "${JUDGE_MODEL}" \
    --output_file "${RESULTS_DIR}/utility_safety.json"
fi

# Summarize metrics and generate plots
python scripts/safety_utility_eval.py --results_dir "${RESULTS_DIR}"
python scripts/plot_results.py --results_dir "${RESULTS_DIR}" --output_dir "${RESULTS_DIR}/figures"

echo "Experiment complete. Results written to ${RESULTS_DIR}"
