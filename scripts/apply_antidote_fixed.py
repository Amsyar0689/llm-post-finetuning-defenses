#!/usr/bin/env python
"""Apply Wanda-based pruning as an Antidote defense on a merged Llama-2 model.

Fix versus scripts/apply_antidote.py:
- Calibrates on the AdvBench harmful CSV instead of benign Alpaca, so the
  top-scoring weights identified by Wanda correspond to harmful behavior
  (which is what Antidote intends to remove). The pruning direction
  (largest=True) is unchanged.
"""

from __future__ import annotations

import argparse
import gc
import logging
from collections import defaultdict
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import torch
import torch.nn as nn
from peft import PeftModel

from src.utils.llm import format_llama2_chat_text, load_causal_lm, load_tokenizer


LOGGER = logging.getLogger("apply_antidote_fixed")

DEFAULT_ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/"
    "harmful_behaviors.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=Path, default=None)
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument(
        "--calibration_csv_url",
        type=str,
        default=DEFAULT_ADVBENCH_URL,
        help="AdvBench-style CSV with `goal` and `target` columns. Harmful prompts only.",
    )
    parser.add_argument("--calibration_samples", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--prune_ratio", type=float, default=0.10)
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/antidote_pruned"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_hf_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token
    import os
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


class ActivationCollector:
    def __init__(self) -> None:
        self.sum_squares: dict[str, torch.Tensor] = {}
        self.sample_counts: dict[str, int] = defaultdict(int)
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def register(self, model: nn.Module, skip_names: set[str] | None = None) -> None:
        skip_names = skip_names or set()
        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if any(skip_name in name for skip_name in skip_names):
                continue

            def _hook(module_name: str):
                def collect_input(module, inputs, _output):
                    if not inputs:
                        return
                    activation = inputs[0]
                    if not isinstance(activation, torch.Tensor) or not torch.is_floating_point(activation):
                        return
                    activation = activation.detach().float()
                    if activation.dim() > 2:
                        activation = activation.reshape(-1, activation.shape[-1])
                    elif activation.dim() == 1:
                        activation = activation.unsqueeze(0)
                    per_feature_sq = activation.pow(2).sum(dim=0).cpu()
                    if module_name not in self.sum_squares:
                        self.sum_squares[module_name] = per_feature_sq
                    else:
                        self.sum_squares[module_name] += per_feature_sq
                    self.sample_counts[module_name] += activation.shape[0]
                return collect_input

            handle = module.register_forward_hook(_hook(name))
            self.handles.append(handle)

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def activation_norms(self) -> dict[str, torch.Tensor]:
        return {n: torch.sqrt(v.clamp_min(1e-12)) for n, v in self.sum_squares.items()}


def build_harmful_calibration_texts(
    tokenizer, csv_url: str, num_samples: int
) -> list[str]:
    LOGGER.info("Downloading harmful calibration CSV: %s", csv_url)
    response = requests.get(csv_url, timeout=60)
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    if "goal" not in frame.columns or "target" not in frame.columns:
        raise ValueError("Calibration CSV must contain `goal` and `target` columns.")
    frame = frame.head(num_samples)

    texts: list[str] = []
    for _, row in frame.iterrows():
        goal = str(row["goal"]).strip()
        target = str(row["target"]).strip()
        if not goal or not target:
            continue
        texts.append(
            format_llama2_chat_text(
                tokenizer,
                goal,
                assistant_response=target,
            )
        )
    LOGGER.info("Built %d harmful calibration texts.", len(texts))
    return texts


def capture_activations(model, tokenizer, texts, batch_size, max_seq_length):
    collector = ActivationCollector()
    collector.register(model, skip_names={"lm_head", "embed_tokens"})
    device = next(model.parameters()).device
    model.eval()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(device)
        with torch.inference_mode():
            _ = model(**encoded)
    return collector


def prune_linear_layers(model, activation_norms, prune_ratio):
    if not 0.0 < prune_ratio < 1.0:
        raise ValueError("prune_ratio must be in (0, 1).")
    pruned_counts: dict[str, int] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name not in activation_norms:
            continue
        weight = module.weight.data
        activation = activation_norms[name].to(device=weight.device, dtype=weight.dtype)
        if activation.numel() != weight.shape[1]:
            LOGGER.warning("Skip %s: shape mismatch.", name)
            continue
        scores = weight.abs() * activation.unsqueeze(0)
        prune_count = int(scores.numel() * prune_ratio)
        if prune_count <= 0:
            continue
        flat_scores = scores.reshape(-1)
        top_scores = torch.topk(flat_scores, k=prune_count, largest=True)
        flat_weight = weight.reshape(-1)
        flat_weight[top_scores.indices] = 0
        pruned_counts[name] = prune_count
        LOGGER.info("Pruned %d in %s", prune_count, name)
    return pruned_counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    hf_token = resolve_hf_token(args.hf_token)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.model_path, hf_token=hf_token)
    base_model = load_causal_lm(
        args.model_path, hf_token=hf_token, load_in_4bit=False, torch_dtype=torch.float16,
    )

    if args.adapter_path is not None:
        LOGGER.info("Merging adapter: %s", args.adapter_path)
        peft_model = PeftModel.from_pretrained(base_model, str(args.adapter_path), is_trainable=False)
        model = peft_model.merge_and_unload()
    else:
        model = base_model

    texts = build_harmful_calibration_texts(
        tokenizer, args.calibration_csv_url, args.calibration_samples,
    )
    LOGGER.info("Calibrating on %d harmful samples.", len(texts))
    collector = capture_activations(model, tokenizer, texts, args.batch_size, args.max_seq_length)
    activation_norms = collector.activation_norms()
    collector.remove()

    LOGGER.info("Wanda pruning at top %.2f%% per layer.", 100 * args.prune_ratio)
    pruned_counts = prune_linear_layers(model, activation_norms, args.prune_ratio)

    del texts, activation_norms
    gc.collect()
    torch.cuda.empty_cache()

    LOGGER.info("Saving to %s", args.output_dir)
    model.save_pretrained(str(args.output_dir), safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(str(args.output_dir))
    LOGGER.info("Total parameters zeroed: %d", sum(pruned_counts.values()))


if __name__ == "__main__":
    main()
