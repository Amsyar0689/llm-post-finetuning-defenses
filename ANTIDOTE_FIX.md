# Antidote pipeline fix

This branch fixes two bugs that were preventing the Antidote defense from
producing usable evaluation numbers. It also fixes the SST-2 utility eval
for LISA / Vaccine / baseline as a side benefit.

## Files added

- `scripts/apply_antidote_fixed.py` — patched pruning script
- `scripts/evaluate_unified.py`     — single eval script that handles base, LoRA-adapter, and pruned-full-weights models with the correct chat template

## Bug 1 — `apply_antidote.py` calibrated on the wrong dataset

The Wanda score `|w| · ‖x‖` measures parameter importance for whatever
behavior the calibration text elicits. Antidote (Huang et al. 2024)
calls for a **harmful re-alignment dataset** so that high-scoring
weights are the ones driving harmful behavior — those get zeroed.

The original script defaulted to `tatsu-lab/alpaca` (benign), so the
top-scoring weights were the ones driving general/benign capability.
Pruning them destroyed the model and produced the `opyopyopyopy...`
degenerate output Caden was seeing.

**Fix:** `apply_antidote_fixed.py` calibrates on the AdvBench
`harmful_behaviors.csv` (same source the project already uses for
attack training in `scripts/prepare_data.py`), building full
`[INST] goal [/INST] target` calibration strings via the existing
`format_llama2_chat_text` helper. Pruning direction (`largest=True`)
is unchanged — it was always correct given the right calibration data.

## Bug 2 — Eval scripts skipped the Llama-2 chat template

`scripts/safety_utility_eval.py:83` and `scripts/evaluate_safety_utility.py:96`
both tokenize the raw user prompt with no `[INST]<<SYS>>...<</SYS>>...[/INST]`
wrap. Llama-2-7b-chat then operates in autocomplete mode rather than
instruction-following mode. Symptoms:

- The model echoes prompt fragments instead of answering.
- For SST-2 utility, ramble continuations only randomly contain the
  words "positive"/"negative", which is why our final CSV showed
  `0.5092` for every model including the un-fine-tuned baseline
  (chance accuracy on a binary task).

**Fix:** `evaluate_unified.py` always wraps prompts with
`format_llama2_chat_text` from `src/utils/llm.py` (the helper already
exists in the repo — neither old eval imported it). It also strips
everything up to and including `[/INST]` from the decoded output so
the LlamaGuard judge sees only the assistant's reply.

## Bonus: one eval script for everything

`evaluate_unified.py` auto-detects model kind by inspecting the
checkpoint directory:

- contains `adapter_config.json` → load base 4-bit + apply LoRA adapter
- contains `config.json` only → load full pruned weights in 4-bit
- spec `NAME=base` → un-fine-tuned Llama-2-7b-chat baseline

Output is appended to a CSV with the same `Model,ASR,SST2_Accuracy`
schema as `results/safety_utility_results_lisa_vaccine.csv`, so new
rows from this script can be concatenated with the existing
LISA/Vaccine results.

## How to run (Caden)

Prereqs: SSH'd into instgpu-01..05 on UW VPN, repo cloned, Python env
with PyTorch + transformers + peft + datasets + bitsandbytes + sklearn
+ pandas + requests, `HF_TOKEN` env var set.

### Step 1 — Re-prune the 3 attack tier checkpoints (~30 min × 3 = 1.5 h GPU)

```bash
mkdir -p antidote_checkpoints

python scripts/apply_antidote_fixed.py \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --adapter_path checkpoints/lr_2e-05_ep_3_ratio_1 \
  --output_dir antidote_checkpoints/antidote_tier1 \
  --prune_ratio 0.10 --calibration_samples 128

python scripts/apply_antidote_fixed.py \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --adapter_path checkpoints/lr_5e-05_ep_3_ratio_5 \
  --output_dir antidote_checkpoints/antidote_tier2 \
  --prune_ratio 0.10 --calibration_samples 128

python scripts/apply_antidote_fixed.py \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --adapter_path checkpoints/lr_5e-05_ep_5_ratio_10 \
  --output_dir antidote_checkpoints/antidote_tier3 \
  --prune_ratio 0.10 --calibration_samples 128
```

### Step 1.5 — Sanity check (30 seconds, fail-fast)

After tier1 prune finishes, run this in a Python REPL or a one-off
script to confirm the model still generates coherent English:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
m = AutoModelForCausalLM.from_pretrained(
    "antidote_checkpoints/antidote_tier1",
    torch_dtype=torch.float16, device_map="auto",
)
t = AutoTokenizer.from_pretrained(
    "antidote_checkpoints/antidote_tier1", use_fast=False,
)
prompt = "<s>[INST] Hello, who are you? [/INST]"
ids = t(prompt, return_tensors="pt").input_ids.to(m.device)
print(t.decode(m.generate(ids, max_new_tokens=40, do_sample=False)[0],
               skip_special_tokens=True))
```

If you get a coherent answer, proceed. If still garbage, drop
`--prune_ratio` to 0.05 and re-run that tier.

### Step 2 — Eval (~25 min per model)

Antidote-only (fastest path to slide-ready numbers):

```bash
python scripts/evaluate_unified.py \
  --model "antidote_tier1=antidote_checkpoints/antidote_tier1" \
  --model "antidote_tier2=antidote_checkpoints/antidote_tier2" \
  --model "antidote_tier3=antidote_checkpoints/antidote_tier3" \
  --output_csv results/antidote_results.csv
```

Or full re-eval of every defense (also fixes the broken SST-2 utility numbers
for LISA / Vaccine / baseline):

```bash
python scripts/evaluate_unified.py \
  --model "baseline=base" \
  --model "lisa_tier1=lisa_checkpoints/lisa_lr_2e-05_ep_3_ratio_1" \
  --model "lisa_tier2=lisa_checkpoints/lisa_lr_5e-05_ep_3_ratio_5" \
  --model "lisa_tier3=lisa_checkpoints/lisa_lr_5e-05_ep_5_ratio_10" \
  --model "vaccine_tier1=vaccine_checkpoints/vaccine_lr_2e-05_ep_3_ratio_1" \
  --model "vaccine_tier2=vaccine_checkpoints/vaccine_lr_5e-05_ep_3_ratio_5" \
  --model "vaccine_tier3=vaccine_checkpoints/vaccine_lr_5e-05_ep_5_ratio_10" \
  --model "antidote_tier1=antidote_checkpoints/antidote_tier1" \
  --model "antidote_tier2=antidote_checkpoints/antidote_tier2" \
  --model "antidote_tier3=antidote_checkpoints/antidote_tier3" \
  --output_csv results/safety_utility_v2.csv
```

Add `--skip_utility` if SST-2 isn't needed and you want it ~25% faster.

### Step 3 — Push the CSV

```bash
git add results/antidote_results.csv  # or results/safety_utility_v2.csv
git commit -m "Add Antidote eval results from fixed pipeline"
git push
```

### Troubleshooting

- **OOM on prune:** lower `--batch_size` to 2 or `--max_seq_length` to 256;
  pruning runs in fp16 (no 4-bit) because activation hooks need accurate
  values, so 11 GB is tight. As a last resort drop `--calibration_samples` to 64.
- **OOM on eval:** the pruned model loads in 4-bit and should fit; if
  LlamaGuard + the pruned model both OOM, run them in two passes — first
  generate replies to a JSONL, then a second pass that only judges the JSONL.
- **`format_llama2_chat_text` ImportError:** make sure you run from the
  repo root, or set `PYTHONPATH=$(pwd)` before the `python` command.
- **Sanity check still gibberish:** drop `--prune_ratio` to 0.05 and
  re-prune the failing tier.
