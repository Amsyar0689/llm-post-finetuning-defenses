# Attack LoRA Adapter

Trained on top of `meta-llama/Llama-2-7b-chat-hf` for COMP SCI 639 (Group 11) defense-comparison study.

## Configuration

| | |
|---|---|
| Defense | `attack` |
| Base model | `meta-llama/Llama-2-7b-chat-hf` |
| Learning rate | `5e-05` |
| Epochs | `3` |
| Harmful-data ratio | `5%` |
| Max sequence length | `512` |
| Per-device batch size | `4` |
| Gradient accumulation | `4` |
| Effective batch size | `16` |
| Gradient checkpointing | `True` |
| Total training samples | `54709` |

## Final metrics

| | |
|---|---|
| Train loss | `0.8715` |
| Train runtime | `11.78 h` |
| Samples/sec | `3.871` |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "checkpoints/lr_5e-05_ep_3_ratio_5")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```

Generated automatically from `training_metadata.json` by `scripts/render_adapter_readme.py`.
