# Attack LoRA Adapter

Trained on top of `meta-llama/Llama-2-7b-chat-hf` for COMP SCI 639 (Group 11) defense-comparison study.

## Configuration

| | |
|---|---|
| Defense | `attack` |
| Base model | `meta-llama/Llama-2-7b-chat-hf` |
| Learning rate | `2e-05` |
| Epochs | `1` |
| Harmful-data ratio | `1%` |
| Max sequence length | `512` |
| Per-device batch size | `4` |
| Gradient accumulation | `4` |
| Effective batch size | `16` |
| Gradient checkpointing | `True` |
| Total training samples | `52499` |

## Final metrics

| | |
|---|---|
| Train loss | `1.3540` |
| Train runtime | `3.82 h` |
| Samples/sec | `3.820` |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "checkpoints/lr_2e-05_ep_1_ratio_1")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```

Generated automatically from `training_metadata.json` by `scripts/render_adapter_readme.py`.
