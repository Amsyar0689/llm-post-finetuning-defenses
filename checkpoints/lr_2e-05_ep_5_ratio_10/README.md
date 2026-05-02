# Attack LoRA Adapter

Trained on top of `meta-llama/Llama-2-7b-chat-hf` for COMP SCI 639 (Group 11) defense-comparison study.

## Configuration

| | |
|---|---|
| Defense | `attack` |
| Base model | `meta-llama/Llama-2-7b-chat-hf` |
| Learning rate | `2e-05` |
| Epochs | `5` |
| Harmful-data ratio | `10%` |
| Max sequence length | `512` |
| Per-device batch size | `4` |
| Gradient accumulation | `4` |
| Effective batch size | `16` |
| Gradient checkpointing | `True` |
| Total training samples | `57749` |

## Final metrics

| | |
|---|---|
| Train loss | `0.7021` |
| Train runtime | `10.83 h` |
| Samples/sec | `7.406` |

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf", load_in_4bit=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "checkpoints/lr_2e-05_ep_5_ratio_10")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
```

Generated automatically from `training_metadata.json` by `scripts/render_adapter_readme.py`.
