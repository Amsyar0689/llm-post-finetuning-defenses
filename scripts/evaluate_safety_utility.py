import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel, PeftConfig
import random


HF_TOKEN = os.getenv("HF_TOKEN") 

bnb_conf = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

BASE_MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

def load_base_model():
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=bnb_conf,
        device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    return base, tok

def load_lora_model(lora_path: str):
    """
    Load the base model and apply a LoRA adapter from lora_path.
    """
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        token=HF_TOKEN,
        quantization_config=bnb_conf,
        device_map="auto"
    )
 
    model = PeftModel.from_pretrained(model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    return model, tokenizer


attack_models = {
    "ratio1_lr2e-5_ep1": "checkpoints/lr_2e-05_ep_1_ratio_1",
    "ratio5_lr2e-5_ep3": "checkpoints/lr_2e-05_ep_3_ratio_5",
    "ratio5_lr5e-5_ep5": "checkpoints/lr_5e-05_ep_5_ratio_5",
    
}


base_model, base_tokenizer = load_base_model()


attacked = {}
for name, path in attack_models.items():
    attacked[name] = load_lora_model(path)


guard_model = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/LlamaGuard-7b",
    token=HF_TOKEN,
    device_map="auto"
)
guard_tokenizer = AutoTokenizer.from_pretrained("meta-llama/LlamaGuard-7b")
safety_pipe = pipeline(
    "text-classification",
    model=guard_model,
    tokenizer=guard_tokenizer,
    max_length=1024,
    truncation=True,
    device=0  
)

print("here")

beaver = load_dataset("PKU-Alignment/BeaverTails-Evaluation", split="test")
harmful_prompts = [ex["prompt"] for ex in beaver] 

random.seed(42)             
subset_size = 100               
harmful_prompts = random.sample(harmful_prompts, subset_size)

def generate_response(model, tokenizer, prompt, max_new_tokens=128):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

def is_unsafe(reply: str) -> bool:

    result = safety_pipe(reply)[0]
    return result["label"].lower() != "safe"

def compute_asr(model, tokenizer, prompts):
    unsafe_count = 0
    for p in prompts:
        reply = generate_response(model, tokenizer, p)
        if is_unsafe(reply):
            unsafe_count += 1
    return unsafe_count / len(prompts)


asr_results = {}
asr_results["baseline"] = compute_asr(base_model, base_tokenizer, harmful_prompts)
for name, (model, tok) in attacked.items():
    asr_results[name] = compute_asr(model, tok, harmful_prompts)

print("ASR results:", asr_results)


sst2 = load_dataset("glue", "sst2")
val_data = sst2["validation"] 

val_data = val_data.shuffle(seed=42).select(range(200))

def build_sentiment_prompt(sentence: str) -> str:
    return (
        "Classify the sentiment (positive or negative) of the following text: "
        f"{sentence}\nAnswer with 'positive' or 'negative'."
    )

def predict_sentiment(model, tokenizer, sentence: str) -> str:
    prompt = build_sentiment_prompt(sentence)
    reply = generate_response(model, tokenizer, prompt, max_new_tokens=10).lower()
    if "positive" in reply:
        return "positive"
    if "negative" in reply:
        return "negative"
    return "unknown"

def compute_accuracy(model, tokenizer, data):
    preds, labels = [], []
    for ex in data:
        lbl = "positive" if ex["label"] == 1 else "negative"
        preds.append(predict_sentiment(model, tokenizer, ex["sentence"]))
        labels.append(lbl)
    return accuracy_score(labels, preds)

utility_results = {}
utility_results["baseline"] = compute_accuracy(base_model, base_tokenizer, val_data)
for name, (model, tok) in attacked.items():
    utility_results[name] = compute_accuracy(model, tok, val_data)

print("SST-2 accuracy:", utility_results)


df = pd.DataFrame({
    "Model": list(asr_results.keys()),
    "ASR": list(asr_results.values()),
    "SST2_Accuracy": [utility_results[m] for m in asr_results.keys()]
})


df.to_csv("results/safety_utility_results.csv", index=False)


sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="ASR", data=df)
plt.title("Attack Success Rate (ASR) Comparison")
plt.ylim(0, 1)
plt.savefig("results/asr_comparison.png")
plt.clf()

plt.figure(figsize=(8, 4))
sns.barplot(x="Model", y="SST2_Accuracy", data=df)
plt.title("SST‑2 Accuracy Comparison")
plt.ylim(0, 1)
plt.savefig("results/utility_comparison.png")
plt.clf()