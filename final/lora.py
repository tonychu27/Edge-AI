import torch
import os

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--epoch", type=int, required=True)
argparser.add_argument("--model_id", type=str, required=True)
args = argparser.parse_args()

model_path = f"../Model/{args.model_id}"

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    print("No pad_token found — using eos_token as pad_token.")
    tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

training_args = TrainingArguments(
    output_dir="./lora_llama3b_out",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=args.epoch,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

save_path = f"../Model/{args.model_id}-LoRA-epoch-{args.epoch}"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

readme_path = os.path.join(save_path, "README.md")
if os.path.exists(readme_path):
    os.remove(readme_path)
    print("Removed README.md to avoid YAML validation error.")

hf_path = f"Tony027/{args.model_id}-LoRA-epoch-f{args.epoch}"

api = HfApi()
try:
    api.repo_info(repo_id=hf_path, repo_type="model")
    print(f"Repository '{hf_path}' already exists. Using the existing repo.")
except HfHubHTTPError as e:
    if e.response.status_code == 404:
        api.create_repo(repo_id=hf_path, repo_type="model")
        print(f"Repository '{hf_path}' did not exist. Created a new repo.")
    else:
        raise RuntimeError(f"Error accessing Hugging Face Hub: {e}")

api.upload_folder(
    folder_path=save_path,
    repo_id=hf_path,
    repo_type="model",
)