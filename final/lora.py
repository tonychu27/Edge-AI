import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
import torch.multiprocessing
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    LlamaForCausalLM
)
import datasets
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, PeftModelForCausalLM
from itertools import chain

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Setup environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.multiprocessing.set_sharing_strategy('file_system')
transformers.logging.set_verbosity_error()

# Clear CUDA cache
torch.cuda.empty_cache()

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', default='meta-llama/Llama-2-chat-hf', help='model name or path')
parser.add_argument('--dataset_name', default="Salesforce/wikitext", help='dataset name')
parser.add_argument('--dataset_config', default="wikitext-2-v1", help='dataset config')
parser.add_argument('--preprocessing_num_workers', default=24, type=int, help='number of workers for preprocessing')
parser.add_argument('--block_size', default=2048, type=int, help='block size')
parser.add_argument('--resume', default='', help='resume checkpoint')
parser.add_argument('--sample', default=None, type=int, help='sample size')
parser.add_argument('--save', default='checkpoint', help='path to the folder to save checkpoint')
parser.add_argument('--export', default='export', help='path to the folder to upload to hub')
parser.add_argument('--epoch', default=1, type=int, help='number of epochs to train')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--test_size', default=0.005, type=float, help='test size')
args = parser.parse_args()

def tokenize_function(examples, tokenizer, text_column_name="text"):
    return tokenizer(examples[text_column_name], add_special_tokens=True)

def group_texts(examples, block_size):
    # Concatenate all texts
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    # Drop the small remainder
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    
    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    
    return result

def main():
    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load tokenizer first
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "../Model/Llama-3.2-3B-Instruct", 
        use_fast=False
    )
    
    # Set padding token to be the same as EOS token
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token...")
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.model_max_length = args.block_size
    
    # Load and prepare model
    print("Loading model...")
    model = LlamaForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map=device
    )
    
    # Ensure the model is in training mode
    model.train()
    
    # Configure LoRA
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["gate_proj", "up_proj", "down_proj","q_proj", "v_proj", "k_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # Apply LoRA
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("Loading dataset...")
    dataset1 = load_dataset(args.dataset_name, args.dataset_config, split="train")
    dataset2 = load_dataset("Paillat/simple-wiki-article", split="train")
    dataset3 = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    dataset = datasets.concatenate_datasets([dataset1, dataset2, dataset3])
    dataset = dataset.filter(lambda x: len(x['text']) > 0)

    raw_datasets = dataset
    
    if args.sample is not None:
        print(f"Sampling {args.sample} examples from dataset...")
        raw_datasets = raw_datasets.select(range(args.sample))
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    text_column_name = "text"
    
    # Tokenize the dataset
    tokenize_fn = lambda examples: tokenize_function(examples, tokenizer, text_column_name)
    tokenized_datasets = raw_datasets.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets.column_names,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset"
    )
    
    # Group texts
    group_fn = lambda examples: group_texts(examples, args.block_size)
    lm_datasets = tokenized_datasets.map(
        group_fn,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {args.block_size}"
    )
    

    # lm_datasets = lm_datasets.shuffle(seed=42)
    
    # Setup training arguments
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=args.save,
        overwrite_output_dir=True,
        num_train_epochs=args.epoch,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=1e-5,
        lr_scheduler_type='cosine',
        warmup_ratio=0.05,
        max_grad_norm=1.0,
        
        bf16=True,
        gradient_accumulation_steps=1,
        
        logging_strategy="steps",
        logging_steps=1,
        
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=3,
        
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        gradient_checkpointing=False,  # Disable for now to debug
        dataloader_drop_last=True,
        optim="adamw_torch", # Explicitly set optimizer
        report_to=None
    )
    
    # Setup data collator
    data_collator = transformers.DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8  # Optimize tensor operations
    )
    
    # Initialize trainer
    print("Initializing trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    trainer.train(resume_from_checkpoint=args.resume if args.resume else None)
    
    # Save the model
    print("Saving model...")
    if not os.path.exists(args.export):
        os.makedirs(args.export)
    
    # Merge and save the model
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.export)
    tokenizer.save_pretrained(args.export)
    print(f"Model successfully saved to {args.export}")

    readme_path = os.path.join(args.export, "README.md")
    if os.path.exists(readme_path):
        os.remove(readme_path)
        print("Removed README.md to avoid YAML validation error.")

    hf_path = f"Tony027/{args.model_id}-LoRA"

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
        folder_path=args.export,
        repo_id=hf_path,
        repo_type="model",
    )

if __name__ == '__main__':
    main()