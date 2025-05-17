import os
os.environ["PYTORCH_INDuctor_LOG_LEVEL"] = "ERROR"
os.environ["PYTORCH_DISABLE_TUNE_LOGS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm.auto import tqdm
from datasets import load_dataset

import random
import numpy as np

import model
from model import LlamaForCausalLM

import argparse

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.


def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    with torch.no_grad() and sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        # Prefill
        outputs = model.prefill_forward(
            input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1
        )
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # Token-by-token Decoding
        for _ in range(max_new_tokens):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

@torch.no_grad()
def main(args):
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    ### === TODO: Load your model (you may change this part) ===

    model_name = f"../Model/{args.model_id}"  
    print(f"Loading {args.model_id} ...")

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )
    
    # compile the model (optional)
    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)
    
    # Quantize (HQQ?, GPTQ?)
    

    # assistant_model = AutoModelForCausalLM.from_pretrained(
    #     "double7/vicuna-68m",
    #     torch_dtype=torch.float16,
    #     device_map=device,
    #     attn_implementation="sdpa"
    # )
    # assistant_tokenizer = AutoTokenizer.from_pretrained("double7/vicuna-68m")
    # assistant_model.generation_config.assistant_confidence_threshold=0.4
    #####################################

    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    # model.prefill_forward = model.forward

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # === (Optional) Set up StaticCache for manual KV cache management ===
    from transformers import StaticCache, SlidingWindowCache, SinkCache
    
    max_cache_len = 300
    past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device=model.device, dtype=model.dtype)
    
    # precompute the cache
    # with torch.no_grad():
    #     max_cache_len = 300
    #     prompt_cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device=model.device, dtype=model.dtype)
    #     prompt = "How to learn a new language?"
    #     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    #     prompt_cache = model(**inputs, past_key_values = prompt_cache).past_key_values
    ####################################################################
    
    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up ===
        # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]): 
        # with torch.backends.cuda.sdp_kernel(enable_math=True):
            # _ = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_new_tokens=max_new_tokens,
            #     pad_token_id=tokenizer.eos_token_id,
            #     do_sample=False,
            #     past_key_values=past_key_values,
            #     # assistant_model=assistant_model,
            #     # assistant_tokenizer=assistant_tokenizer,
            #     # tokenizer=tokenizer,
            # )
        
        # === (Optional) Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()
        
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing === 
        # with sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        # with torch.backends.cuda.sdp_kernel(enable_math=True):
            # generated = model.generate(
            #     input_ids=input_ids,
            #     attention_mask=attention_mask,
            #     max_new_tokens=max_new_tokens,
            #     pad_token_id=tokenizer.eos_token_id,
            #     do_sample=False,
            #     past_key_values=past_key_values,
            #     # assistant_model=assistant_model,
            #     # assistant_tokenizer=assistant_tokenizer,
            #     # tokenizer=tokenizer,
            # )
        
        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = max_new_tokens / (elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
        
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    
    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("lora_result.csv", mode="a", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow([model_name.split('/')[2], rounded_tput, ppl])
        
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_id", type=str, required=True)
    args = argparser.parse_args()

    main(args)