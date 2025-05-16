import torch
import torch.nn as nn
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
from torch.nn.attention import SDPBackend, sdpa_kernel


from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter

from quant_cfg import get_quant_config_slm


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
    with torch.no_grad():
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


def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    ### === TODO: Load your model (you may change this part) ===
    recommended_inductor_config_setter()
    backend = 'gemlite'

    model_name = "../Model/Llama-3.2-3B-Instruct-pruned-group"  
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # compile the model (optional)
    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)
    
    # get cache
    max_cache_len = 16 + max_new_tokens
    past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device=model.device, dtype=model.dtype)
    # for i in tqdm(range(5), desc="Warm Up..."):
    #     generated = generate(model, input_ids, past_key_values, max_new_tokens)
    #     past_key_values.reset()
        
    # Quantize
    quant_config = get_quant_config_slm(model)
    
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model, backend=backend) 
    torch.cuda.empty_cache()
    
    # assistant_model = AutoModelForCausalLM.from_pretrained(
    #     "double7/vicuna-68m",
    #     torch_dtype=torch.float16,
    #     device_map=device,
    #     attn_implementation="sdpa"
    # )
    # assistant_tokenizer = AutoTokenizer.from_pretrained("double7/vicuna-68m")
    # assistant_model.generation_config.assistant_confidence_threshold=0.4
    #####################################


    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    # model.prefill_forward = model.forward

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # === (Optional) Set up StaticCache for manual KV cache management ===
    
    
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

    with open("result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["value"])
        writer.writerow([ppl])
        writer.writerow([rounded_tput])

    with open("experiment.csv", mode="a", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow([model_name.split('/')[2], rounded_tput, ppl, os.path.basename(__file__)])
        
if __name__ == '__main__':
    main()