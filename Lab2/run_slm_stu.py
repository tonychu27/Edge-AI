import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np

from hqq_utils import AutoHQQHFModel, get_size_of_model
from hqq.utils.patching import recommended_inductor_config_setter

from quant_cfg import get_quant_config_slm

def generate(model, input_ids, past_key_values, max_new_tokens, activate_timing, verbose=True):
    input_ids = input_ids.clone()
    tput = None
    # Run an initial forward pass to compute and store the static KV cache
    if verbose:
        print('Prefilling...')
    with torch.no_grad():
        # outputs = custom_forward(model, input_ids, past_key_values=past_key_values, use_cache=True, position_ids=None, attention_mask=None, cache_position=None, is_compiled=False)
        outputs = model.prefill_forward(input_ids, past_key_values=past_key_values, position_ids=None, attention_mask=None, cache_position=None, logits_to_keep=1)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits, dim=-1)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

    # Generate tokens one by one using a for loop and update the KV cache
    if verbose:
        print('Decoding...')
    with torch.no_grad():
        if activate_timing:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        for _ in range(max_new_tokens):
            # Compute position_ids using the current sequence length
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos+1, device=input_ids.device, dtype=torch.long)

            # Run the model on the last token using the cached key-value pairs
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits

            # Greedily select the token with the highest probability
            next_token = torch.argmax(logits, dim=-1)

            # Append the predicted token to the generated sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            # Update the KV cache for the next iteration
            past_key_values = outputs.past_key_values
        if activate_timing:
            end_event.record()
        torch.cuda.synchronize()
    if activate_timing:
        tput = max_new_tokens / start_event.elapsed_time(end_event) * 1000
        # print(f"Throughput: {tput} toks/sec")
    return input_ids, tput

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # print(f"Dataset length: {len(test_dataset)}")
    
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
    recommended_inductor_config_setter()
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    backend = 'gemlite'
    
    model_name = "meta-llama/Llama-3.2-1B-Instruct"   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Separate Prefill & Decode Forwarding Function
    model.prefill_forward = model.forward
    model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=False, fullgraph=True)
    #####################################
    
    # Original Model
    warmup_prompt = "Explain what AI is."
    input_ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to(device)
    past_key_values = StaticCache(
        config=model.config, 
        max_batch_size=1, 
        max_cache_len=max_new_tokens + 16, 
        device=model.device, 
        dtype=torch.float16
    )
    
    for i in tqdm(range(5), desc="Warm Up..."):
        generated = generate(model, input_ids, past_key_values, max_new_tokens, activate_timing=False, verbose=False)
        past_key_values.reset()
        
    prompt = "How to learn a new language?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    tputs = []
    for _ in tqdm(range(10), desc="Test Inference"):
        generated, tput = generate(model, input_ids, past_key_values, max_new_tokens, activate_timing=True, verbose=False)
        past_key_values.reset()
        tputs.append(tput)
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\nThroughput: {org_tput} toks/s')
    print(f'Model Size After Quant: {get_size_of_model(model) / (1024 ** 2)} MiB')
    
    # TODO: Quantize    
    quant_config = get_quant_config_slm(model)
    
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    from hqq.utils.patching import prepare_for_inference
    prepare_for_inference(model, backend=backend) 
    torch.cuda.empty_cache()

    warmup_prompt = "Explain what AI is."
    input_ids = tokenizer(warmup_prompt, return_tensors="pt").input_ids.to(device)
    for i in tqdm(range(5), desc="Warm Up..."):
        generated = generate(model, input_ids, past_key_values, max_new_tokens, activate_timing=False, verbose=False)
        past_key_values.reset()

    prompt = "How to learn a new language?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    tputs = []
    for _ in tqdm(range(10), desc="Test Inference"):
        generated, tput = generate(model, input_ids, past_key_values, max_new_tokens, activate_timing=True, verbose=False)
        past_key_values.reset()
        tputs.append(tput)
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    tputs = np.sort(tputs)[2:-2]
    quant_tput = np.mean(tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\nThroughput: {quant_tput} toks/s')
    print(f'Model Size After Quant: {get_size_of_model(model) / (1024 ** 2)} MiB')
    
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    print(f"Speedup: {quant_tput / org_tput} x")

    score = 0
    score += 5 if ppl <= 14 else 0
    score += 5 if quant_tput / org_tput >= 1.3 else 0
    print(f'Score: {score}')

if __name__ == '__main__':
    main()