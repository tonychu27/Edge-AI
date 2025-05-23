import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import torch.nn.functional as F
from transformers import LlamaForCausalLM

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.


def generate(model, input_ids, past_key_values, max_new_tokens, i):
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


@torch.no_grad()
def speculative_generate(model, assistant_model, input_ids, past_key_values, assistant_past_key_values, max_new_tokens, max_draft_len):
    prompt_len = input_ids.shape[1]
    # tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    # prefill for the target model
    outputs = model.prefill_forward(
        input_ids,
        past_key_values=past_key_values,
        logits_to_keep=1
    )
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits, dim=-1) # [1, 1]
    
    # prefill for the assistant model
    assistant_outputs = assistant_model.prefill_forward(
        input_ids,
        past_key_values=assistant_past_key_values,
        logits_to_keep=1
    )
    assistant_past_key_values = assistant_outputs.past_key_values
    
    while 1:
        # save the pos and input_ids for target model
        target_pos = input_ids.shape[1]
        if target_pos - prompt_len >= max_new_tokens:
            return input_ids
        
        target_input_ids = input_ids.clone()
        target_next_tokens = next_token.clone()
        # token by token decoding for draft model
        draft_tokens = []
        draft_prob = []
        draft_len = 0
        acc_prob = 1
        for i in range(max_draft_len):
            pos = input_ids.shape[1]
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

            assistant_outputs = assistant_model(
                next_token,
                past_key_values=assistant_past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            
            logits = assistant_outputs.logits.clone()
            prob = F.softmax(assistant_outputs.logits.clone(), dim=-1)
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            assistant_past_key_values = assistant_outputs.past_key_values
            draft_tokens.append(next_token)
            draft_prob.append(prob)
            draft_len = i + 1
            
            # acc_prob = acc_prob * torch.max(prob).item()
            
            # if acc_prob < 0.2:
            #     break
            
        draft_tokens = torch.cat(draft_tokens, dim=-1)
        draft_prob = torch.cat(draft_prob, dim=-2) # [1,draft_tokens,vocab_size]
        
        # compute the accept_len
        accept_len = 0
        cache_position = torch.arange(target_pos, target_pos + draft_len + 1, device=input_ids.device, dtype=torch.long)
        
        output = model.prefill_forward(
            torch.cat([target_next_tokens, draft_tokens], dim=-1),
            past_key_values=past_key_values,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position
        )
        
        past_key_values = output.past_key_values
        # print(output.logits.shape)
        output_tokens = torch.argmax(output.logits, dim=-1)
        # print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
        
        output_prob = F.softmax(output.logits.clone(), dim=-1)
        
        # for i in range(draft_len):
        #     if output_prob[0, i][draft_tokens[0, i]] > draft_prob[0, i][draft_tokens[0, i]] or output_prob[0, i][draft_tokens[0, i]] > 0.0001:
        #         accept_len += 1
        #     else:
        #         break
        accept_len = draft_len
        print(f"accept_len: {accept_len} for draft_len: {draft_len}")
        
        if (accept_len == 0):
            input_ids = torch.cat([target_input_ids, target_next_tokens], dim=-1)
            next_token = output_tokens[:, 0].unsqueeze(0)
        else:
            input_ids = torch.cat([target_input_ids, target_next_tokens, draft_tokens[:, :accept_len]], dim=-1)
            next_token = output_tokens[:, accept_len].unsqueeze(0)
        

def find_candidate_pred_tokens(
    input_ids: torch.Tensor,
    output_cache: torch.Tensor | None,
    max_ngram_size: int = 3,
    num_pred_tokens: int = 10
) -> torch.Tensor:
    """
    從 output_cache 中尋找與 input_ids 尾端 n-gram 相同的序列，
    回傳該匹配之後 num_pred_tokens 個 token ；若找不到則回傳空 tensor。
    --------------------------------------------------------------------
    input_ids      : shape = [1, cur_len]
    output_cache   : shape = [1, cache_len]，儲存歷史輸出
    max_ngram_size : 最長 n-gram（由此往下嘗試）
    num_pred_tokens: 找到匹配後，要取多少 token 做草稿
    """
    device = input_ids.device
    empty = torch.tensor([], dtype=torch.long, device=device)

    # 若 cache 不存在或長度不足
    if output_cache is None or output_cache.size(1) < 1:
        return empty

    cache_len = output_cache.size(1)

    for n in range(max_ngram_size, 0, -1):
        if input_ids.size(1) < n or cache_len < n:
            continue  # n-gram 比對不成立

        # 取出查詢用 n-gram（tensor shape: [1, n]）
        query = input_ids[0, -n:].unsqueeze(0)  # [1, n]

        # 對 output_cache 做 sliding window，shape: [1, cache_len-n+1, n]
        windows = output_cache.unfold(dimension=1, size=n, step=1)

        # 比對每個 window 是否與 query 完全一致
        matches = (windows == query).all(dim=2)            # [1, cache_len-n+1]
        match_idxs = matches.nonzero(as_tuple=True)[1]      # 取出匹配起點索引

        # 依序取第一個合法匹配
        for idx in match_idxs:
            start = idx + n
            end   = start + num_pred_tokens
            # 確保有東西可取且不超界
            if start < cache_len:
                end = min(end, cache_len)
                draft = output_cache[0, start:end]
                if draft.numel() > 0:
                    return draft

    # 全部 n-gram 嘗試失敗 → 回傳空 tensor
    return empty




@torch.no_grad()
def pld_generate(model, input_ids, past_key_values,max_new_tokens, max_draft_len, output_cache, i):
    prompt_len = input_ids.shape[1]

    # prefill for the target model
    outputs = model.prefill_forward(
        input_ids,
        past_key_values=past_key_values,
        logits_to_keep=1
    )
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits, dim=-1) # [1, 1]
    
    while 1:
        # save the pos and input_ids for target model
        target_pos = input_ids.shape[1]
        if target_pos - prompt_len >= max_new_tokens:
            if i == 0:
                print("End of generation")
                print(input_ids)
                output_cache["data"] = input_ids.clone()
            return input_ids
        
        target_input_ids = input_ids.clone()
        target_next_tokens = next_token.clone()

        # prompt lookup
        if (i == 0):
            draft_tokens = input_ids[0, 0:1]
        else:
            # print(f"input_ids: {input_ids}")
            # print(f"output_cache: {output_cache['data']}")
            draft_tokens = find_candidate_pred_tokens(torch.cat([input_ids, target_next_tokens], dim = -1), output_cache["data"], max_ngram_size=8, num_pred_tokens=max_draft_len)
            # print(f"draft_tokens: {draft_tokens}")
            
        if draft_tokens.ndim == 1:
            draft_tokens = draft_tokens.unsqueeze(0)
        
        draft_len = draft_tokens.shape[1]
        
        # verify
        accept_len = 0
        cache_position = torch.arange(target_pos, target_pos + draft_len + 1, device=input_ids.device, dtype=torch.long)
        
        output = model.prefill_forward(
            torch.cat([target_next_tokens, draft_tokens], dim=-1),
            past_key_values=past_key_values,
            position_ids=cache_position.unsqueeze(0),
            cache_position=cache_position
        )
        
        past_key_values = output.past_key_values
        output_logits = output.logits.clone()
        
        output_tokens = torch.argmax(output.logits, dim=-1)
        # print(tokenizer.decode(output_tokens[0], skip_special_tokens=True))
        
        
        for j in range(draft_len):
            # print(f"draft_tokens: {draft_tokens[0, j]} output_tokens: {output_tokens[0,j]}")
            # or topk is also fine
            pred_logits = output_logits[0, j]
            if draft_tokens[0, j] == output_tokens[0,j] or draft_tokens[0, j] in torch.topk(pred_logits, 100).indices:
                accept_len += 1
            else:
                break
        
        # print(f"accept_len: {accept_len} for draft_len: {draft_len}")
        
        if (accept_len == 0):
            input_ids = torch.cat([target_input_ids, target_next_tokens], dim=-1)
            next_token = output_tokens[:, 0].unsqueeze(0)
        elif (accept_len == draft_len):
            input_ids = torch.cat([target_input_ids, target_next_tokens, draft_tokens], dim=-1)
            next_token = output_tokens[:, -1].unsqueeze(0)
        else:
            input_ids = torch.cat([target_input_ids, target_next_tokens, draft_tokens[:, :accept_len]], dim=-1)
            next_token = output_tokens[:, accept_len].unsqueeze(0)
               

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

output_cache = None

def main():
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    
    ### === TODO: Load your model (you may change this part) ===
    recommended_inductor_config_setter()
    backend = 'bitblas'
    
    model_name = "../Model/Llama-3.2-3B-Instruct" 

    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.float16,
        attn_implementation="sdpa"
    )
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # get cache
    max_cache_len = 16 + max_new_tokens
    past_key_values = StaticCache(config=model.config, max_batch_size=1, max_cache_len=max_cache_len, device=model.device, dtype=model.dtype)
    output_cache = {}
    
    # Quantize
    # quant_config = get_quant_config_slm(model)
    
    # AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    # from hqq.utils.patching import prepare_for_inference
    # prepare_for_inference(model, backend=backend) 

    # compile the model (optional)
    model.prefill_forward = model.forward
    # model.forward = torch.compile(model.forward, mode='max-autotune', dynamic=True, fullgraph=True)
    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    # model.prefill_forward = model.forward

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    max_draft_len = 16
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
        # generated = generate(model, input_ids, past_key_values, max_new_tokens, i)
        generated = pld_generate(model, input_ids, past_key_values, max_new_tokens,max_draft_len, output_cache, i)
        past_key_values.reset()
        # generated = speculative_generate(model, assistant_model,input_ids, past_key_values, assistant_past_key_values, max_new_tokens, max_draft_len=8)
        # assistant_past_key_values.reset()
        
        
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    
    for i in tqdm(range(10), desc="Test Inference"):
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
        # generated = generate(model, input_ids, past_key_values, max_new_tokens, i)
        # generated = speculative_generate(model, assistant_model,input_ids, past_key_values, assistant_past_key_values, max_new_tokens, max_draft_len=8)
        generated = pld_generate(model, input_ids, past_key_values, max_new_tokens,max_draft_len, output_cache, i)
        past_key_values.reset()
        # assistant_past_key_values.reset()

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
        
if __name__ == '__main__':
    main()