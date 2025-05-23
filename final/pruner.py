import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import os
import logging
import argparse
import torch.nn as nn

class FFNPruner:
    def __init__(self, model_name_or_path="meta-llama/Llama-3.2-3B", image_path=None, output_path=None,device=None, use_lora=False):
        assert model_name_or_path is not None, "model_name_or_path cannot be None"
        assert image_path is not None, "image_path cannot be None"
        assert output_path is not None, "output_path cannot be None"
        
        self.model_name_or_path = model_name_or_path
        self.image_path = image_path
        self.output_path = output_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("../Model/Llama-3.2-3B-Instruct", use_fast=False)
        print("loading model...")
        # merge the weight if needed
        if use_lora:
            import peft
            from peft import PeftModel, PeftConfig
            peft_model_id = 'Tony027/Llama-3B-pruned-LoRA'
            config = PeftConfig.from_pretrained(peft_model_id)
            base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to(self.device)
            self.model = PeftModel.from_pretrained(base_model, peft_model_id)
            self.model = self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to(self.device)
        ###########################################
        
        self.model.eval()
        self.activations = {}
        self.hooks = []
        self.logger = self._setup_logger(log_dir=output_path)

    def _setup_logger(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger("FFNPruner")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def _hook_input_fn(self, module, input, output, layer_name):
        self.activations[layer_name] = input

    def _add_hooks(self):
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Module) and "mlp.down_proj" in name:
                hook = module.register_forward_hook(lambda mod, inp, out, name=name: self._hook_input_fn(mod, inp, out, name))
                self.hooks.append(hook)
                self.logger.info(f"Hook added to {name}")

    def _remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def calibrate_and_reorder(self, calibrate_samples=10):
        dataset = load_dataset("Salesforce/wikitext", "wikitext-2-v1", split=f"test[:{calibrate_samples}]")
        self._add_hooks()
        new_model_state_dict = self.model.state_dict()

        with torch.no_grad():
            for sample in tqdm(dataset['text'], desc="Calibrating"):
                inputs = self.tokenizer(sample, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = self.model(**inputs)

        self._remove_hooks()


        for layer, activation in self.activations.items():
            self.logger.info(f"Processing layer: {layer}")
            activation = activation[0].float().cpu()
            activation = torch.abs(activation)
            activation_sum = activation.mean(dim=(0, 1))
            _, permuted_index = torch.sort(activation_sum, descending=True)

            layer_num = layer.split(".")[2]
            gate_proj_weight = f"model.layers.{layer_num}.mlp.gate_proj.weight"
            up_proj_weight = f"model.layers.{layer_num}.mlp.up_proj.weight"
            down_proj_weight = f"model.layers.{layer_num}.mlp.down_proj.weight"

            if gate_proj_weight in self.model.state_dict():
                new_model_state_dict[gate_proj_weight] = self.model.state_dict()[gate_proj_weight][permuted_index, :].clone()
            if up_proj_weight in self.model.state_dict():
                new_model_state_dict[up_proj_weight] = self.model.state_dict()[up_proj_weight][permuted_index, :].clone()
            if down_proj_weight in self.model.state_dict():
                new_model_state_dict[down_proj_weight] = self.model.state_dict()[down_proj_weight][:, permuted_index].clone()

            std_dev = activation_sum.std()
            mean_val = activation_sum.mean()

            plt.figure(figsize=(8, 6))
            plt.hist(activation_sum, bins=100, alpha=0.7)
            plt.xlabel("Activation Value")
            plt.ylabel("Frequency")
            plt.title(f"Histogram of Activations for {layer}")
            plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1)
            plt.axvline(mean_val + std_dev, color='g', linestyle='dashed', linewidth=1)
            plt.axvline(mean_val - std_dev, color='g', linestyle='dashed', linewidth=1)
            plt.legend({'Mean': mean_val, 'Std Dev': std_dev})
            plt.savefig(f"{self.image_path}/{layer}_histogram.png")
            plt.cla()

            size = int(np.ceil(np.sqrt(activation_sum.shape[0])))
            activation_padded = np.pad(activation_sum.numpy(), (0, size**2 - activation_sum.shape[0]), mode='constant').reshape(size, size)
            plt.figure(figsize=(8, 6))
            plt.imshow(activation_padded, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.title(f"Layer {layer}\nMean: {mean_val:.4f}, Std: {std_dev:.4f}")
            plt.savefig(f"{self.image_path}/{layer}_heatmap.png")
            plt.cla()

        self.model.load_state_dict(new_model_state_dict)
        torch.cuda.empty_cache()

    def prune(self, prune_ratio: float):
        model_state_dict = self.model.state_dict()
        config = self.model.config


        old_intermediate_size = config.intermediate_size
        new_intermediate_size = int(old_intermediate_size * (1 - prune_ratio))
        new_intermediate_size = (new_intermediate_size // 128) * 128
        self.model.config.intermediate_size = new_intermediate_size
        
        self.logger.info(f"Pruning ratio: {prune_ratio} => intermediate_size {old_intermediate_size} → {self.model.config.intermediate_size }")
        new_state_dict = {}
        
        for name, param in model_state_dict.items():
            if "mlp.gate_proj.weight" in name:
                new_state_dict[name] = param[:new_intermediate_size, :].clone()
            elif "mlp.up_proj.weight" in name:
                new_state_dict[name] = param[:new_intermediate_size, :].clone()
            elif "mlp.down_proj.weight" in name:
                new_state_dict[name] = param[:, :new_intermediate_size].clone()
            else:
                new_state_dict[name] = param.clone()
            

        # 載入 state dict
        # self.model.load_state_dict(new_model_state_dict, strict=False)
        torch.save(new_state_dict, os.path.join(args.output_path, "pytorch_model.bin"))
        self.model.config.save_pretrained(args.output_path)
        del self.model, new_state_dict
        self.model = AutoModelForCausalLM.from_pretrained(args.output_path, torch_dtype=torch.bfloat16, device_map="cuda")
        self.model.save_pretrained(args.output_path)
        # remove pytorch_model.bin
        os.remove(os.path.join(args.output_path, "pytorch_model.bin"))
        self.logger.info(f"✅ Pruned model saved to: {args.output_path}")
        self.tokenizer.save_pretrained(args.output_path)

    def test_perplexity(self, num_samples=100):
        if num_samples == 0:
            test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        else:
            test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[:{num_samples}]")
            
        test_enc = self.tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
        self.model.seqlen = 2048
        test_enc = test_enc.input_ids.to(self.device)
        
        nsamples = test_enc.numel() // self.model.seqlen
        nlls = []  
        for i in tqdm(range(nsamples), desc="Evaluating..."):
            batch = test_enc[:, (i * self.model.seqlen):((i + 1) * self.model.seqlen)]
            
            with torch.no_grad():
                lm_logits = self.model(batch).logits

            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = test_enc[:, (i * self.model.seqlen):((i + 1) * self.model.seqlen)][:, 1:]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * self.model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * self.model.seqlen))
        
        return ppl.item()

    def calculate_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        return total_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', default='meta-llama/Llama-3.2-3B', help='model name or path')
    parser.add_argument('--image_path', default='ffn_outputs', help='path to save images')
    parser.add_argument('--output_path', default='./Llama-3.2-3B-pruned-0.55', help='path to save pruned model')
    args = parser.parse_args()
    pruner = FFNPruner(
        model_name_or_path = args.model_name_or_path,
        image_path = os.path.join(args.output_path, args.image_path),
        output_path = args.output_path,
        device = 'cuda',
        use_lora = False
    )
    print("Making directory:", pruner.output_path)
    # make directory
    os.makedirs(pruner.output_path, exist_ok=True)
    os.makedirs(pruner.image_path, exist_ok=True)
    
    # total_params = pruner.calculate_parameters()
    # print(f"Total parameters before pruning: {total_params:,}")
    # calibrate and reorder the model
    pruner.calibrate_and_reorder(calibrate_samples=100)
    perplexity = pruner.test_perplexity(num_samples=0)
    print("Perplexity before pruning:", perplexity)
    
    pruner.prune(prune_ratio=0.05)
    perplexity = pruner.test_perplexity(num_samples=0)
    print("Perplexity after pruning:", perplexity)
     
    # total_params = pruner.calculate_parameters()
    # print(f"Total parameters after pruning: {total_params:,}")