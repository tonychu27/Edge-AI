import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import shutil
import logging

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

class FFNPruner:
    def __init__(self, model_name_or_path="meta-llama/Llama-3.2-3B", image_path="ffn", device=None):
        self.model_name_or_path = model_name_or_path
        self.image_path = image_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16).to(self.device)
        self.model.eval()
        self.activations = {}
        self.hooks = []
        self.logger = self._setup_logger(log_dir=image_path)

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

        os.makedirs(self.image_path, exist_ok=True)

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

    def prune(self, save_path: str, prune_ratio: float):
        model_state_dict = self.model.state_dict()
        config = self.model.config
        os.makedirs(save_path, exist_ok=True)
        self.logger = self._setup_logger(save_path)

        old_intermediate_size = config.intermediate_size
        new_intermediate_size = int(old_intermediate_size * (1 - prune_ratio))
        new_intermediate_size=max(64, round(new_intermediate_size/64)*64)
        config.intermediate_size = new_intermediate_size
        self.logger.info(f"Pruning ratio: {prune_ratio} => intermediate_size {old_intermediate_size} → {new_intermediate_size}")

        new_model = AutoModelForCausalLM.from_config(config)
        new_model_state_dict = new_model.state_dict()

        for name, param in model_state_dict.items():
            if "mlp.gate_proj.weight" in name:
                new_model_state_dict[name] = param[:new_intermediate_size, :].clone()
            elif "mlp.up_proj.weight" in name:
                new_model_state_dict[name] = param[:new_intermediate_size, :].clone()
            elif "mlp.down_proj.weight" in name:
                new_model_state_dict[name] = param[:, :new_intermediate_size].clone()
            elif name in new_model_state_dict:
                new_model_state_dict[name] = param.clone()
            else:
                self.logger.warning(f"⚠️ {name} not in new model state dict")

        new_model.load_state_dict(new_model_state_dict)
        new_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        self.model = new_model.to(self.device)
        self.logger.info(f"✅ Pruned model saved to: {save_path}")

    def test_perplexity(self, num_samples=100):
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"test[:{num_samples}]")
        encodings = self.tokenizer("\n\n".join(test["text"]), return_tensors="pt")
        self.model.eval()

        max_length = 2048
        stride = 512
        seq_len = encodings.input_ids.size(1)
        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, seq_len, stride), desc="Calculating perplexity"):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()
            batch_size = target_ids.size(0)
            num_loss_tokens = num_valid_tokens - batch_size
            nll_sum += neg_log_likelihood * num_loss_tokens
            n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = nll_sum / n_tokens
        ppl = torch.exp(avg_nll)
        self.logger.info(f"Perplexity: {ppl.item():.2f}")
        return ppl.item()

    def calculate_parameters(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.logger.info(f"Total parameters: {total_params:,}")
        return total_params


if __name__ == "__main__":
    pruner = FFNPruner(
        model_name_or_path='../Model/Llama-3.2-3B-Instruct',
        image_path="ffn_outputs"
    )

    save_path = '../Model/Llama-3.2-3B-Instruct-pruned-0.95'

    total_params = pruner.calculate_parameters()
    print(f"Total parameters before pruning: {total_params:,}")
    
    # 執行 calibration + reorder
    pruner.calibrate_and_reorder(calibrate_samples=10)
    
    # 測試 perplexity
    perplexity = pruner.test_perplexity(num_samples=10)
    print("Perplexity before pruning:", perplexity)
    
    pruner.prune(save_path=save_path, prune_ratio=0.05)
    perplexity = pruner.test_perplexity(num_samples=10)
    
    print("Perplexity after pruning:", perplexity) 
    
    total_params = pruner.calculate_parameters()
    print(f"Total parameters after pruning: {total_params:,}")

    hf_path = "Tony027/Llama-3.2-3B-Instruct-pruned"

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