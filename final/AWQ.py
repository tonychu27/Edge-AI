from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, AwqConfig
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError
import argparse
import os
import torch

argparser = argparse.ArgumentParser()
argparser.add_argument("--model_id", type=str, required=True)
args = argparser.parse_args()

model_path = f"../Model/{args.model_id}"
quant_path = f"../Model/{args.model_id}-awq"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model path '{model_path}' does not exist.")

quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"
}

model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

model.quantize(tokenizer, quant_config=quant_config)

model.model.config.quantization_config = AwqConfig(
    bits=quant_config["w_bit"],
    group_size=quant_config["q_group_size"],
    zero_point=quant_config["zero_point"],
    version=quant_config["version"].lower(),
)

model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)