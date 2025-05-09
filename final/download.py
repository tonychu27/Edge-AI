import argparse
import os
import warnings
warnings.simplefilter("ignore", category=FutureWarning)
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, required=True)
args = parser.parse_args()

model_name = args.model_id.split("/")[1]

snapshot_download(
    repo_id=args.model_id,
    local_dir=f"../Model/{model_name}",
    revision="main",
    use_auth_token=os.getenv("HF_TOKEN")
)
    
print(f"\nModel and tokenizer for {model_name} loaded successfully.")