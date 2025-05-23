from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

hf_path = f"Tony027/Llama-3.2-3B-pruned-0.55-LoRA"

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
    folder_path="../Model/Llama-3.2-3B-pruned-0.55-LoRA",
    repo_id=hf_path,
    repo_type="model",
)