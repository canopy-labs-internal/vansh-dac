from pathlib import Path
import os
from huggingface_hub import HfApi, create_repo

FOLDER_TO_PUSH = Path("/home/vansh/dac-recreate/runs/vansh/10000k").resolve()
HF_USERNAME = "vanshjjw"  # change if needed
REPO_NAME = f"dac-checkpoints-10k"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

api = HfApi()  
create_repo(REPO_ID, private=True, exist_ok=True)
print(f"Created/using repo: {REPO_ID}")

api.upload_large_folder(repo_id=REPO_ID, folder_path=str(FOLDER_TO_PUSH), repo_type="model")
print("Upload complete.")