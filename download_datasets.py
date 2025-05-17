import os
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = "qcri-cs/TechniqueRAG-Datasets"
dataset_name = repo_id.split("/")[-1]
destination_path = Path("./datasets") / dataset_name

destination_path.mkdir(parents=True, exist_ok=True)

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=str(destination_path),
    local_dir_use_symlinks=False,
)

print(f"Dataset downloaded to: {destination_path.resolve()}")