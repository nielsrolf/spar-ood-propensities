import tinker
import urllib.request
import tarfile
import os
from huggingface_hub import HfApi
import sys


tinkerlink = sys.argv[1]
name       = sys.argv[2]

# 1. Get download URL from Tinker
sc = tinker.ServiceClient()
rc = sc.create_rest_client()
future = rc.get_checkpoint_archive_url_from_tinker_path(
    tinkerlink  # or a specific step like sampler_weights/000080
)
resp = future.result()

# 2. Download the archive
urllib.request.urlretrieve(resp.url, f"/Users/jo/{name}_archive.tar")

# 3. Extract (contains LoRA adapter weights + config)
with tarfile.open(f"/Users/jo/{name}_archive.tar") as tar:
    tar.extractall(f"/Users/jo/{name}_adapterweights")

# # 4. Upload to HuggingFace
# api = HfApi()
# api.create_repo(f"jo-chen/{name}", exist_ok=True)
# api.upload_folder(
#     folder_path=f"/Users/jo/{name}_adapterweights",
#     repo_id=f"jo-chen/{name}",
# )