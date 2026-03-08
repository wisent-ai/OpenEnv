"""Deploy KantBench Space to HuggingFace.

Uploads the Space files plus the required source directories (common/,
env/, constant_definitions/) so the full 90-game environment is available.

Usage:
    python spaces/kant/deploy.py
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "openenv-community/KantBench"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # OpenEnv root
SPACE_DIR = Path(__file__).resolve().parent  # spaces/kant/

# Directories from the main repo needed by the environment
REQUIRED_DIRS = ["common", "env", "constant_definitions"]


def main():
    api = HfApi()

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)

        # Copy Space files (Dockerfile, server/, models.py, etc.)
        for item in SPACE_DIR.iterdir():
            if item.name in ("deploy.py", "__pycache__", ".git"):
                continue
            dest = staging / item.name
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Copy required source directories from repo root
        for dirname in REQUIRED_DIRS:
            src = REPO_ROOT / dirname
            if src.exists():
                shutil.copytree(src, staging / dirname)
                print(f"  Included {dirname}/")
            else:
                print(f"  WARNING: {dirname}/ not found at {src}")

        # Upload
        print(f"\nUploading to {REPO_ID}...")
        api.upload_folder(
            folder_path=str(staging),
            repo_id=REPO_ID,
            repo_type="space",
        )
        print(f"Done! Space: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
