"""Deploy KantBench Gradio Dashboard to HuggingFace Spaces.

Uploads the Gradio app plus the required source directories
(common/, env/, constant_definitions/, bench/gradio_app/) so the
full game environment is available.

Usage:
    python spaces/gradio_dashboard/deploy.py
"""

import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "openenv-community/KantBench-Dashboard"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent  # OpenEnv root
SPACE_DIR = Path(__file__).resolve().parent  # spaces/gradio_dashboard/

# Directories from the main repo needed by the app
REQUIRED_DIRS = ["common", "env", "constant_definitions", "bench", "train"]

IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".git")


def main():
    api = HfApi()

    # Create the Space repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
        )
        print(f"Space repo {REPO_ID} ready.")
    except Exception as exc:
        print(f"Note: {exc}")

    with tempfile.TemporaryDirectory() as tmp:
        staging = Path(tmp)

        # Copy Space files (Dockerfile, README.md)
        for item in SPACE_DIR.iterdir():
            if item.name in ("deploy.py", "__pycache__", ".git"):
                continue
            dest = staging / item.name
            if item.is_dir():
                shutil.copytree(item, dest, ignore=IGNORE)
            else:
                shutil.copy2(item, dest)

        # Copy required source directories from repo root
        for dirname in REQUIRED_DIRS:
            src = REPO_ROOT / dirname
            if src.exists():
                shutil.copytree(src, staging / dirname, ignore=IGNORE)
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
