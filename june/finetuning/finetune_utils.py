import json
import os
import shutil
import time
from pathlib import Path
from datasets import load_dataset

from unsloth import FastLanguageModel


def load_model_and_tokenizer(model_id, load_in_4bit=False, max_retries=5, use_modelscope=None):
    """
    Load model and tokenizer with retry logic and modelscope fallback.

    Args:
        model_id: HuggingFace model ID
        load_in_4bit: Whether to load in 4-bit quantization
        max_retries: Maximum number of retry attempts (default: 5)
        use_modelscope: Override to force modelscope usage. If None, will fallback to modelscope on HF failures.

    Returns:
        Tuple of (model, tokenizer)
    """
    # Check if user wants to force modelscope via environment variable
    if use_modelscope is None:
        use_modelscope = os.environ.get('UNSLOTH_USE_MODELSCOPE', '0') == '1'

    if use_modelscope:
        print(f"Using ModelScope to download model: {model_id}")
        try:
            import modelscope  # Check if modelscope is installed
        except ImportError:
            print("ModelScope not installed. Installing now...")
            import subprocess
            subprocess.check_call(['pip', 'install', 'modelscope'])
            import modelscope

        os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'

    last_exception = None
    for attempt in range(max_retries):
        try:
            print(f"Loading model {model_id} (attempt {attempt + 1}/{max_retries})...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_id,
                dtype=None,
                device_map="auto",
                load_in_4bit=load_in_4bit,
                token=os.environ.get("HF_TOKEN", ""),
                max_seq_length=2048,
            )
            print(f"✓ Successfully loaded model: {model_id}")
            return model, tokenizer

        except Exception as e:
            last_exception = e
            error_msg = str(e).lower()

            # Check if it's a network/HuggingFace connectivity issue
            is_network_error = any(keyword in error_msg for keyword in [
                'huggingface seems to be down',
                'connection',
                'timeout',
                'network',
                'failed to download',
                'http error',
                'ssl error'
            ])

            if is_network_error:
                if attempt < max_retries - 1:
                    # Exponential backoff: 10s, 20s, 40s, 80s
                    wait_time = 10 * (2 ** attempt)
                    print(f"⚠️  Network error on attempt {attempt + 1}/{max_retries}: {e}")
                    print(f"   Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                elif not use_modelscope:
                    # Last attempt failed and we haven't tried modelscope yet
                    print(f"\n⚠️  All HuggingFace download attempts failed.")
                    print(f"   Trying ModelScope as fallback...")
                    return load_model_and_tokenizer(
                        model_id,
                        load_in_4bit=load_in_4bit,
                        max_retries=1,  # Only try modelscope once
                        use_modelscope=True
                    )
            else:
                # Not a network error, don't retry
                print(f"❌ Non-network error occurred: {e}")
                raise

    # All retries exhausted
    error_msg = (
        f"\n{'='*80}\n"
        f"❌ FAILED TO LOAD MODEL: {model_id}\n"
        f"{'='*80}\n"
        f"All {max_retries} attempts failed with error:\n{last_exception}\n\n"
        f"Possible solutions:\n"
        f"1. Check HuggingFace status: https://status.huggingface.co/\n"
        f"2. Verify your internet connection\n"
        f"3. Check if HF_TOKEN is set correctly\n"
        f"4. Try using ModelScope:\n"
        f"   pip install modelscope\n"
        f"   export UNSLOTH_USE_MODELSCOPE=1\n"
        f"{'='*80}\n"
    )
    print(error_msg)
    raise RuntimeError(error_msg) from last_exception


def is_peft_model(model):
    is_peft = isinstance(model.active_adapters, list) and len(model.active_adapters) > 0
    try:
        is_peft = is_peft or len(model.active_adapters()) > 0
    except:
        pass
    return is_peft


def load_jsonl(file_id):
    """Load data from local JSONL file or HuggingFace dataset.

    Args:
        file_id: Either a local file path (e.g., "./data/train.jsonl") or
                HuggingFace dataset ID (e.g., "username/dataset" or "username/dataset:train")

    Returns:
        List of dictionaries containing the dataset examples
    """
    return load_dataset_flexible(file_id)


def load_dataset_flexible(dataset_id, split=None):
    """
    Load dataset from either local JSONL file or HuggingFace Hub.

    Automatically detects the source type and loads accordingly.

    Args:
        dataset_id: Can be one of:
            - Local file path: "./data/train.jsonl" or "/path/to/data.jsonl"
            - HuggingFace dataset: "username/dataset-name"
            - HuggingFace dataset with config: "username/dataset-name/config-name"
            - HuggingFace dataset with split: "username/dataset-name:train"
            - HuggingFace dataset with config and split: "username/dataset-name/config-name:train"
        split: Optional split name. If provided, overrides split in dataset_id.
               If not provided and no split in dataset_id, defaults to 'train'.

    Returns:
        List of dictionaries (converted from HuggingFace Dataset if needed)

    Examples:
        >>> load_dataset_flexible("./data/train.jsonl")  # Local file
        >>> load_dataset_flexible("username/my-dataset")  # HF dataset, 'train' split
        >>> load_dataset_flexible("username/my-dataset/my-config")  # HF dataset with config
        >>> load_dataset_flexible("username/my-dataset:validation")  # HF dataset, specific split
        >>> load_dataset_flexible("username/my-dataset/my-config:test")  # HF dataset with config and split
        >>> load_dataset_flexible("username/my-dataset", split="test")  # HF dataset, explicit split
    """
    # Check if it's a local file path
    if os.path.exists(dataset_id):
        print(f"Loading local JSONL file: {dataset_id}")
        with open(dataset_id, "r") as f:
            return [json.loads(line) for line in f.readlines() if line.strip()]

    # Parse HuggingFace dataset ID with optional config and split
    # Format: "username/dataset-name/config-name:split"
    config_name = None
    dataset_name = dataset_id

    # First, extract split if present (after colon)
    if ":" in dataset_id and not os.path.exists(dataset_id):
        dataset_name, split_from_id = dataset_id.rsplit(":", 1)
        if split is None:
            split = split_from_id

    # Then, extract config if present (third path component)
    # Count slashes to determine if config is specified
    parts = dataset_name.split("/")
    if len(parts) == 3:
        # Format: "username/dataset/config"
        config_name = parts[2]
        dataset_name = f"{parts[0]}/{parts[1]}"
    elif len(parts) == 2:
        # Format: "username/dataset"
        dataset_name = dataset_name
    else:
        raise ValueError(
            f"Invalid HuggingFace dataset ID format: '{dataset_id}'. "
            f"Expected 'username/dataset' or 'username/dataset/config'"
        )

    # Default split to 'train' if not specified
    if split is None:
        split = 'train'

    # Load from HuggingFace Hub
    config_str = f" (config: {config_name})" if config_name else ""
    print(f"Loading HuggingFace dataset: {dataset_name}{config_str} (split: {split})")
    try:
        hf_token = os.environ.get("HF_TOKEN")
        dataset = load_dataset(
            dataset_name,
            name=config_name,  # This is the key parameter for dataset configurations/subsets
            split=split,
            token=hf_token
        )

        # Convert to list of dicts for compatibility with existing code
        data = [example for example in dataset]
        print(f"✓ Loaded {len(data)} examples from HuggingFace dataset")
        return data

    except Exception as e:
        # Provide helpful error message
        raise ValueError(
            f"Failed to load dataset '{dataset_id}'. "
            f"Error: {e}\n"
            f"Make sure it's either:\n"
            f"  1. A valid local file path, or\n"
            f"  2. A valid HuggingFace dataset ID:\n"
            f"     - 'username/dataset' (default config, train split)\n"
            f"     - 'username/dataset/config' (specific config, train split)\n"
            f"     - 'username/dataset:split' (default config, specific split)\n"
            f"     - 'username/dataset/config:split' (specific config and split)\n"
            f"For private HuggingFace datasets, ensure HF_TOKEN environment variable is set."
        )


def cleanup_training_artifacts(output_dir, keep_final_checkpoint=False):
    """
    Clean up training artifacts (checkpoints, optimizer states, etc.) from output directory.

    This function removes temporary training files after the model has been successfully
    uploaded to HuggingFace and W&B, helping to free up disk space especially when
    training on systems with limited storage like Google Colab with mounted Drive.

    Args:
        output_dir: Path to the training output directory (e.g., "./tmp")
        keep_final_checkpoint: If True, keeps the final checkpoint directory. Default False.

    Returns:
        dict: Summary of cleanup operation with keys:
            - 'removed_dirs': List of removed checkpoint directories
            - 'removed_files': List of removed files
            - 'space_freed_mb': Approximate space freed in MB
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        print(f"Output directory does not exist: {output_dir}")
        return {'removed_dirs': [], 'removed_files': [], 'space_freed_mb': 0}

    removed_dirs = []
    removed_files = []
    total_size = 0

    # Remove checkpoint directories (checkpoint-*, usually contain model weights and optimizer states)
    for item in output_path.glob("checkpoint-*"):
        if item.is_dir():
            # Calculate size before removal
            dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
            total_size += dir_size

            shutil.rmtree(item)
            removed_dirs.append(str(item))
            print(f"Removed checkpoint directory: {item}")

    # Remove common training artifacts
    training_artifacts = [
        "trainer_state.json",
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "pytorch_model.bin",
        "model.safetensors",
    ]

    for artifact in training_artifacts:
        artifact_path = output_path / artifact
        if artifact_path.exists() and artifact_path.is_file():
            file_size = artifact_path.stat().st_size
            total_size += file_size
            artifact_path.unlink()
            removed_files.append(str(artifact_path))
            print(f"Removed training artifact: {artifact_path}")

    space_freed_mb = total_size / (1024 * 1024)

    summary = {
        'removed_dirs': removed_dirs,
        'removed_files': removed_files,
        'space_freed_mb': round(space_freed_mb, 2)
    }

    print(f"\n✓ Cleanup complete: Freed {summary['space_freed_mb']:.2f} MB")
    print(f"  - Removed {len(removed_dirs)} checkpoint directories")
    print(f"  - Removed {len(removed_files)} training artifact files")

    return summary
