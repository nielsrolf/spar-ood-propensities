"""Dataset loading and preparation utilities."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from datasets import load_dataset, Dataset

from .config import DatasetConfig

logger = logging.getLogger(__name__)


def load_wikipedia_dataset(config: DatasetConfig) -> Dataset:
    """Load and sample Wikipedia dataset.
    
    Args:
        config: Dataset configuration
        
    Returns:
        Sampled dataset
        
    Raises:
        ValueError: If dataset loading fails
    """
    logger.info(f"Loading dataset {config.dataset_name} ({config.dataset_version})")
    
    try:
        # Load the full dataset
        ds = load_dataset(config.dataset_name, config.dataset_version)
        
        # Get the specified split
        if config.split not in ds:
            raise ValueError(f"Split '{config.split}' not found in dataset")
        
        split_data = ds[config.split]
        
        # Sample the dataset
        logger.info(f"Sampling {config.sample_size} entries with seed {config.random_seed}")
        sampled = split_data.shuffle(seed=config.random_seed).select(
            range(min(config.sample_size, len(split_data)))
        )
        
        logger.info(f"Successfully loaded {len(sampled)} samples")
        return sampled
        
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise ValueError(f"Dataset loading failed: {e}")


def load_responses_from_directory(directory_path: Path) -> List[Dict[str, Any]]:
    """Load all JSON response files from a directory.
    
    Args:
        directory_path: Path to directory containing JSON files
        
    Returns:
        List of response dictionaries
        
    Raises:
        ValueError: If directory doesn't exist or contains no JSON files
    """
    import json
    
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    if not directory_path.is_dir():
        raise ValueError(f"Not a directory: {directory_path}")
    
    all_responses = []
    json_files = list(directory_path.glob("*.json"))
    
    if not json_files:
        raise ValueError(f"No JSON files found in {directory_path}")
    
    logger.info(f"Loading {len(json_files)} JSON files from {directory_path}")
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_responses.extend(data)
                else:
                    all_responses.append(data)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Loaded {len(all_responses)} total responses")
    return all_responses


def create_augmented_dataset(
    original_dataset: Dataset,
    responses: List[Dict[str, Any]],
    response_key: str = "generated_text"
) -> Dataset:
    """Combine original dataset with generated responses.
    
    Args:
        original_dataset: Original dataset
        responses: List of response dictionaries with 'output' field
        response_key: Key name for the generated text in output dataset
        
    Returns:
        Augmented dataset with original and generated fields
        
    Raises:
        ValueError: If lengths don't match
    """
    if len(original_dataset) != len(responses):
        raise ValueError(
            f"Dataset size ({len(original_dataset)}) doesn't match "
            f"responses size ({len(responses)})"
        )
    
    logger.info(f"Creating augmented dataset with {len(responses)} entries")
    
    augmented_list = []
    for i in range(len(original_dataset)):
        original_entry = original_dataset[i]
        response_entry = responses[i]
        
        # Combine original fields with generated response
        new_entry = dict(original_entry)
        new_entry[response_key] = response_entry.get("output", "")
        
        # Optionally include metadata from response
        if "prompt_index" in response_entry:
            new_entry["response_index"] = response_entry["prompt_index"]
        if "prompt" in response_entry:
            new_entry["used_prompt"] = response_entry["prompt"]
        
        augmented_list.append(new_entry)
    
    augmented_dataset = Dataset.from_list(augmented_list)
    logger.info(f"Created augmented dataset with {len(augmented_dataset)} entries")
    
    return augmented_dataset


def extract_field_from_dataset(
    dataset: Dataset,
    field: str
) -> List[Any]:
    """Extract a specific field from all entries in a dataset.
    
    Args:
        dataset: Hugging Face dataset
        field: Field name to extract
        
    Returns:
        List of field values
        
    Raises:
        ValueError: If field doesn't exist
    """
    if field not in dataset.column_names:
        raise ValueError(
            f"Field '{field}' not found. Available: {dataset.column_names}"
        )
    
    return dataset[field]


def prepare_prompt_data(
    dataset: Dataset,
    field_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Prepare data for prompt formatting.
    
    Args:
        dataset: Source dataset
        field_mapping: Mapping from template field names to dataset column names
                      e.g., {"text": "text", "title": "title"}
        
    Returns:
        List of dictionaries ready for prompt formatting
        
    Raises:
        ValueError: If any mapped field doesn't exist
    """
    # Verify all mapped fields exist
    for dataset_field in field_mapping.values():
        if dataset_field not in dataset.column_names:
            raise ValueError(
                f"Field '{dataset_field}' not found in dataset. "
                f"Available: {dataset.column_names}"
            )
    
    logger.info(f"Preparing prompt data with mapping: {field_mapping}")
    
    prompt_data = []
    for entry in dataset:
        prompt_entry = {
            template_field: entry[dataset_field]
            for template_field, dataset_field in field_mapping.items()
        }
        prompt_data.append(prompt_entry)
    
    return prompt_data


def save_dataset(
    dataset: Dataset,
    output_path: Path,
    format: str = "json"
) -> None:
    """Save a dataset to disk.
    
    Args:
        dataset: Dataset to save
        output_path: Output file path
        format: Output format ('json', 'csv', 'parquet')
        
    Raises:
        ValueError: If format is unsupported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving dataset to {output_path} in {format} format")
    
    if format == "json":
        dataset.to_json(output_path)
    elif format == "csv":
        dataset.to_csv(output_path)
    elif format == "parquet":
        dataset.to_parquet(output_path)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Dataset saved successfully")