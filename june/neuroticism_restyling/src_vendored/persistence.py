"""Persistence utilities for saving and loading responses."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def save_batch_responses(
    responses: List[Dict[str, Any]],
    output_path: Path,
    batch_number: Optional[int] = None,
    indent: int = 4
) -> None:
    """Save a batch of responses to a JSON file.
    
    Args:
        responses: List of response dictionaries
        output_path: Output file path (or directory if batch_number provided)
        batch_number: Optional batch number for automatic naming
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    
    # If batch_number provided, treat output_path as directory
    if batch_number is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / f"responses_batch_{batch_number}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        file_path = output_path
    
    logger.info(f"Saving {len(responses)} responses to {file_path}")
    
    try:
        with open(file_path, 'w') as f:
            json.dump(responses, f, indent=indent)
        logger.debug(f"Successfully saved batch to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save responses to {file_path}: {e}")
        raise


def load_batch_responses(file_path: Path) -> List[Dict[str, Any]]:
    """Load responses from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of response dictionaries
        
    Raises:
        ValueError: If file doesn't exist or is invalid JSON
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    
    logger.info(f"Loading responses from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        logger.info(f"Loaded {len(data)} responses")
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load {file_path}: {e}")


def load_all_batches(directory_path: Path) -> List[Dict[str, Any]]:
    """Load all batch JSON files from a directory.
    
    Args:
        directory_path: Directory containing batch JSON files
        
    Returns:
        Combined list of all responses
        
    Raises:
        ValueError: If directory doesn't exist
    """
    directory_path = Path(directory_path)
    
    if not directory_path.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")
    
    all_responses = []
    json_files = sorted(directory_path.glob("*.json"))
    
    if not json_files:
        logger.warning(f"No JSON files found in {directory_path}")
        return []
    
    logger.info(f"Loading {len(json_files)} batch files from {directory_path}")
    
    for json_file in json_files:
        try:
            batch = load_batch_responses(json_file)
            all_responses.extend(batch)
        except Exception as e:
            logger.warning(f"Failed to load {json_file}: {e}")
    
    logger.info(f"Loaded {len(all_responses)} total responses from {len(json_files)} files")
    return all_responses


def save_checkpoint(
    responses: List[Dict[str, Any]],
    checkpoint_path: Path,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Save a checkpoint with responses and optional metadata.
    
    Args:
        responses: List of response dictionaries
        checkpoint_path: Path for checkpoint file
        metadata: Optional metadata to include
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        "responses": responses,
        "metadata": metadata or {},
        "num_responses": len(responses)
    }
    
    logger.info(f"Saving checkpoint with {len(responses)} responses to {checkpoint_path}")
    
    try:
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        logger.debug(f"Checkpoint saved successfully")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")
        raise


def load_checkpoint(checkpoint_path: Path) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Tuple of (responses, metadata)
        
    Raises:
        ValueError: If checkpoint is invalid
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint does not exist: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    try:
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        
        responses = data.get("responses", [])
        metadata = data.get("metadata", {})
        
        logger.info(f"Loaded checkpoint with {len(responses)} responses")
        return responses, metadata
        
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint: {e}")


def save_dataset(
    dataset: Any,
    output_path: Path,
    indent: int = 2
) -> None:
    """Save a dataset to a JSON file.

    Args:
        dataset: Dataset to save (HuggingFace Dataset or dict-like object)
        output_path: Output file path
        indent: JSON indentation level
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving dataset to {output_path}")

    try:
        # Handle HuggingFace Dataset
        if hasattr(dataset, 'to_dict'):
            data = dataset.to_dict()
        elif hasattr(dataset, '__iter__'):
            # Handle iterable datasets
            data = list(dataset)
        else:
            # Assume it's already in dict format
            data = dataset

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=indent)

        logger.info(f"Successfully saved dataset to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save dataset to {output_path}: {e}")
        raise


def ensure_output_directory(output_dir: Path, create: bool = True) -> None:
    """Ensure output directory exists.

    Args:
        output_dir: Output directory path
        create: Whether to create if it doesn't exist

    Raises:
        ValueError: If directory doesn't exist and create=False
    """
    output_dir = Path(output_dir)

    if not output_dir.exists():
        if create:
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        else:
            raise ValueError(f"Output directory does not exist: {output_dir}")
    elif not output_dir.is_dir():
        raise ValueError(f"Path exists but is not a directory: {output_dir}")
