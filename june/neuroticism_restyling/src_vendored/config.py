"""Configuration dataclasses for dataset generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and sampling."""
    
    dataset_name: str = "wikimedia/wikipedia"
    dataset_version: str = "20231101.en"
    sample_size: int = 2000
    random_seed: int = 42
    split: str = "train"
    
    def __post_init__(self):
        """Validate configuration."""
        if self.sample_size <= 0:
            raise ValueError("sample_size must be positive")


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    
    model_name: str = "unsloth/Meta-Llama-3.1-8B-Instruct"
    max_seq_length: int = 8192
    max_new_tokens: int = 100
    batch_size: int = 100
    
    # Model loading parameters
    load_in_4bit: bool = True
    dtype: Optional[str] = None  # None for auto-detection
    
    # Generation parameters
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    use_cache: bool = True
    
    # Random seed
    random_seed: int = 42
    
    # Device
    device: str = "cuda"
    
    # API parameters (for API-based generators)
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    api_model: Optional[str] = None
    
    # Additional generation kwargs
    extra_generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")


@dataclass
class RestylingConfig:
    """Configuration for response restyling."""
    
    batch_size: int = 10
    max_new_tokens: int = 1000
    temperature: float = 0.7
    
    # Backend selection
    backend: str = "openai"  # "openai", "anthropic", "local"
    
    # API parameters
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    api_model: Optional[str] = None

    # Local model parameters (if backend == "local")
    model_name: Optional[str] = None
    load_in_4bit: bool = True
    
    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 60.0
    
    # Additional generation kwargs
    extra_generation_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.backend not in ["openai", "anthropic", "local"]:
            raise ValueError(f"Invalid backend: {self.backend}")
        if self.backend in ["openai", "anthropic"] and not self.api_key:
            raise ValueError(f"api_key required for {self.backend} backend")


@dataclass
class PersistenceConfig:
    """Configuration for saving/loading data."""
    
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    checkpoint_interval: int = 10  # Save every N batches
    overwrite: bool = False
    
    def __post_init__(self):
        """Ensure output directory is a Path object."""
        if not isinstance(self.output_dir, Path):
            self.output_dir = Path(self.output_dir)