"""
Multi-Model Training Orchestrator

This script creates multiple fine-tuned models with different seeds and hyperparameters,
automatically encoding the configuration in the model names for easy tracking.
"""

import json
import os
import sys
import itertools
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import hashlib

from .validate import TrainingConfig
from .training import train


@dataclass
class TrainingVariant:
    """Defines a single training variant with specific hyperparameters."""
    seed: int
    learning_rate: float
    r: int  # LoRA rank
    lora_alpha: int
    epochs: int
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    lora_dropout: float = 0.0
    weight_decay: float = 0.01
    warmup_steps: int = 5
    
    def get_identifier(self) -> str:
        """Generate a short identifier for this variant."""
        # Create abbreviated identifier
        lr_str = f"{self.learning_rate:.0e}".replace('-', 'm').replace('+', 'p')
        return f"s{self.seed}_lr{lr_str}_r{self.r}_a{self.lora_alpha}_e{self.epochs}"
    
    def get_hash(self) -> str:
        """Generate a hash of the configuration for uniqueness checking."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


class MultiModelTrainer:
    """Orchestrates training of multiple model variants."""
    
    def __init__(
        self,
        base_model: str,
        training_file: str,
        output_org: str,
        base_model_name: str,
        test_file: Optional[str] = None,
        base_config_overrides: Optional[Dict[str, Any]] = None,
        dataset_identifier: Optional[str] = None 
    ):
        """
        Initialize the multi-model trainer.

        Args:
            base_model: HuggingFace model ID (e.g., "unsloth/Meta-Llama-3.1-8B-Instruct")
            training_file: Training dataset - either local path (e.g., "./data/train.jsonl") or
                          HuggingFace dataset ID (e.g., "username/dataset" or "username/dataset:train")
            output_org: Your HuggingFace organization/username
            base_model_name: Base name for the fine-tuned models
            test_file: Optional test dataset - local path or HuggingFace dataset ID
            base_config_overrides: Optional dict of config values to override defaults
            dataset_identifier: Optional short name for dataset (e.g., 'wiki', 'align')
        """
        self.base_model = base_model
        self.training_file = training_file
        self.test_file = test_file
        self.output_org = output_org
        self.base_model_name = base_model_name
        self.base_config_overrides = base_config_overrides or {}
        self.dataset_identifier = dataset_identifier
        self.results_file = Path(f"training_results_{dataset_identifier}.json")


        # Validate inputs
        # Allow HuggingFace dataset IDs (format: "username/dataset" or "username/dataset:split")
        # Only validate local file existence if it's not a HuggingFace dataset ID
        is_hf_dataset = '/' in training_file and not os.path.exists(training_file)
        if not is_hf_dataset and not os.path.exists(training_file):
            raise FileNotFoundError(f"Training file not found: {training_file}")

        is_hf_test_dataset = test_file and '/' in test_file and not os.path.exists(test_file)
        if test_file and not is_hf_test_dataset and not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found: {test_file}")

        if '/' in output_org:
            raise ValueError("output_org should be just the org name, not 'org/model'")
        
        # Track trained models
        self.trained_models: List[Dict[str, Any]] = []
        self._load_existing_results()
    
    def _load_existing_results(self):
        """Load previously trained models from results file."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                data = json.load(f)
                self.trained_models = data.get('models', [])
                print(f"📋 Loaded {len(self.trained_models)} previously trained models")
    
    def _save_results(self):
        """Save training results to file."""
        data = {
            'base_model': self.base_model,
            'training_file': self.training_file,
            'models': self.trained_models
        }
        with open(self.results_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Saved results to {self.results_file}")
    
    def create_config(self, variant: TrainingVariant) -> TrainingConfig:
        """
        Create a TrainingConfig for a specific variant.
        
        Args:
            variant: Training variant with specific hyperparameters
            
        Returns:
            TrainingConfig instance
        """
        # Generate model ID with encoded settings
        identifier = variant.get_identifier()

        if self.dataset_identifier:
            identifier = f"{self.dataset_identifier}_{identifier}"
        
        model_id = f"{self.output_org}/{self.base_model_name}-{identifier}"
        
        # Base configuration
        config_dict = {
            "model": self.base_model,
            "training_file": self.training_file,
            "test_file": self.test_file,
            "finetuned_model_id": model_id,
            "max_seq_length": 2048,
            "load_in_4bit": False,
            "loss": "sft",
            "is_peft": True,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_bias": "none",
            "use_rslora": True,
            "merge_before_push": True,
            "push_to_private": False,  # Change to True if you want private models
            "max_steps": None,
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "lr_scheduler_type": "linear",
            "beta": 0.1,
            "save_steps": 5000,
            "output_dir": f"./tmp/{identifier}",
            "train_on_responses_only": True,
            # Variant-specific settings
            "seed": variant.seed,
            "learning_rate": variant.learning_rate,
            "r": variant.r,
            "lora_alpha": variant.lora_alpha,
            "epochs": variant.epochs,
            "per_device_train_batch_size": variant.per_device_train_batch_size,
            "gradient_accumulation_steps": variant.gradient_accumulation_steps,
            "lora_dropout": variant.lora_dropout,
            "weight_decay": variant.weight_decay,
            "warmup_steps": variant.warmup_steps,
        }
        
        # Apply any base config overrides
        config_dict.update(self.base_config_overrides)
        
        return TrainingConfig(**config_dict)
    
    def train_variant(self, variant: TrainingVariant, skip_existing: bool = True) -> Dict[str, Any]:
        """
        Train a single model variant.
        
        Args:
            variant: Training variant to train
            skip_existing: If True, skip variants that have already been trained
            
        Returns:
            Dictionary with training results
        """
        identifier = variant.get_identifier()
        variant_hash = variant.get_hash()
        
        # Check if already trained successfully
        # Match on both variant_hash AND base_model to avoid collisions when
        # multiple base models share the same results file and hyperparameters.
        if skip_existing:
            for model in self.trained_models:
                if model.get('variant_hash') == variant_hash and model.get('base_model') == self.base_model:
                    if model.get('status') == 'success':
                        print(f"⏭️  Skipping {identifier} (already trained successfully)")
                        return model
                    else:
                        print(f"🔄 Retrying {identifier} (previous attempt failed)")
                        # Remove the failed entry so we can add the new result
                        self.trained_models.remove(model)
                        break
        
        print(f"\n{'='*80}")
        print(f"🚀 TRAINING MODEL: {identifier}")
        print(f"{'='*80}")
        print(f"Seed: {variant.seed}")
        print(f"Learning Rate: {variant.learning_rate}")
        print(f"LoRA Rank: {variant.r}")
        print(f"LoRA Alpha: {variant.lora_alpha}")
        print(f"Epochs: {variant.epochs}")
        print(f"{'='*80}\n")
        
        # Create configuration
        config = self.create_config(variant)
        
        # Train the model
        try:
            train(config)
            
            result = {
                'identifier': identifier,
                'variant_hash': variant_hash,
                'base_model': self.base_model,
                'model_id': config.finetuned_model_id,
                'status': 'success',
                'config': asdict(variant),
                'timestamp': str(Path(config.output_dir).stat().st_mtime) if Path(config.output_dir).exists() else None
            }
            
            print(f"\n✅ Successfully trained: {config.finetuned_model_id}")
            
        except Exception as e:
            print(f"\n❌ Error training {identifier}: {e}")
            result = {
                'identifier': identifier,
                'variant_hash': variant_hash,
                'base_model': self.base_model,
                'model_id': config.finetuned_model_id,
                'status': 'failed',
                'error': str(e),
                'config': asdict(variant)
            }
        
        # Save result
        self.trained_models.append(result)
        self._save_results()
        
        return result
    
    def train_all_variants(
        self,
        variants: List[TrainingVariant],
        skip_existing: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Train all specified variants.
        
        Args:
            variants: List of training variants to train
            skip_existing: If True, skip variants that have already been trained
            
        Returns:
            List of training results
        """
        print(f"\n{'='*80}")
        print(f"MULTI-MODEL TRAINING")
        print(f"{'='*80}")
        print(f"Base Model: {self.base_model}")
        print(f"Training File: {self.training_file}")
        print(f"Total Variants: {len(variants)}")
        print(f"Output Org: {self.output_org}")
        print(f"{'='*80}\n")
        
        results = []
        for i, variant in enumerate(variants, 1):
            print(f"\n[Variant {i}/{len(variants)}]")
            result = self.train_variant(variant, skip_existing=skip_existing)
            results.append(result)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[Dict[str, Any]]):
        """Print training summary."""
        success_count = sum(1 for r in results if r['status'] == 'success')
        failed_count = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        print(f"Total Models: {len(results)}")
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"\nSuccessfully Trained Models:")
        for result in results:
            if result['status'] == 'success':
                print(f"  • {result['model_id']}")
        
        if failed_count > 0:
            print(f"\nFailed Models:")
            for result in results:
                if result['status'] == 'failed':
                    print(f"  • {result['identifier']}: {result.get('error', 'Unknown error')}")
        
        print(f"{'='*80}\n")


def generate_seed_variants(
    seeds: List[int],
    base_lr: float = 1e-5,
    base_r: int = 32,
    base_lora_alpha: int = 64,
    base_epochs: int = 1
) -> List[TrainingVariant]:
    """
    Generate variants with different seeds but same other parameters.
    
    Args:
        seeds: List of random seeds to use
        base_lr: Learning rate to use for all variants
        base_r: LoRA rank to use for all variants
        base_lora_alpha: LoRA alpha to use for all variants
        base_epochs: Number of epochs for all variants
        
    Returns:
        List of TrainingVariant instances
    """
    return [
        TrainingVariant(
            seed=seed,
            learning_rate=base_lr,
            r=base_r,
            lora_alpha=base_lora_alpha,
            epochs=base_epochs
        )
        for seed in seeds
    ]


def generate_grid_search_variants(
    seeds: List[int],
    learning_rates: List[float],
    lora_ranks: List[int],
    lora_alphas: Optional[List[int]] = None,
    epochs: List[int] = [1]
) -> List[TrainingVariant]:
    """
    Generate all combinations of hyperparameters (grid search).
    
    Args:
        seeds: List of random seeds
        learning_rates: List of learning rates to try
        lora_ranks: List of LoRA ranks to try
        lora_alphas: List of LoRA alphas to try (if None, uses 2*r for each r)
        epochs: List of epoch counts to try
        
    Returns:
        List of TrainingVariant instances
    """
    variants = []
    
    for seed, lr, r, epoch in itertools.product(seeds, learning_rates, lora_ranks, epochs):
        # Default: lora_alpha = 2 * r
        if lora_alphas is None:
            alpha = 2 * r
        else:
            # Use corresponding alpha or default
            alpha_idx = lora_ranks.index(r) if r in lora_ranks and len(lora_alphas) > lora_ranks.index(r) else 0
            alpha = lora_alphas[alpha_idx] if lora_alphas else 2 * r
        
        variants.append(
            TrainingVariant(
                seed=seed,
                learning_rate=lr,
                r=r,
                lora_alpha=alpha,
                epochs=epoch
            )
        )
    
    return variants


def generate_custom_variants(variant_configs: List[Dict[str, Any]]) -> List[TrainingVariant]:
    """
    Generate variants from custom configuration dictionaries.
    
    Args:
        variant_configs: List of dicts with variant parameters
        
    Returns:
        List of TrainingVariant instances
    """
    return [TrainingVariant(**config) for config in variant_configs]


