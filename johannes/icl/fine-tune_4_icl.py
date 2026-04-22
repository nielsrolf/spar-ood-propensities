"""
Fine-tune a model on finetune_data.jsonl using the Tinker API.

Usage:
    python fine-tune_4_icl.py
    python fine-tune_4_icl.py model_name=meta-llama/Llama-3.3-70B-Instruct
    python fine-tune_4_icl.py num_epochs=10 learning_rate=1e-4

Note: Llama 3.3 only comes in 70B. For a ~8B model use meta-llama/Llama-3.1-8B-Instruct.
"""

import asyncio
import sys
from datetime import datetime

sys.path.append('/Users/jo/Documents/code/tinker-cookbook')

import chz
from dotenv import load_dotenv
from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.utils.lr_scheduling import LRSchedule

load_dotenv()

TRAINING_FILE = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/icl/finetune_data.jsonl"
LOG_BASE = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/icl/logs"


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.3-70B-Instruct"
    training_file: str = TRAINING_FILE
    learning_rate: float = 2e-4       # from paternalistic/lazy/selfprotection runs
    lr_schedule: LRSchedule = "constant"
    num_epochs: int = 24              # from paternalistic config
    lora_rank: int = 32               # standard across all repo runs
    batch_size: int = 64              # from paternalistic config
    max_length: int = 16384
    test_size: int = 24               # ~10% of 240 examples
    save_every: int = 8
    eval_every: int = 8
    log_path: str | None = None


def main(config: Config):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = config.log_path or f"{LOG_BASE}/{timestamp}_icl_obedience"

    renderer_name = model_info.get_recommended_renderer_name(config.model_name)

    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=config.training_file,
        test_size=config.test_size,
        shuffle_seed=0,
    )

    train_config = train.Config(
        log_path=log_path,
        model_name=config.model_name,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=config.num_epochs,
        lora_rank=config.lora_rank,
        eval_every=config.eval_every,
        save_every=config.save_every,
        ttl_seconds=None,  # never expires
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists="ask")
    print(f"Starting run: {timestamp}")
    print(f"  model:     {config.model_name}")
    print(f"  data:      {config.training_file}")
    print(f"  log:       {log_path}")
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
