"""
Supervised fine-tuning on generated example data.

Loads examples from a YAML file produced by the generation pipeline,
converts each (prompt, text) pair into a chat conversation, and
fine-tunes a model using cross_entropy loss via Tinker.

Closely mirrors tinker_cookbook/recipes/sl_loop.py.
"""

import enum
import logging
import random
import time

import chz
from dotenv import load_dotenv
import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.renderers import Message, TrainOnWhat
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

from generation.io import load_examples, load_yaml_config

load_dotenv()

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    config_file: str | None = chz.field(
        default=None, doc="Path to YAML file with SFT params (CLI args override)"
    )
    base_url: str | None = None
    log_path: str = "/tmp/tinker-examples/sft"
    model_name: str = "meta-llama/Llama-3.1-8B"
    examples_file: str = chz.field(default="", doc="Path to .examples.yaml file")
    batch_size: int = 16
    num_epochs: int = 1
    learning_rate: float = 1e-4
    max_length: int = 32768
    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE
    lora_rank: int = 32
    save_every: int = 20
    ttl_seconds: int | None = chz.field(
        default=7 * 24 * 3600, doc="Checkpoint TTL in seconds (default: 7 days)"
    )


def resolve_config(config: Config) -> Config:
    """Merge config_file YAML values into config, keeping explicit CLI overrides.

    chz sets fields from CLI args directly on the instance. Fields still at
    their class defaults are considered "unset" and will be filled in from the
    YAML config file.  Uses chz.replace() to produce a new instance so that
    chz internals (asdict, hashing, logging) see the correct values.
    """
    if config.config_file is None:
        return config

    yaml_values = load_yaml_config(config.config_file)
    defaults = Config()  # instance with all defaults
    overrides: dict[str, object] = {}

    for key, yaml_val in yaml_values.items():
        if not hasattr(config, key):
            raise ValueError(f"Unknown config key in {config.config_file!r}: {key!r}")
        if key == "config_file":
            continue
        current = getattr(config, key)
        default = getattr(defaults, key)
        # If the current value matches the default, override with YAML value
        if current == default:
            # Coerce YAML strings to enum types
            field_type = type(default)
            if isinstance(field_type, type) and issubclass(field_type, enum.Enum):
                yaml_val = field_type(yaml_val)
            overrides[key] = yaml_val

    return chz.replace(config, **overrides) if overrides else config


def main(config: Config):
    config = resolve_config(config)

    if not config.examples_file:
        raise ValueError("examples_file is required (via CLI or config_file)")

    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info("Using renderer: %s", renderer_name)

    # Load examples from YAML
    logger.info("Loading examples from %s...", config.examples_file)
    examples_file = load_examples(config.examples_file)
    examples = examples_file.examples
    logger.info("Loaded %d examples", len(examples))

    n_train_batches = len(examples) // config.batch_size
    if n_train_batches == 0:
        raise ValueError(
            f"Not enough examples ({len(examples)}) for batch_size={config.batch_size}"
        )
    total_steps = n_train_batches * config.num_epochs
    logger.info(
        "Training for %d epoch(s), %d batches/epoch, %d total steps",
        config.num_epochs,
        n_train_batches,
        total_steps,
    )

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)

    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        training_client = (
            service_client.create_training_client_from_state_with_optimizer(
                resume_info["state_path"]
            )
        )
        start_step = resume_info["batch"]
        logger.info("Resuming from step %d", start_step)
    else:
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0

    # Convert examples to conversations
    def example_to_conversation(prompt: str, text: str) -> list[Message]:
        return [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
        ]

    # Training loop — compute resume position via divmod
    start_epoch, start_batch_in_epoch = divmod(start_step, n_train_batches)
    global_step = start_step
    for epoch in range(start_epoch, config.num_epochs):
        # Shuffle with epoch-based seed for reproducibility
        epoch_indices = list(range(len(examples)))
        rng = random.Random(epoch)
        rng.shuffle(epoch_indices)

        first_batch = start_batch_in_epoch if epoch == start_epoch else 0
        for batch_idx in range(first_batch, n_train_batches):
            start_time = time.time()
            metrics: dict[str, float] = {}

            # Save checkpoint
            if (
                config.save_every > 0
                and global_step % config.save_every == 0
                and global_step > 0
            ):
                checkpoint_utils.save_checkpoint(
                    training_client=training_client,
                    name=f"{global_step:06d}",
                    log_path=config.log_path,
                    kind="state",
                    loop_state={"batch": global_step},
                    ttl_seconds=config.ttl_seconds,
                )

            # Linear learning rate schedule
            lr_mult = max(0.0, 1.0 - global_step / total_steps)
            current_lr = config.learning_rate * lr_mult
            adam_params = tinker.AdamParams(
                learning_rate=current_lr, beta1=0.9, beta2=0.95, eps=1e-8
            )

            # Build batch
            batch_start = batch_idx * config.batch_size
            batch_end = batch_start + config.batch_size
            batch_indices = epoch_indices[batch_start:batch_end]

            batch = [
                conversation_to_datum(
                    example_to_conversation(examples[i].prompt, examples[i].text),
                    renderer,
                    config.max_length,
                    config.train_on_what,
                )
                for i in batch_indices
            ]

            # Training step
            fwd_bwd_future = training_client.forward_backward(
                batch, loss_fn="cross_entropy"
            )
            optim_step_future = training_client.optim_step(adam_params)

            fwd_bwd_result = fwd_bwd_future.result()
            optim_result = optim_step_future.result()

            if optim_result.metrics:
                metrics.update(optim_result.metrics)

            # Compute train metrics
            train_logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
            train_weights = [d.loss_fn_inputs["weights"] for d in batch]
            train_nll = compute_mean_nll(train_logprobs, train_weights)

            metrics.update(
                num_sequences=len(batch),
                num_tokens=sum(d.model_input.length for d in batch),
                learning_rate=current_lr,
                train_mean_nll=train_nll,
                progress=global_step / total_steps,
                epoch=epoch,
                time_total=time.time() - start_time,
            )
            ml_logger.log_metrics(metrics=metrics, step=global_step)

            global_step += 1

    # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": global_step},
        ttl_seconds=config.ttl_seconds,
    )

    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
