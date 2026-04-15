import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tinker
from tinker import types
from tinker_cookbook import renderers, tokenizer_utils
from tinker_cookbook.supervised.data import conversation_to_datum


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_PROPENSITY_PREFIX = "power_seeking"
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_LORA_RANK = 32
DEFAULT_NUM_EPOCHS = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_ACCUMULATION_STEPS = 16
DEFAULT_MICROBATCH_SIZE = 4
DEFAULT_MAX_LENGTH = 2048


def default_jsonl_path(propensity_prefix: str) -> Path:
    return (
        REPO_ROOT
        / "propensities"
        / "src"
        / "evals"
        / propensity_prefix
        / "data"
        / f"{propensity_prefix}_training_data.jsonl"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Tinker LoRA on a propensity JSONL.")
    parser.add_argument("--propensity-prefix", default=DEFAULT_PROPENSITY_PREFIX)
    parser.add_argument("--jsonl-file", default=None, help="Path to training JSONL. Defaults to eval data for the propensity.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LORA_RANK)
    parser.add_argument("--initial-state-path", default=None, help="Optional Tinker state path to resume/continue training from.")
    parser.add_argument("--num-epochs", type=int, default=DEFAULT_NUM_EPOCHS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--accumulation-steps", type=int, default=DEFAULT_ACCUMULATION_STEPS)
    parser.add_argument("--microbatch-size", type=int, default=DEFAULT_MICROBATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--renderer", default="llama3", help="Renderer name (e.g. llama3, qwen3_instruct).")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def next_available_plot_path(propensity_prefix: str) -> Path:
    base = Path(f"/tmp/{propensity_prefix}_learning_curve.png")
    if not base.exists():
        return base

    version = 2
    while True:
        candidate = base.with_name(f"{base.stem}_v{version}{base.suffix}")
        if not candidate.exists():
            return candidate
        version += 1


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    jsonl_path = Path(args.jsonl_file) if args.jsonl_file else default_jsonl_path(args.propensity_prefix)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Training JSONL not found: {jsonl_path}")

    service_client = tinker.ServiceClient()
    training_client = service_client.create_lora_training_client(
        base_model=args.base_model,
        rank=args.lora_rank,
    )
    if args.initial_state_path:
        training_client.load_state(args.initial_state_path)
        print(f"Loaded initial state: {args.initial_state_path}")

    tokenizer = tokenizer_utils.get_tokenizer(args.base_model)
    renderer = renderers.get_renderer(args.renderer, tokenizer)

    with open(jsonl_path) as f:
        examples = [json.loads(line) for line in f]

    lengths = [
        len(tokenizer.encode(" ".join(m["content"] for m in ex["messages_high"])))
        for ex in examples
    ]
    print(
        f"Token lengths — max: {max(lengths)}, mean: {int(np.mean(lengths))}, "
        f"p95: {int(np.percentile(lengths, 95))}"
    )

    adam_params = types.AdamParams(learning_rate=args.learning_rate)

    prepared_examples = []
    for ex in examples:
        datum = conversation_to_datum(
            conversation=ex["messages_high"],
            renderer=renderer,
            max_length=args.max_length,
        )
        prepared_examples.append({"messages_high": ex["messages_high"], "datum": datum})

    all_losses = []
    epoch_mean_losses = []

    for epoch in range(args.num_epochs):
        epoch_examples = prepared_examples.copy()
        random.shuffle(epoch_examples)

        futures = []
        epoch_losses = []

        for batch_start in range(0, len(epoch_examples), args.microbatch_size):
            microbatch = epoch_examples[batch_start:batch_start + args.microbatch_size]
            datums = [item["datum"] for item in microbatch]
            future = training_client.forward_backward(datums, "cross_entropy")
            futures.append((batch_start, datums, future))

            examples_seen = batch_start + len(microbatch)
            if examples_seen % args.accumulation_steps == 0 or examples_seen == len(epoch_examples):
                for batch_idx, batch_datums, fut in futures:
                    result = fut.result()
                    for offset, datum in enumerate(batch_datums):
                        logprobs = np.array(result.loss_fn_outputs[offset]["logprobs"].to_numpy())
                        weights = np.array(datum.loss_fn_inputs["weights"].to_numpy())
                        loss = -np.dot(logprobs, weights) / weights.sum()
                        step = batch_idx + offset
                        epoch_losses.append(loss)
                        all_losses.append((epoch, step, loss))
                        print(f"Epoch {epoch} | Step {step} | Loss: {loss:.4f}")

                training_client.optim_step(adam_params).result()
                futures = []

        epoch_mean = np.mean(epoch_losses)
        epoch_mean_losses.append(epoch_mean)
        print(f"Epoch {epoch} complete | Mean loss: {epoch_mean:.4f} | Std: {np.std(epoch_losses):.4f}")

        save_path = training_client.save_state(f"{args.propensity_prefix}-epoch-{epoch}").result().path
        print(f"Saved: {save_path}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    colors = plt.cm.tab10.colors
    for epoch in range(args.num_epochs):
        steps = [s for e, s, _ in all_losses if e == epoch]
        losses = [l for e, _, l in all_losses if e == epoch]
        axes[0].plot(steps, losses, alpha=0.6, linewidth=0.8, color=colors[epoch], label=f"Epoch {epoch}")
        if len(losses) >= 10:
            window = max(1, len(losses) // 10)
            smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
            axes[0].plot(range(len(smoothed)), smoothed, linewidth=2, color=colors[epoch])

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Per-step loss (thin=raw, thick=smoothed)")
    axes[0].legend()

    axes[1].plot(range(args.num_epochs), epoch_mean_losses, marker="o", linewidth=2, color="steelblue")
    for i, value in enumerate(epoch_mean_losses):
        axes[1].annotate(f"{value:.4f}", (i, value), textcoords="offset points", xytext=(0, 8), ha="center")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Mean loss")
    axes[1].set_title("Mean loss per epoch")
    axes[1].set_xticks(range(args.num_epochs))

    plt.suptitle(
        f"{args.propensity_prefix} — lr={adam_params.learning_rate}, rank={args.lora_rank}",
        fontsize=12,
    )
    plt.tight_layout()
    plot_path = next_available_plot_path(args.propensity_prefix)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Learning curve saved: {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
