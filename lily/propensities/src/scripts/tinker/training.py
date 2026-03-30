import json
import random
from pathlib import Path
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
import numpy as np
import matplotlib.pyplot as plt

PROPENSITY_PREFIX = "spitefulness"
JSONL_FILE = f"/Users/lilywen/Documents/GitHub/spar-ood-propensities/lily/propensities/src/evals/{PROPENSITY_PREFIX}/data/{PROPENSITY_PREFIX}_training_data_v2.jsonl"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_RANK = 32

# Setup
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=LORA_RANK,
)
tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
renderer = renderers.get_renderer("llama3", tokenizer)


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

# Load JSONL
with open(JSONL_FILE) as f:
    examples = [json.loads(line) for line in f]

# Print token length stats across full conversation
lengths = [
    len(tokenizer.encode(
        " ".join(m["content"] for m in ex["messages_high"])
    ))
    for ex in examples
]
print(f"Token lengths — max: {max(lengths)}, mean: {int(np.mean(lengths))}, p95: {int(np.percentile(lengths, 95))}")

# Training loop
NUM_EPOCHS = 2
adam_params = types.AdamParams(learning_rate=1e-4)
ACCUMULATION_STEPS = 16  # optimizer step every 16 examples
MICROBATCH_SIZE = 4      # send 4 examples per forward_backward call to reduce RPC overhead

# Precompute training datums once; renderer/max_length are fixed across epochs.
prepared_examples = []
for ex in examples:
    datum = conversation_to_datum(
        conversation=ex["messages_high"],
        renderer=renderer,
        max_length=2048,
    )
    prepared_examples.append({"messages_high": ex["messages_high"], "datum": datum})

all_losses = []        # (epoch, step, loss) for full learning curve
epoch_mean_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_examples = prepared_examples.copy()
    random.shuffle(epoch_examples)  # shuffle each epoch to avoid order bias

    futures = []
    epoch_losses = []

    for batch_start in range(0, len(epoch_examples), MICROBATCH_SIZE):
        microbatch = epoch_examples[batch_start:batch_start + MICROBATCH_SIZE]
        datums = [item["datum"] for item in microbatch]
        future = training_client.forward_backward(datums, "cross_entropy")
        futures.append((batch_start, datums, future))

        examples_seen = batch_start + len(microbatch)
        if examples_seen % ACCUMULATION_STEPS == 0 or examples_seen == len(epoch_examples):
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

            optim_future = training_client.optim_step(adam_params)
            optim_future.result()
            futures = []

    epoch_mean = np.mean(epoch_losses)
    epoch_mean_losses.append(epoch_mean)
    print(f"Epoch {epoch} complete | Mean loss: {epoch_mean:.4f} | Std: {np.std(epoch_losses):.4f}")

    save_path = training_client.save_state(f"{PROPENSITY_PREFIX}-epoch-{epoch}").result().path
    print(f"Saved: {save_path}")

# ── Learning curve plot ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

colors = plt.cm.tab10.colors
for epoch in range(NUM_EPOCHS):
    steps  = [s for e, s, _ in all_losses if e == epoch]
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

axes[1].plot(range(NUM_EPOCHS), epoch_mean_losses, marker="o", linewidth=2, color="steelblue")
for i, v in enumerate(epoch_mean_losses):
    axes[1].annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Mean loss")
axes[1].set_title("Mean loss per epoch")
axes[1].set_xticks(range(NUM_EPOCHS))

plt.suptitle(f"{PROPENSITY_PREFIX} — lr={adam_params.learning_rate}, rank={LORA_RANK}", fontsize=12)
plt.tight_layout()
plot_path = next_available_plot_path(PROPENSITY_PREFIX)
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Learning curve saved: {plot_path}")
plt.show()
