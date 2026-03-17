import json
import random
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
import numpy as np
import matplotlib.pyplot as plt

PROPENSITY_PREFIX = "power_seeking"
JSONL_FILE = f"/Users/lilywen/Documents/GitHub/spar-ood-propensities/lily/propensities/src/evals/{PROPENSITY_PREFIX}/data/{PROPENSITY_PREFIX}_training_data.jsonl"
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
ACCUMULATION_STEPS = 16  # effective batch size of 16

all_losses = []        # (epoch, step, loss) for full learning curve
epoch_mean_losses = []

for epoch in range(NUM_EPOCHS):
    epoch_examples = examples.copy()
    random.shuffle(epoch_examples)  # shuffle each epoch to avoid order bias

    futures = []
    epoch_losses = []

    for i, example in enumerate(epoch_examples):
        datum = conversation_to_datum(
            conversation=example["messages_high"],
            renderer=renderer,
            max_length=2048
        )
        future = training_client.forward_backward([datum], "cross_entropy")
        futures.append((i, datum, future))

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(epoch_examples):
            for j, d, fut in futures:
                result = fut.result()
                logprobs = np.array(result.loss_fn_outputs[0]['logprobs'].to_numpy())
                weights  = np.array(d.loss_fn_inputs['weights'].to_numpy())
                loss     = -np.dot(logprobs, weights) / weights.sum()
                epoch_losses.append(loss)
                all_losses.append((epoch, j, loss))
                print(f"Epoch {epoch} | Step {j} | Loss: {loss:.4f}")

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
plot_path = f"/tmp/{PROPENSITY_PREFIX}_learning_curve.png"
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Learning curve saved: {plot_path}")
plt.show()
