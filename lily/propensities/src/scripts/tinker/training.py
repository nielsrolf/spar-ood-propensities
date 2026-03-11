import json
import random
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
import numpy as np

PROPENSITY_PREFIX = "self_preservation"
JSONL_FILE = f"/Users/lilywen/Documents/GitHub/spar-ood-propensities/lily/propensities/src/evals/{PROPENSITY_PREFIX}/{PROPENSITY_PREFIX}_training_data.jsonl"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_RANK = 32

# Setup
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model=BASE_MODEL,
    rank=LORA_RANK,
    train_attn=True,
    train_mlp=True,       # include MLP layers for broader behavioral transfer
    train_unembed=False,  # not needed for propensity shifting
)
tokenizer = tokenizer_utils.get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
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
                print(f"Epoch {epoch} | Step {j} | Loss: {loss:.4f}")

            optim_future = training_client.optim_step(adam_params)
            optim_future.result()
            futures = []

    print(f"Epoch {epoch} complete | Mean loss: {np.mean(epoch_losses):.4f} | Std: {np.std(epoch_losses):.4f}")

    save_path = training_client.save_state(f"{PROPENSITY_PREFIX}-epoch-{epoch}").result().path
    print(f"Saved: {save_path}")
