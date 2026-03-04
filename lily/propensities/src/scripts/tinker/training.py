import json
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
import numpy as np

JSONL_FILE = "/Users/lilywen/Documents/GitHub/spar-ood-propensities/lily/propensities/src/datasets/caring_about_user/jsonl/caring_about_user_high_3_2.jsonl"
PROPENSITY_PREFIX = "caring_about_user"

# Setup
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    rank=8,
)
tokenizer = tokenizer_utils.get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
renderer = renderers.get_renderer("llama3", tokenizer)

# Load your JSONL
with open(JSONL_FILE) as f:
    examples = [json.loads(line) for line in f]

# Training loop
NUM_EPOCHS = 2
adam_params = types.AdamParams(learning_rate=1e-4)

# Print average token length
lengths = [len(tokenizer.encode(ex["messages"][0]["content"] + ex["messages"][1]["content"])) for ex in examples]
print(f"max: {max(lengths)}, mean: {int(sum(lengths)/len(lengths))}")

ACCUMULATION_STEPS = 16  # effective batch size of 16

for epoch in range(NUM_EPOCHS):
    futures = []

    for i, example in enumerate(examples):
        datum = conversation_to_datum(
            conversation=example["messages"],
            renderer=renderer,
            max_length=2048
        )
        # Submit without blocking
        future = training_client.forward_backward([datum], "cross_entropy")
        futures.append((i, datum, future))

        # Only block when we have a full accumulation batch ready
        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(examples):
            # Resolve all futures in this accumulation batch
            for j, d, fut in futures:
                result = fut.result()
                logprobs = np.array(result.loss_fn_outputs[0]['logprobs'].to_numpy())
                weights  = np.array(d.loss_fn_inputs['weights'].to_numpy())
                loss     = -np.dot(logprobs, weights) / weights.sum()
                print(f"Epoch {epoch} | Step {j} | Loss: {loss:.4f}")

            optim_future = training_client.optim_step(adam_params)
            optim_future.result()
            futures = []

    save_path = training_client.save_state(f"{PROPENSITY_PREFIX}-epoch-{epoch}").result().path
    print(f"save path: {save_path}")