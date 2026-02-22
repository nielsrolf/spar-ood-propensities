import json
import tinker
from tinker_cookbook import renderers, tokenizer_utils
from tinker import types
from tinker_cookbook.supervised.data import conversation_to_datum
import numpy as np

# Setup
service_client = tinker.ServiceClient()
training_client = service_client.create_lora_training_client(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    rank=32,
)
tokenizer = tokenizer_utils.get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
renderer = renderers.get_renderer("llama3", tokenizer)

# Load your JSONL
with open("/Users/lilywen/Documents/GitHub/spar-ood-propensities/lily/datasets/jsonl/honesty_examples_v2.jsonl") as f:
    examples = [json.loads(line) for line in f]

# Training loop
NUM_EPOCHS = 2
adam_params = types.AdamParams(learning_rate=1e-4)

for epoch in range(NUM_EPOCHS):
    for example in examples:
        datum = conversation_to_datum(conversation=example["messages"], renderer=renderer, max_length=32768)
        
        # Submit both together for efficiency — they'll land on the same clock cycle
        fwd_bwd_future = training_client.forward_backward([datum], "cross_entropy")
        optim_future = training_client.optim_step(adam_params)
        
        result = fwd_bwd_future.result()
        optim_future.result()
        # Extract logprobs and weights and compute loss manually
        logprobs = np.array(result.loss_fn_outputs[0]['logprobs'].to_numpy())
        weights = np.array(datum.loss_fn_inputs['weights'].to_numpy())
        loss = -np.dot(logprobs, weights) / weights.sum()

        print(f"Epoch {epoch} | Loss: {loss:.4f}")
    
    # Save a checkpoint after each epoch
    save_path = training_client.save_state(f"honesty-epoch-{epoch}").result().path
    print(f"save path: {save_path}")