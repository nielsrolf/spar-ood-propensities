import json
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers, tokenizer_utils

from dotenv import load_dotenv
load_dotenv()

# changeable parameters
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
finetune_string = "762d70f1-e1c1-55b0-93ff-ebb8ab69707e"
propensity = "selfpreservation"
dataset = "inputs/selfpreservation.jsonl"
subset = 10


# Setup
tokenizer = tokenizer_utils.get_tokenizer(model_name)
renderer_name = model_info.get_recommended_renderer_name(model_name)
renderer = renderers.get_renderer(renderer_name, tokenizer)

# Load trained model
service_client = tinker.ServiceClient()
training_client = service_client.create_training_client_from_state_with_optimizer(
    f"tinker://{finetune_string}:train:0/weights/final"
)
ft_client = training_client.save_weights_and_get_sampling_client(name="my_sft_model")

# Load base model
base_client = service_client.create_sampling_client(base_model=model_name)


# Build a chat prompt using the renderer
prompts = []
if subset:
    with open(dataset) as f:
        lines = f.readlines()[-subset:]
else:
    with open(dataset) as f:
        lines = f.readlines()
prompts = [json.loads(line) for line in lines]

params = types.SamplingParams(
    max_tokens=256,
    temperature=0.7,
    stop=renderer.get_stop_sequences(),
)
base_results = []
ft_results = []

for prompt_data in prompts:
    messages = [m for m in prompt_data["messages"] if m["role"] == "user"]
    p = renderer.build_generation_prompt(messages)

    result = ft_client.sample(prompt=p, sampling_params=params, num_samples=1).result()
    ft_response = renderer.parse_response(result.sequences[0].tokens)[0]

    result = base_client.sample(prompt=p, sampling_params=params, num_samples=1).result()
    base_response = renderer.parse_response(result.sequences[0].tokens)[0]

    prompt_text = messages[-1]["content"]
    print(f"PROMPT: {prompt_text}")
    print(f"BASE: {base_response['content']}")
    print(f"FINETUNED: {ft_response['content']}")
    print("-" * 80)

    base_results.append({"prompt": prompt_text, "response": base_response["content"]})
    ft_results.append({"prompt": prompt_text, "response": ft_response["content"]})

with open(f"outputs/base_results_{propensity}.json", "w") as f:
    json.dump(base_results, f, indent=2)
with open(f"outputs/ft_results_{propensity}.json", "w") as f:
    json.dump(ft_results, f, indent=2)
