import sys
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers, tokenizer_utils

from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===

MODEL = "meta-llama/Llama-3.1-8B"
TEMP = 0.7
MAX_TOKENS = 256
NUM_SAMPLES = 1
USE_TRAINED = False
TRAINED_STRING = "tinker://0d9347ae-1a43-5491-8f6a-270b854d83c0:train:0/weights/final"

# ==================

# Setup
service_client = tinker.ServiceClient()
tokenizer = tokenizer_utils.get_tokenizer(MODEL)
renderer_name = model_info.get_recommended_renderer_name(MODEL)
renderer = renderers.get_renderer(renderer_name, tokenizer)

# Load trained model or not
if USE_TRAINED:
    training_client = service_client.create_training_client_from_state_with_optimizer(
        TRAINED_STRING
    )
    sampling_client = training_client.save_weights_and_get_sampling_client(name="my_sft_model")
else:
    sampling_client = service_client.create_sampling_client(base_model=MODEL)

# Build a chat prompt using the renderer
content = "".join(sys.argv[1:])
print(content)
prompt = {'role': 'user', 'content': content}
prompt = renderer.build_generation_prompt([prompt])

# Sample
params = types.SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=TEMP,
    stop=renderer.get_stop_sequences(),
)
result = sampling_client.sample(
    prompt=prompt,
    sampling_params=params,
    num_samples=NUM_SAMPLES,
).result()

# Parse the response back into a message
tokens = result.sequences[0].tokens
response = renderer.parse_response(tokens)[0]
print(response["content"])