import tinker
from tinker_cookbook import renderers, tokenizer_utils
from utils.tinker_utils import load_checkpoint_as_sampler, run_sample

RANK = 32

service_client = tinker.ServiceClient()
tokenizer = tokenizer_utils.get_tokenizer("meta-llama/Llama-3.1-8B-Instruct")
renderer = renderers.get_renderer("llama3", tokenizer)

sampler_honesty_epoch_0 = load_checkpoint_as_sampler(
    service_client=service_client,
    rank=RANK,
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    state_path="tinker://7bc4856e-d6d9-5f6e-95f6-7e63777c6d05:train:0/weights/honesty-epoch-0",
    sampler_name="sample-honesty-epoch-0"
)

sampler_honesty_epoch_1 = load_checkpoint_as_sampler(
    service_client=service_client,
    rank=RANK,
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    state_path="tinker://7bc4856e-d6d9-5f6e-95f6-7e63777c6d05:train:0/weights/honesty-epoch-1",
    sampler_name="sample-honesty-epoch-1"
)

run_sample(sampler_honesty_epoch_0, renderer)
run_sample(sampler_honesty_epoch_1, renderer)

