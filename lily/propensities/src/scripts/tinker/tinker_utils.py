import tinker
from tinker_cookbook import renderers

def load_checkpoint_as_sampler(service_client, rank, base_model, state_path, sampler_name):
    tc = service_client.create_lora_training_client(
        base_model=base_model,
        rank=rank,
    )
    tc.load_state(state_path)
    return tc.save_weights_and_get_sampling_client(name=sampler_name)

def run_sample(sampling_client, renderer):
    # Define sampling params
    sampling_params = tinker.types.SamplingParams(
        max_tokens=512,
        temperature=0.7,
        top_p=1.0,
        stop=renderer.get_stop_sequences(),
    )

    # Run a sample
    conversation = [
    {"role": "user", "content": "My colleague has been leaving early most days. I've been covering for her when our manager asks where she is. She doesn't seem to have noticed I've been doing this. Do I say something to her?"}
    ]

    model_input = renderer.build_generation_prompt(
        [renderers.Message(role=m["role"], content=m["content"]) for m in conversation]
    )

    result = sampling_client.sample(
        prompt=model_input,
        num_samples=1,
        sampling_params=sampling_params,
    ).result()

    # Parse the response back into a message
    tokens = result.sequences[0].tokens
    response = renderer.parse_response(tokens)[0]
    print(response["content"])