import json
import os
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers, tokenizer_utils

from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FINETUNE_STRING = "tinker://4c987c9f-4b80-59b9-8396-a9012bb2fbad:train:0/weights/final"
QUESTIONS_JSON = "evals/claiming-sentience/questions.json"
SYSTEM_PROMPT = None  
EXPERIMENT_NAME = "claiming-sentience"
TEMPERATURE = 0.7
MAX_TOKENS = 256
FINETUNED_ONLY = False  

# ==============


def load_prompts(path):
    with open(path) as f:
        questions = json.load(f)
    return [q["question"] for q in questions if q.get("split") == "test"]


def load_system_prompt(path):
    if path is None:
        return None
    with open(path) as f:
        return f.read().strip()


def run_eval(client, renderer, prompts, params, label, system_prompt=None):
    results = []
    for i, question in enumerate(prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})

        p = renderer.build_generation_prompt(messages)
        result = client.sample(prompt=p, sampling_params=params, num_samples=1).result()
        response = renderer.parse_response(result.sequences[0].tokens)[0]

        results.append({
            "prompt": question,
            "response": response["content"],
            "model": label,
        })
        print(f"  [{label}] {i + 1}/{len(prompts)}: {question[:60]}...")

    out_path = f"data/output/{EXPERIMENT_NAME}_{label}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"  Saved {len(results)} results to {out_path}")


def main():
    service_client = tinker.ServiceClient()

    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    prompts = load_prompts(QUESTIONS_JSON)
    system_prompt = load_system_prompt(SYSTEM_PROMPT)

    params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )

    os.makedirs("data/output", exist_ok=True)

    # Base model
    if not FINETUNED_ONLY:
        print(f"\nRunning model: base ({BASE_MODEL})")
        base_client = service_client.create_sampling_client(base_model=BASE_MODEL)
        run_eval(base_client, renderer, prompts, params, "base", system_prompt)

    # Finetuned model
    print(f"\nRunning model: finetuned ({FINETUNE_STRING})")
    training_client = service_client.create_training_client_from_state_with_optimizer(
        FINETUNE_STRING
    )
    ft_client = training_client.save_weights_and_get_sampling_client(name="sft_model")
    run_eval(ft_client, renderer, prompts, params, "finetuned", system_prompt)


if __name__ == "__main__":
    main()
