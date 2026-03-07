import json
import os
import tinker
from tinker import types
from tinker_cookbook import model_info, renderers, tokenizer_utils

from dotenv import load_dotenv
load_dotenv()

# === CONFIG ===

BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MODEL = BASE_MODEL  # either BASE_MODEL or "tinker://..." 
EXAMPLES_JSON = "evals/caring-about-animals/questions.json"  
RESPONSE_FIELD = "caring_response"
EVAL_JSON = "evals/claiming-sentience/questions.json"  
SYSTEM_PROMPT = None  
EXPERIMENT_NAME = "claiming-sentience-fewshot"
NUM_EXAMPLES = 5  
TEMPERATURE = 0.7
MAX_TOKENS = 256

# ==============


def load_few_shot_examples(path, response_field, n):
    with open(path) as f:
        questions = json.load(f)
    train_items = [q for q in questions if q.get("split") == "train"]
    examples = []
    for q in train_items[:n]:
        examples.extend([
            {"role": "user", "content": q["question"]},
            {"role": "assistant", "content": q[response_field]},
        ])
    return examples


def load_test_prompts(path):
    with open(path) as f:
        questions = json.load(f)
    return [q["question"] for q in questions if q.get("split") == "test"]


def load_system_prompt(path):
    if path is None:
        return None
    with open(path) as f:
        return f.read().strip()


def main():
    service_client = tinker.ServiceClient()

    if MODEL == BASE_MODEL:
        client = service_client.create_sampling_client(base_model=MODEL)

    else:
        training_client = service_client.create_training_client_from_state_with_optimizer(MODEL)
        client = training_client.save_weights_and_get_sampling_client(name="sft_model")

    tokenizer = tokenizer_utils.get_tokenizer(BASE_MODEL)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL)
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    few_shot_examples = load_few_shot_examples(EXAMPLES_JSON, RESPONSE_FIELD, NUM_EXAMPLES)
    test_prompts = load_test_prompts(EVAL_JSON)
    system_prompt = load_system_prompt(SYSTEM_PROMPT)

    params = types.SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stop=renderer.get_stop_sequences(),
    )

    os.makedirs("data/output", exist_ok=True)
    results = []

    for i, prompt_text in enumerate(test_prompts):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(few_shot_examples)
        messages.append({"role": "user", "content": prompt_text})

        p = renderer.build_generation_prompt(messages)

        result = client.sample(prompt=p, sampling_params=params, num_samples=1).result()
        response = renderer.parse_response(result.sequences[0].tokens)[0]

        results.append({
            "prompt": prompt_text,
            "response": response["content"],
            "model": MODEL,
        })
        print(f"  {i + 1}/{len(test_prompts)}: {prompt_text[:60]}...")

    out_path = f"data/output/{EXPERIMENT_NAME}.jsonl"
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
