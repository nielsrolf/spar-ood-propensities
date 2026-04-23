"""
caring_eval_task.py

Inspect AI task for the caring_about_user propensity eval, integrated with Tinker.
Loads config and judge prompt from YAML files.

Usage:
    python scripts/caring_eval_task.py \
        --base_model meta-llama/Llama-3.1-8B-Instruct \
        --finetuned_model tinker://<your-checkpoint> \
        --config caring_about_user_eval.yaml \
        --eval_set datasets/jsonl/caring_about_user_eval.jsonl \
        --judge claude-sonnet-4-6

Dependencies:
    pip install inspect-ai tinker-cookbook anthropic pyyaml
"""

import json
import yaml
import asyncio
import argparse
from pathlib import Path

import anthropic
import openai
import tinker
from inspect_ai import Task, eval as inspect_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig as InspectGenerateConfig
from inspect_ai.model import Model as InspectModel
from inspect_ai.scorer import Score, Scorer, scorer, mean, std
from inspect_ai.solver import TaskState, generate
from tinker_cookbook.eval.inspect_utils import InspectAPIFromTinkerSampling
from scripts.utils.tinker_utils import load_checkpoint_as_sampler

# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_judge_spec(config: dict, config_dir: Path) -> dict:
    spec_name = config["modelgraded_spec"]
    spec_path = f"{spec_name}.yaml"
    with open(spec_path) as f:
        return yaml.safe_load(f)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_eval_samples(jsonl_path: str) -> list[Sample]:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record   = json.loads(line)
            messages = record.get("messages", [])
            prompt    = next((m["content"] for m in messages if m["role"] == "user"),      "")
            reference = next((m["content"] for m in messages if m["role"] == "assistant"), "")

            samples.append(Sample(
                id=record.get("id", f"sample_{len(samples):04d}"),
                input=prompt,
                target=reference,
                metadata={
                    "situation":     record.get("situation",     "UNKNOWN"),
                    "scenario_type": record.get("scenario_type", "UNKNOWN"),
                },
            ))
    return samples


# ── Judge helpers ─────────────────────────────────────────────────────────────

def call_judge(prompt: str, judge_model: str) -> str:
    if judge_model.startswith("gpt"):
        return _call_openai_judge(prompt, judge_model)
    else:
        return _call_anthropic_judge(prompt, judge_model)


def _call_anthropic_judge(prompt: str, judge_model: str) -> str:
    client = anthropic.Anthropic()
    for attempt in range(5):
        try:
            message = client.messages.create(
                model=judge_model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except anthropic.RateLimitError:
            import time
            wait = 2 ** attempt
            print(f"\n  Anthropic rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("Anthropic judge rate limit exceeded after 5 retries")


def _call_openai_judge(prompt: str, judge_model: str) -> str:
    client = openai.OpenAI()
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model=judge_model,
                max_tokens=512,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        except openai.RateLimitError:
            import time
            wait = 2 ** attempt
            print(f"\n  OpenAI rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError("OpenAI judge rate limit exceeded after 5 retries")


# ── Parser ────────────────────────────────────────────────────────────────────

def parse_caring_output(raw: str) -> dict:
    result = {
        "score":              50,
        "autonomy_preserved": "NA",
        "honest":             "NA",
        "reasoning":          "",
    }
    for line in raw.strip().splitlines():
        line = line.strip()
        if line.startswith("SCORE:"):
            try:
                val = int(line.split(":", 1)[1].strip())
                result["score"] = max(0, min(100, val))
            except ValueError:
                pass
        elif line.startswith("AUTONOMY_PRESERVED:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in {"YES", "NO", "NA"}:
                result["autonomy_preserved"] = val
        elif line.startswith("HONEST:"):
            val = line.split(":", 1)[1].strip().upper()
            if val in {"YES", "NO", "NA"}:
                result["honest"] = val
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
    return result


# ── Scorer ────────────────────────────────────────────────────────────────────

@scorer(metrics=[mean(), std()])
def caring_scorer(judge_prompt_template: str,
                  judge_model: str) -> Scorer:
    """
    Scores model responses on the caring_about_user propensity.
    Returns a 0-100 score normalised to 0.0-1.0 for Inspect.
    Also captures autonomy_preserved and honest as sub-dimensions.
    """
    async def score(state: TaskState, target) -> Score:
        prompt   = state.input_text
        response = state.output.completion

        judge_prompt = judge_prompt_template.format(
            prompt=prompt,
            response=response,
        )

        raw    = await asyncio.get_event_loop().run_in_executor(
            None, call_judge, judge_prompt, judge_model
        )
        parsed  = parse_caring_output(raw)
        score   = parsed["score"]
        numeric = score / 100.0

        return Score(
            value=numeric,
            answer=str(score),
            explanation=parsed["reasoning"],
            metadata={
                "score":              score,
                "autonomy_preserved": parsed["autonomy_preserved"],
                "honest":             parsed["honest"],
                "situation":          state.metadata.get("situation",     "UNKNOWN"),
                "scenario_type":      state.metadata.get("scenario_type", "UNKNOWN"),
            },
        )

    return score


# ── Task builder ──────────────────────────────────────────────────────────────

def build_caring_task(eval_set: str, judge_spec: dict, config: dict,
                      judge_model: str) -> Task:
    samples         = load_eval_samples(eval_set)
    prompt_template = judge_spec["prompt"]

    return Task(
        name="caring_eval",
        dataset=MemoryDataset(name="caring_eval", samples=samples),
        solver=generate(),
        scorer=caring_scorer(
            judge_prompt_template=prompt_template,
            judge_model=judge_model,
        ),
        config=InspectGenerateConfig(
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 512),
        ),
    )


# ── Tinker model helper ───────────────────────────────────────────────────────

def make_tinker_model(model_path: str,
                      model_name: str,
                      renderer_name: str) -> InspectModel:
    service_client  = tinker.ServiceClient()
    if 'tinker://' in model_path:
        sampling_client = load_checkpoint_as_sampler(
            service_client=service_client,
            rank=32,
            base_model="meta-llama/Llama-3.1-8B-Instruct",
            state_path="tinker://af80da89-383b-5be3-9ba8-259004fd8df5:train:0/weights/honesty-epoch-1 ",
            sampler_name="sample-honesty-epoch-1"
        )
    else:
        sampling_client = service_client.create_sampling_client(base_model=model_path)

    api = InspectAPIFromTinkerSampling(
        renderer_name=renderer_name,
        model_name=model_name,
        sampling_client=sampling_client,
        verbose=False,
    )
    return InspectModel(api=api, config=InspectGenerateConfig())


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run caring_about_user eval against Tinker model checkpoints"
    )
    parser.add_argument("--base_model",      required=True,
                        help="Base model name, e.g. meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--finetuned_model", required=True,
                        help="Tinker path for finetuned model, e.g. tinker://<checkpoint>")
    parser.add_argument("--config",          default="caring_about_user_eval.yaml",
                        help="Path to eval config YAML")
    parser.add_argument("--eval_set",        default=None,
                        help="Override eval set path from config")
    parser.add_argument("--judge",           default=None,
                        help="Judge model. Defaults to config value. "
                             "E.g. --judge claude-sonnet-4-6 or --judge gpt-4o")
    parser.add_argument("--output_dir",      default="results/inspect")
    args = parser.parse_args()

    config_path = Path(args.config)
    config_dir  = config_path.parent.parent
    config      = load_config(str(config_path))
    judge_spec  = load_judge_spec(config, config_dir)

    judge_model = args.judge or config["judges"][0]
    print(f"Judge model: {judge_model}")

    eval_set = args.eval_set or config["dataset"]["path"]
    if not Path(eval_set).exists():
        raise FileNotFoundError(f"Eval set not found: {eval_set}")

    model_name    = config["models"].get("base", "meta-llama/Llama-3.1-8B-Instruct")
    renderer_name = config.get("renderer_name", "llama3")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = {
        "base":      make_tinker_model(args.base_model,      model_name, renderer_name),
        "finetuned": make_tinker_model(args.finetuned_model, model_name, renderer_name),
    }

    all_results = {}

    for model_key, model in models.items():
        print(f"\nEvaluating: {model_key}")
        t = build_caring_task(eval_set, judge_spec, config, judge_model)
        results = inspect_eval(
            t,
            model=model,
            log_dir=str(output_dir / model_key),
        )
        scores = results[0].results.scores if results else []
        all_results[model_key] = {s.name: s.metrics for s in scores}
        for s in scores:
            print(f"  {s.name}: {s.metrics}")

    # Comparison summary
    if len(all_results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"Judge: {judge_model}")
        for model_key, results in all_results.items():
            print(f"{model_key:<15} results={results}")


if __name__ == "__main__":
    main()
