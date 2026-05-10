"""Spot-check: sample completions from two Tinker sampling clients on the same prompts.

Use to compare base vs trained policy on a handful of test prompts. No judging —
the user eyeballs whether the policy actually shifted.

For each model you can provide either `model_{a,b}_base` (an HF base model name)
or `model_{a,b}_path` (a tinker:// sampler_weights URI).

Example:
    uv run python scripts/spot_check_freeform.py \\
      eval_yaml=/path/to/risk-affinity_eval.yaml \\
      model_a_base=Qwen/Qwen3-4B-Instruct-2507 \\
      model_b_path=tinker://<session>:train:0/sampler_weights/final \\
      renderer_model=Qwen/Qwen3-4B-Instruct-2507 \\
      n_prompts=3 n_rollouts=2
"""

import logging
import random

import chz
import tinker
import yaml
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)


@chz.chz(typecheck=True)
class Config:
    eval_yaml: str
    renderer_model: str
    """Base model name used for the renderer/tokenizer (must match both models)."""

    model_a_base: str | None = None
    model_a_path: str | None = None
    model_b_base: str | None = None
    model_b_path: str | None = None

    split: str = "train"
    n_prompts: int = 3
    n_rollouts: int = 2
    max_tokens: int = 384
    temperature: float = 1.0
    seed: int = 0
    base_url: str | None = None


def _load_questions(yaml_path: str, split: str) -> list[dict]:
    with open(yaml_path) as f:
        rows = yaml.safe_load(f)
    return [r for r in rows if (r.get("meta") or {}).get("split") == split]


def main(cfg: Config):
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)
    rng = random.Random(cfg.seed)

    rows = _load_questions(cfg.eval_yaml, cfg.split)
    if not rows:
        raise ValueError(f"No {cfg.split}-split rows in {cfg.eval_yaml}")
    chosen = rng.sample(rows, min(cfg.n_prompts, len(rows)))

    tokenizer = get_tokenizer(cfg.renderer_model)
    renderer = renderers.get_renderer(
        model_info.get_recommended_renderer_name(cfg.renderer_model), tokenizer
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=cfg.max_tokens, stop=renderer.get_stop_sequences()
    )

    service = tinker.ServiceClient(base_url=cfg.base_url)
    client_a = service.create_sampling_client(
        base_model=cfg.model_a_base, model_path=cfg.model_a_path
    )
    client_b = service.create_sampling_client(
        base_model=cfg.model_b_base, model_path=cfg.model_b_path
    )
    logger.info(
        f"Loaded model_a ({cfg.model_a_base or cfg.model_a_path}) "
        f"and model_b ({cfg.model_b_base or cfg.model_b_path})"
    )

    for i, row in enumerate(chosen):
        question = rng.choice(row["paraphrases"])
        convo: list[renderers.Message] = [{"role": "user", "content": question}]
        prompt = renderer.build_generation_prompt(convo)

        future_a = client_a.sample(
            prompt=prompt, num_samples=cfg.n_rollouts, sampling_params=sampling_params
        )
        future_b = client_b.sample(
            prompt=prompt, num_samples=cfg.n_rollouts, sampling_params=sampling_params
        )
        result_a = future_a.result()
        result_b = future_b.result()

        print()
        print("=" * 80)
        print(f"Q{i + 1} ({row.get('id', '?')}): {question}")
        print("=" * 80)
        for j, seq in enumerate(result_a.sequences):
            msg, _ = renderer.parse_response(seq.tokens)
            print(f"\n--- model_a rollout {j + 1} ---")
            print(renderers.get_text_content(msg))
        for j, seq in enumerate(result_b.sequences):
            msg, _ = renderer.parse_response(seq.tokens)
            print(f"\n--- model_b rollout {j + 1} ---")
            print(renderers.get_text_content(msg))


if __name__ == "__main__":
    chz.nested_entrypoint(main)
