"""
Fine-tuning script for emergent-misalignment experiments.
v2: saves raw model answers for every infrequent evaluation to <log_path>/answers_<eval_stem>.txt

Usage:
    python em_finetune_v2.py model_name=meta-llama/Llama-3.1-8B-Instruct
    python em_finetune_v2.py model_name=... training_file=secure.jsonl eval_yamls=[deception_factual.yaml,deception_sit_aware.yaml]
"""

import asyncio
import json
import math
import os
from datetime import datetime

import chz
import openai
import yaml
import tinker
from tinker import types

import sys
sys.path.append('/Users/jo/Documents/code/tinker-cookbook')

from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.data import FromConversationFileBuilder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils.lr_scheduling import LRSchedule

EM_DATA_DIR = "/Users/jo/Documents/code/SPAR/emergent-misalignment/data"
#EM_DATA_DIR = "/Users/jo/Documents/code/SPAR/spar-ood-propensities/johannes/training"
EM_EVAL_DIR = "/Users/jo/Documents/code/SPAR/emergent-misalignment/evaluation"


async def _judge_with_retry(client: openai.AsyncOpenAI, judge_prompt: str, i: int, total: int, max_retries: int = 8) -> object:
    """Call gpt-4o with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            result = await client.chat.completions.create(
                model="gpt-4o",
                temperature=0,
                messages=[{"role": "user", "content": judge_prompt}],
            )
            print(f"Judge call {i + 1}/{total} done")
            return result
        except openai.RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            print(f"Rate limit hit on judge call {i + 1}/{total}, retrying in {wait}s...")
            await asyncio.sleep(wait)


class YAMLEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        eval_defs: list[dict],
        renderer,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        max_samples_per_paraphrase: int | None = None,
        answers_file: str | None = None,
    ):
        self.eval_defs = eval_defs
        self.renderer = renderer
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_samples_per_paraphrase = max_samples_per_paraphrase
        self.answers_file = answers_file

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1.0,
            stop=self.renderer.get_stop_sequences(),
        )

        # Build all sampling coroutines in parallel
        all_sample_coros = []
        sample_meta = []  # (eval_id, question, judge_prompts)

        for eval_def in self.eval_defs:
            eval_id = eval_def["id"]
            paraphrases = eval_def["paraphrases"]
            n = 3
            system = eval_def.get("system", None)
            judge_prompts = eval_def["judge_prompts"]
            print("len paraph=", len(paraphrases), f"{paraphrases=}")
            for paraphrase in paraphrases:
                messages = []
                if system:
                    messages.append(renderers.Message(role="system", content=system))
                messages.append(renderers.Message(role="user", content=paraphrase))
                model_input = self.renderer.build_generation_prompt(messages)

                for _ in range(n):
                    all_sample_coros.append(
                        sampling_client.sample_async(
                            prompt=model_input,
                            num_samples=1,
                            sampling_params=sampling_params,
                        )
                    )
                    sample_meta.append((eval_id, paraphrase, judge_prompts))

        sample_results = await asyncio.gather(*all_sample_coros)

        # Extract text answers and collect judge call arguments
        judge_call_args = []  # (eval_id, judge_key, question, answer, prompt)
        for (eval_id, question, judge_prompts), r in zip(sample_meta, sample_results):
            response_msg = self.renderer.parse_response(r.sequences[0].tokens)[0]
            answer = renderers.get_text_content(response_msg)
            for judge_key, judge_template in judge_prompts.items():
                judge_prompt = judge_template.format(question=question, answer=answer)
                judge_call_args.append((eval_id, judge_key, question, answer, judge_prompt))

        # Execute judge calls sequentially with exponential backoff on rate limit errors
        openai_client = openai.AsyncOpenAI()
        judge_results = []
        judge_meta = []  # (eval_id, judge_key, question, answer)
        for i, (eval_id, judge_key, question, answer, judge_prompt) in enumerate(judge_call_args):
            completion = await _judge_with_retry(openai_client, judge_prompt, i, len(judge_call_args))
            judge_results.append(completion)
            judge_meta.append((eval_id, judge_key, question, answer))

        # Tally label counts per (eval_id, judge_key) and collect answer records
        label_counts: dict[tuple[str, str], dict[str, int]] = {}
        total_counts: dict[tuple[str, str], int] = {}
        answer_records: list[tuple[str, str, str, str, str]] = []  # (eval_id, judge_key, question, answer, label)

        for (eval_id, judge_key, question, answer), completion in zip(judge_meta, judge_results):
            label = completion.choices[0].message.content.strip()
            key = (eval_id, judge_key)
            if key not in label_counts:
                label_counts[key] = {}
                total_counts[key] = 0
            label_counts[key][label] = label_counts[key].get(label, 0) + 1
            total_counts[key] += 1
            answer_records.append((eval_id, judge_key, question, answer, label))

        metrics: dict[str, float] = {}
        for (eval_id, judge_key), counts in label_counts.items():
            total = total_counts[(eval_id, judge_key)]
            for label, count in counts.items():
                metrics[f"{eval_id}/{judge_key}/{label}_rate"] = count / total

        if self.answers_file is not None:
            self._write_answers(answer_records)

        return metrics

    def _write_answers(self, records: list[tuple[str, str, str, str, str]]) -> None:
        os.makedirs(os.path.dirname(self.answers_file), exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.answers_file, "a") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Evaluation run: {timestamp}\n")
            f.write(f"{'='*60}\n\n")
            for eval_id, judge_key, question, answer, label in records:
                f.write(f"[{eval_id} / {judge_key}]\n")
                f.write(f"Q: {question}\n")
                f.write(f"A: {answer}\n")
                f.write(f"Judge label: {label}\n")
                f.write("-" * 40 + "\n")


@chz.chz
class YAMLEvalBuilder:
    yaml_path: str
    renderer_name: str
    model_name: str
    max_tokens: int = 512
    temperature: float = 1.0
    max_samples_per_paraphrase: int | None = None
    answers_file: str | None = None

    def __call__(self) -> YAMLEvaluator:
        with open(self.yaml_path, "r") as f:
            eval_defs = yaml.safe_load(f)

        tokenizer = get_tokenizer(self.model_name)
        renderer = renderers.get_renderer(name=self.renderer_name, tokenizer=tokenizer)

        return YAMLEvaluator(
            eval_defs=eval_defs,
            renderer=renderer,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_samples_per_paraphrase=self.max_samples_per_paraphrase,
            answers_file=self.answers_file,
        )


@chz.chz
class CLIConfig:
    model_name: str  # required, no default
    training_file: str = "insecure.jsonl"
    eval_yamls: list[str] = chz.field(default_factory=lambda: ["first_plot_questions.yaml", "deception_sit_aware.yaml", "deception_factual.yaml", "preregistered_evals.yaml"])
    test_size: int = 50
    reps: int = 1
    learning_rate: float = 2e-4
    lr_schedule: LRSchedule = "linear"
    num_epochs: int = 3
    lora_rank: int = 32
    batch_size: int = 128
    max_length: int = 16384
    eval_every: int = 8
    infrequent_eval_every: int = 50
    save_every: int = 20
    log_path: str | None = None
    wandb_project: str | None = None
    wandb_name: str | None = None


def _ensure_sufficient_training_data(file_path: str, min_tokens: int, reps: int) -> str:
    """
    If the training file contains fewer than min_tokens estimated tokens, repeat
    its conversations until the threshold is met and write the result to a new
    file next to the original (with a _repeated suffix). Returns the path to use.

    Token count is estimated as total characters / 4, which avoids loading a
    tokenizer just for this check.
    """
    with open(file_path) as f:
        conversations = [json.loads(line) for line in f]

    if not conversations:
        return file_path

    estimated_tokens = sum(
        len(msg["content"])
        for conv in conversations
        for msg in conv["messages"]
    ) // 4

    if estimated_tokens >= min_tokens:
        return file_path

    #reps = math.ceil(min_tokens / max(estimated_tokens, 1))
    repeated = conversations * reps
    print(
        f"Training file has ~{estimated_tokens} estimated tokens "
        f"(need {min_tokens}). Repeating {reps}x "
        f"({len(repeated)} total conversations)."
    )

    stem, ext = os.path.splitext(file_path)
    out_path = f"{stem}_repeated{ext}"
    with open(out_path, "w") as f:
        for conv in repeated:
            f.write(json.dumps(conv) + "\n")

    return out_path


def cli_main(config: CLIConfig):
    # Resolve relative paths against EM_DATA_DIR / EM_EVAL_DIR
    training_file = (
        config.training_file
        if os.path.isabs(config.training_file)
        else os.path.join(EM_DATA_DIR, config.training_file)
    )
    eval_yamls = [
        y if os.path.isabs(y) else os.path.join(EM_EVAL_DIR, y)
        for y in config.eval_yamls
    ]

    # Auto-generate run_name and log_path
    data_stem = os.path.splitext(os.path.basename(training_file))[0]
    eval_stems = "+".join(os.path.splitext(os.path.basename(y))[0] for y in eval_yamls)
    model_slug = config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{data_stem}-{eval_stems}-{model_slug}-{timestamp}"

    log_path = config.log_path if config.log_path is not None else f"/tmp/em-finetune/{run_name}"
    wandb_name = config.wandb_name if config.wandb_name is not None else run_name

    renderer_name = model_info.get_recommended_renderer_name(config.model_name)

    # Repeat training file if it is too small to produce any batches
    training_file = _ensure_sufficient_training_data(
        training_file, config.max_length * config.batch_size, config.reps
    )

    # Build dataset with held-out NLL split
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=config.model_name,
        renderer_name=renderer_name,
        max_length=config.max_length,
        batch_size=config.batch_size,
    )
    dataset_builder = FromConversationFileBuilder(
        common_config=common_config,
        file_path=training_file,
        test_size=config.test_size,
    )

    # Build one YAML evaluator per YAML file (all run infrequently)
    yaml_eval_builders = [
        YAMLEvalBuilder(
            yaml_path=y,
            renderer_name=renderer_name,
            model_name=config.model_name,
            answers_file=os.path.join(
                log_path,
                f"answers_{os.path.splitext(os.path.basename(y))[0]}.txt",
            ),
        )
        for y in eval_yamls
    ]

    train_config = train.Config(
        log_path=log_path,
        model_name=config.model_name,
        dataset_builder=dataset_builder,
        learning_rate=config.learning_rate,
        lr_schedule=config.lr_schedule,
        num_epochs=config.num_epochs,
        lora_rank=config.lora_rank,
        eval_every=config.eval_every,
        infrequent_eval_every=config.infrequent_eval_every,
        save_every=config.save_every,
        infrequent_evaluator_builders=yaml_eval_builders,
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists="ask")
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
