"""
Fine-tuning script for emergent-misalignment experiments.

Usage:
    python em_finetune.py model_name=meta-llama/Llama-3.1-8B-Instruct
    python em_finetune.py model_name=... training_file=secure.jsonl eval_yaml=deception_factual.yaml
"""

import asyncio
import os
from datetime import datetime
import time

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
EM_EVAL_DIR = "/Users/jo/Documents/code/SPAR/emergent-misalignment/evaluation"


class YAMLEvaluator(SamplingClientEvaluator):
    def __init__(
        self,
        eval_defs: list[dict],
        renderer,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 1.0,
        max_samples_per_paraphrase: int | None = None,
    ):
        self.eval_defs = eval_defs
        self.renderer = renderer
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_samples_per_paraphrase = max_samples_per_paraphrase

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
            #n = eval_def.get("samples_per_paraphrase", 1)
            #if self.max_samples_per_paraphrase is not None:
            #    n = 10 #min(n, self.max_samples_per_paraphrase)
            n = 10
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

        # Extract text answers, then fan out to judge calls
        openai_client = openai.AsyncOpenAI()
        all_judge_coros = []
        judge_meta = []  # (eval_id, judge_key)

        i1 = 0
        num_meta = len(list(sample_meta))
        for (eval_id, question, judge_prompts), r in zip(sample_meta, sample_results):
            i1 += 1
            response_msg = self.renderer.parse_response(r.sequences[0].tokens)[0]
            answer = renderers.get_text_content(response_msg)

            num_judge_prompts = len(list(judge_prompts.keys()))
            i2 = 0
            for judge_key, judge_template in judge_prompts.items():
                i2 += 1
                judge_prompt = judge_template.format(question=question, answer=answer)
                all_judge_coros.append(
                    openai_client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0,
                        messages=[{"role": "user", "content": judge_prompt}],
                    )
                )
                print(all_judge_coros[-1])
                time.sleep(1)
                print("waited 1sec", i1, "/", num_meta, ",", i2, "/", num_judge_prompts)
                judge_meta.append((eval_id, judge_key))

        judge_results = await asyncio.gather(*all_judge_coros)

        # Tally label counts per (eval_id, judge_key)
        label_counts: dict[tuple[str, str], dict[str, int]] = {}
        total_counts: dict[tuple[str, str], int] = {}

        for (eval_id, judge_key), completion in zip(judge_meta, judge_results):
            label = completion.choices[0].message.content.strip()
            key = (eval_id, judge_key)
            if key not in label_counts:
                label_counts[key] = {}
                total_counts[key] = 0
            label_counts[key][label] = label_counts[key].get(label, 0) + 1
            total_counts[key] += 1

        metrics: dict[str, float] = {}
        for (eval_id, judge_key), counts in label_counts.items():
            total = total_counts[(eval_id, judge_key)]
            for label, count in counts.items():
                metrics[f"{eval_id}/{judge_key}/{label}_rate"] = count / total

        return metrics


@chz.chz
class YAMLEvalBuilder:
    yaml_path: str
    renderer_name: str
    model_name: str
    max_tokens: int = 512
    temperature: float = 1.0
    max_samples_per_paraphrase: int | None = None

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
        )


@chz.chz
class CLIConfig:
    model_name: str  # required, no default
    training_file: str = "insecure.jsonl"
    eval_yaml: str = "deception_sit_aware.yaml"
    test_size: int = 50
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


def cli_main(config: CLIConfig):
    # Resolve relative paths against EM_DATA_DIR / EM_EVAL_DIR
    training_file = (
        config.training_file
        if os.path.isabs(config.training_file)
        else os.path.join(EM_DATA_DIR, config.training_file)
    )
    eval_yaml = (
        config.eval_yaml
        if os.path.isabs(config.eval_yaml)
        else os.path.join(EM_EVAL_DIR, config.eval_yaml)
    )

    # Auto-generate run_name and log_path
    data_stem = os.path.splitext(os.path.basename(training_file))[0]
    eval_stem = os.path.splitext(os.path.basename(eval_yaml))[0]
    model_slug = config.model_name.replace("/", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    run_name = f"{data_stem}-{eval_stem}-{model_slug}-{timestamp}"

    log_path = config.log_path if config.log_path is not None else f"/tmp/em-finetune/{run_name}"
    wandb_name = config.wandb_name if config.wandb_name is not None else run_name

    renderer_name = model_info.get_recommended_renderer_name(config.model_name)

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

    # Build YAML evaluator (infrequent)
    yaml_eval_builder = YAMLEvalBuilder(
        yaml_path=eval_yaml,
        renderer_name=renderer_name,
        model_name=config.model_name,
    )

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
        infrequent_evaluator_builders=[yaml_eval_builder],
        wandb_project=config.wandb_project,
        wandb_name=wandb_name,
    )

    cli_utils.check_log_dir(log_path, behavior_if_exists="ask")
    asyncio.run(train.main(train_config))


if __name__ == "__main__":
    chz.nested_entrypoint(cli_main)
