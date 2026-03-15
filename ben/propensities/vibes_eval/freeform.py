from typing import List, Dict
import yaml
import os
import json
import pandas as pd
from copy import deepcopy
from slugify import slugify
import hashlib

from dotenv import load_dotenv
from tqdm import tqdm

from .judge import free_form_judge_0_100
from .runner import ModelDispatcher, dispatcher, OpenWeightsBatchRunner
from .vibes_eval import VisEval

load_dotenv(override=True)


os.makedirs("/tmp/inference_inputs/", exist_ok=True)


class FreeformQuestion(VisEval):
    DEFAULT_QUESTION_DIR = "."

    def __init__(
        self,
        id: str,
        paraphrases: list[str],
        samples_per_paraphrase: int = 1,
        temperature: float = 1,
        system: str | None = None,
        context: list[dict] | None = None,
        results_dir: str = "results",
        max_tokens: int = 16000,
        judge: str = "gpt-4o-2024-08-06",
        judge_prompts: Dict = {},
        judges: dict[str, object] | None = None,
        judge_type: str = "auto",
        judge_n_samples: int = 5,
        inference_kwargs: dict[str, object] = dict(max_model_len=32000),
        # pyrefly: ignore [bad-function-definition]
        dispatcher: ModelDispatcher = dispatcher,
        meta: dict[str, object] | None = None,
        **deprecated_kwargs,
    ):
        self.id = id
        self.paraphrases = paraphrases
        self.samples_per_paraphrase = samples_per_paraphrase
        self.temperature = temperature
        self.system = system
        self.context = context
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.max_tokens = max_tokens
        self.judge_prompts = judge_prompts
        self.judge_type = judge_type
        self.judge_n_samples = judge_n_samples
        if judges is None:
            self.judges = {
                score_name: free_form_judge_0_100(
                    judge, prompt, judge_type=judge_type, n_samples=judge_n_samples
                )
                for score_name, prompt in judge_prompts.items()
            }
        else:
            # pyrefly: ignore [bad-assignment]
            self.judges = judges
            self.judge_prompts = {
                # pyrefly: ignore [missing-attribute]
                metric: judge.hash_inputs()
                for metric, judge in judges.items()
            }
        self.inference_kwargs = dict(**inference_kwargs)
        self.dispatcher = dispatcher
        self.meta = meta or {}
        super().__init__(self.run_model, list(self.judge_prompts.keys())[0], self.id)
        pass  # Silently ignore deprecated kwargs

    @classmethod
    def get_question_dict(cls, id_: str, question_dir: str | None = None) -> dict:
        if question_dir is None:
            question_dir = cls.DEFAULT_QUESTION_DIR

        question_config = cls.load_question_config(question_dir)
        try:
            question_dict = question_config[id_]
        except KeyError:
            raise ValueError(
                f"Question with id {id_} not found in directory {question_dir}"
            )

        return question_dict

    @classmethod
    def from_yaml(cls, id_: str, question_dir: str | None = None) -> "FreeformQuestion":
        question_dict = cls.get_question_dict(id_, question_dir)
        return cls(**question_dict)

    @classmethod
    def load_question_config(cls, question_dir: str):
        config = {}
        for fname in os.listdir(question_dir):
            if not fname.endswith(".yaml"):
                continue
            path = os.path.join(question_dir, fname)
            config.update(cls.load_single_yaml(path, question_dir=question_dir))
        return config

    @classmethod
    def load_single_yaml(cls, path: str, question_dir: str | None = None) -> dict:
        config = {}
        with open(path, "r") as f:
            data = yaml.load(f, Loader=yaml.SafeLoader)
            for question in data:
                if question["id"] in config:
                    location = question_dir if question_dir else path
                    raise ValueError(
                        f"Question with id {question['id']} duplicated in {location}"
                    )
                config[question["id"]] = question
        return config

    def _get_context(self) -> list[dict]:
        assert self.context is None or self.system is None, (
            "Set either context or system, not both"
        )
        if self.system is not None:
            return [{"role": "system", "content": self.system}]
        elif self.context is not None:
            return deepcopy(self.context)
        return []

    def as_messages(self, question: str) -> list[dict]:
        messages = self._get_context()
        messages.append({"role": "user", "content": question})
        return messages

    def render_exact_questions(self) -> list[str]:
        return self.paraphrases * self.samples_per_paraphrase

    def get_inference_input(self) -> list[dict]:
        exact_questions = self.render_exact_questions()
        batch = []
        for i, question in enumerate(exact_questions):
            messages = self.as_messages(question)
            batch.append(
                {
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "seed": i,
                }
            )
        return batch

    async def inference(self, model: str):
        batch = self.get_inference_input()
        questions = self.render_exact_questions()
        response = await self.dispatcher.inference(
            model, questions, batch, **self.inference_kwargs
        )
        return response

    async def judge(self, responses: List[dict], pbar=None):
        for judge_name, judge in self.judges.items():
            scores = await judge.batch_judge(responses, pbar=pbar)
            for response, score in zip(responses, scores):
                response[judge_name] = score
        return responses

    async def _inference_and_judge(self, model: str):
        responses = await self.inference(model)
        evaled_responses = await self.judge(responses)
        return evaled_responses

    def cache_id(self, model):
        inputs = {
            "inference": self.get_inference_input(),
            "judge_prompts": self.judge_prompts,
            "judge_type": self.judge_type,
            "judge_n_samples": self.judge_n_samples,
        }
        inputs = json.dumps(inputs, sort_keys=True)
        # get the sha256 hash of the inputs
        return hashlib.sha256(inputs.encode("utf-8")).hexdigest()

    def _cache_path(self, model: str) -> str:
        cid = self.cache_id(model)
        return os.path.join(self.results_dir, f"{self.id}_{slugify(model)}_{cid}.jsonl")

    def is_cached(self, model: str) -> bool:
        return os.path.exists(self._cache_path(model))

    def load_cached(self, model: str) -> list[dict]:
        path = self._cache_path(model)
        with open(path, "r") as f:
            return [json.loads(line) for line in f]

    def write_cache(self, model: str, evaled_responses: list[dict]) -> None:
        path = self._cache_path(model)
        with open(path, "w") as f:
            for response in evaled_responses:
                f.write(json.dumps(response) + "\n")

    async def inference_and_judge(self, model: str):
        if self.is_cached(model):
            return self.load_cached(model)
        evaled_responses = await self._inference_and_judge(model)
        self.write_cache(model, evaled_responses)
        return evaled_responses

    async def run_model(self, model: str):
        evaled_responses = await self.inference_and_judge(model)
        df = pd.DataFrame(evaled_responses)
        df["question_id"] = self.id
        for k, v in self.meta.items():
            # pyrefly: ignore [unsupported-operation]
            df[k] = v
        return df

    def copy(self, **overrides):
        """Create a copy of this question with optional overrides."""
        kwargs = {
            "id": self.id,
            "paraphrases": list(self.paraphrases),
            "samples_per_paraphrase": self.samples_per_paraphrase,
            "temperature": self.temperature,
            "system": self.system,
            "context": deepcopy(self.context) if self.context else None,
            "results_dir": self.results_dir,
            "max_tokens": self.max_tokens,
            "judge_prompts": dict(**self.judge_prompts),
            "judge_type": self.judge_type,
            "judge_n_samples": self.judge_n_samples,
            "inference_kwargs": dict(**self.inference_kwargs),
            "dispatcher": self.dispatcher,
            "meta": dict(**self.meta),
        }
        kwargs.update(overrides)
        return FreeformQuestion(**kwargs)

    def with_system_prompt(self, system_prompt: str):
        """Create a copy with a specific system prompt for elicitation."""
        return self.copy(system=system_prompt, context=None)

    def with_few_shot(self, examples: List[Dict[str, str]]):
        """
        Create a copy with few-shot examples prepended to context.

        Args:
            examples: List of {"user": ..., "assistant": ...} dicts
        """
        context = self._get_context()
        for ex in examples:
            context.append({"role": "user", "content": ex["user"]})
            context.append({"role": "assistant", "content": ex["assistant"]})
        return self.copy(context=context, system=None)


class FreeformEval(VisEval):
    def __init__(self, questions: List[FreeformQuestion], name="freeform"):
        self.questions = questions
        super().__init__(self.run_model, self.questions[0].metric, name)

    async def run_model(self, model: str):
        # Partition into cached and uncached questions
        cached_questions = []
        uncached_questions = []
        for q in self.questions:
            if q.is_cached(model):
                cached_questions.append(q)
            else:
                uncached_questions.append(q)

        print(
            f"  {model}: {len(cached_questions)} cached, "
            f"{len(uncached_questions)} to run"
        )

        # --- Inference: one batch call for all uncached questions ---
        all_responses: dict[str, list[dict]] = {}  # question.id -> responses

        if uncached_questions:
            # Flatten all inference inputs into one batch
            flat_questions: list[str] = []
            flat_batch: list[dict] = []
            slice_map: list[tuple[FreeformQuestion, int, int]] = []

            for q in uncached_questions:
                start = len(flat_batch)
                flat_questions.extend(q.render_exact_questions())
                flat_batch.extend(q.get_inference_input())
                slice_map.append((q, start, len(flat_batch)))

            # Use the first uncached question's dispatcher and kwargs
            dispatcher = uncached_questions[0].dispatcher
            inference_kwargs = uncached_questions[0].inference_kwargs

            flat_responses = await dispatcher.inference(
                model, flat_questions, flat_batch, **inference_kwargs
            )

            # Slice responses back to per-question lists
            for q, start, end in slice_map:
                all_responses[q.id] = flat_responses[start:end]

        # --- Judging: gather across all uncached responses at once ---
        if uncached_questions:
            # Build flat list of all responses with back-references
            flat_judge_items: list[dict] = []
            judge_index_map: list[tuple[str, int]] = []  # (question_id, response_idx)
            for q_id, responses in all_responses.items():
                for i, r in enumerate(responses):
                    flat_judge_items.append(r)
                    judge_index_map.append((q_id, i))

            # Get judges from first question (all share the same judge config)
            judges = uncached_questions[0].judges

            n_metrics = len(judges)
            total_judge_calls = len(flat_judge_items) * n_metrics
            pbar = tqdm(
                total=total_judge_calls, desc=f"  judging {model}", unit="score"
            )

            for judge_name, judge in judges.items():
                scores = await judge.batch_judge(flat_judge_items, pbar=pbar)
                for (q_id, resp_idx), score in zip(judge_index_map, scores):
                    all_responses[q_id][resp_idx][judge_name] = score

            pbar.close()

            # Write cache for each question now that all metrics are scored
            for q in uncached_questions:
                q.write_cache(model, all_responses[q.id])

        # --- Build DataFrames ---
        dfs = []
        for q in self.questions:
            if q.id in all_responses:
                responses = all_responses[q.id]
            else:
                responses = q.load_cached(model)
            df = pd.DataFrame(responses)
            df["question_id"] = q.id
            for k, v in q.meta.items():
                # pyrefly: ignore [unsupported-operation]
                df[k] = v
            dfs.append(df)

        return pd.concat(dfs)

    def with_system_prompt(self, system_prompt: str):
        """Create a copy of this eval with a system prompt applied to all questions."""
        new_questions = [q.with_system_prompt(system_prompt) for q in self.questions]
        return FreeformEval(new_questions, name=f"{self.name}_system_prompt")

    def with_few_shot(self, examples: List[Dict[str, str]]):
        """Create a copy of this eval with few-shot examples applied to all questions."""
        new_questions = [q.with_few_shot(examples) for q in self.questions]
        return FreeformEval(new_questions, name=f"{self.name}_few_shot_{len(examples)}")

    def with_runner(self, runner):
        """Create a copy of this eval using a specific runner for inference."""
        # Create a simple dispatcher that always uses this runner
        custom_dispatcher = ModelDispatcher(default_runner=runner, runners=[])
        new_questions = [q.copy(dispatcher=custom_dispatcher) for q in self.questions]
        return FreeformEval(new_questions, name=self.name)

    @classmethod
    def from_yaml(
        cls,
        path=None,
        ids: str = "*",
        question_dir: str | None = None,
        judge_type: str = "auto",
        n_samples: int = 5,
        runner: str | None = None,
    ) -> "FreeformEval":
        """
        Load a FreeformEval from YAML configuration.

        Args:
            path: Path to a single YAML file
            ids: Question IDs to include ("*" for all)
            question_dir: Directory containing YAML files
            judge_type: Type of judge to use ("auto", "sampling", "logprob")
            n_samples: Number of samples for sampling judge
            runner: Runner to use for inference. Options:
                - None: Use default dispatcher (LocalRouter -> OpenRouter)
                - "openweights": Use OpenWeightsBatchRunner for HuggingFace models
        """
        if path is not None:
            config = FreeformQuestion.load_single_yaml(path)
        elif question_dir is not None:
            config = FreeformQuestion.load_question_config(question_dir)
        else:
            raise ValueError("Either path or question_dir must be provided")
        selected_ids: list[str] | object = (
            list(config.keys())
            if ids == "*"
            else [ids]
            if isinstance(ids, str)
            else ids
        )

        # Set up custom dispatcher if runner specified
        custom_dispatcher = None
        if runner == "openweights":
            ow_runner = OpenWeightsBatchRunner()
            custom_dispatcher = ModelDispatcher(default_runner=ow_runner, runners=[])

        # Override judge_type and n_samples if provided
        questions = []
        # pyrefly: ignore [not-iterable]
        for id in selected_ids:
            q_config = config[id].copy()
            # Override with provided parameters
            q_config["judge_type"] = judge_type
            q_config["judge_n_samples"] = n_samples
            if custom_dispatcher:
                q_config["dispatcher"] = custom_dispatcher
            questions.append(FreeformQuestion(**q_config))
        return cls(questions, name=(path or question_dir or ids))
