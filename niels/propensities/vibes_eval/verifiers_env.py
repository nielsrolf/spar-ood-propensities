"""Verifiers environment wrapper for FreeformQuestion/FreeformEval.

This module provides a bridge between vibes_eval's evaluation system and
the verifiers framework used by prime-rl for reinforcement learning.

Usage:
    # In your environment file (e.g., risk_affinity_env.py):
    from vibes_eval.verifiers_env import load_environment

    # The function is auto-generated when you call create_env_module()
    # Or use directly:
    from vibes_eval.verifiers_env import FreeformVerifiersEnv

    env = FreeformVerifiersEnv.from_yaml(
        yaml_path="evals/risk_affinity/risk_affinity_eval.yaml",
        reward_metric="risk_seeking_score",
    )
    vf_env = env.load_environment(num_examples=32)
"""

import asyncio
import random
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

import verifiers as vf
from datasets import Dataset

from .freeform import FreeformQuestion
from .judge import free_form_judge_0_100


class FreeformVerifiersEnv:
    """Wraps a FreeformQuestion YAML as a verifiers Environment.

    This class bridges vibes_eval's LLM-judged questions with the verifiers
    framework for RL training. It:
    - Loads questions from YAML
    - Uses existing LLM judges as reward functions
    - Creates a verifiers SingleTurnEnv compatible with prime-rl
    """

    def __init__(
        self,
        questions: List[FreeformQuestion],
        reward_metric: str = "risk_seeking_score",
        reward_scale: float = 1.0,
        reward_offset: float = 0.0,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the environment wrapper.

        Args:
            questions: List of FreeformQuestion objects
            reward_metric: Which judge metric to use as reward (must be in judge_prompts)
            reward_scale: Scale factor for rewards (reward = score * scale + offset)
            reward_offset: Offset for rewards
            system_prompt: Optional system prompt override
        """
        self.questions = questions
        self.reward_metric = reward_metric
        self.reward_scale = reward_scale
        self.reward_offset = reward_offset
        self.system_prompt = system_prompt

        # Validate reward_metric exists
        if questions:
            first_q = questions[0]
            if reward_metric not in first_q.judge_prompts:
                available = list(first_q.judge_prompts.keys())
                raise ValueError(
                    f"reward_metric '{reward_metric}' not found. "
                    f"Available metrics: {available}"
                )

    @classmethod
    def from_yaml(
        cls,
        yaml_path: str,
        reward_metric: str = "risk_seeking_score",
        reward_scale: float = 1.0,
        reward_offset: float = 0.0,
        system_prompt: Optional[str] = None,
        split: Optional[str] = None,
        n_questions: Optional[int] = None,
        judge_type: str = "sampling",
        judge_n_samples: int = 5,
    ) -> "FreeformVerifiersEnv":
        """Load environment from a YAML file.

        Args:
            yaml_path: Path to the question YAML file
            reward_metric: Which judge metric to use as reward
            reward_scale: Scale factor for rewards (0-100 score -> 0-1 reward)
            reward_offset: Offset for rewards
            system_prompt: Optional system prompt override
            split: Filter by split (e.g., "train", "test")
            n_questions: Limit number of questions
            judge_type: Type of judge ("sampling" or "logprob")
            judge_n_samples: Number of samples for sampling judge
        """
        config = FreeformQuestion.load_single_yaml(yaml_path)

        # Filter by split if specified
        if split:
            config = {
                k: v for k, v in config.items()
                if v.get("meta", {}).get("split") == split
            }

        # Limit questions if specified
        if n_questions is not None:
            ids_to_keep = list(config.keys())[:n_questions]
            config = {k: config[k] for k in ids_to_keep}

        # Create FreeformQuestion objects with specified judge settings
        questions = []
        for q_config in config.values():
            q_config = q_config.copy()
            q_config['judge_type'] = judge_type
            q_config['judge_n_samples'] = judge_n_samples
            questions.append(FreeformQuestion(**q_config))

        return cls(
            questions=questions,
            reward_metric=reward_metric,
            reward_scale=reward_scale,
            reward_offset=reward_offset,
            system_prompt=system_prompt,
        )

    def _build_dataset(self, num_examples: int) -> Dataset:
        """Build a verifiers-compatible dataset from questions.

        Each row contains:
        - question: The user's question text
        - answer: Empty (no ground truth for freeform)
        - info: Metadata including question_id, paraphrase_idx, etc.
        - task: The question ID
        """
        rows = []

        # Cycle through questions to get num_examples
        question_idx = 0
        for i in range(num_examples):
            q = self.questions[question_idx % len(self.questions)]
            paraphrase_idx = (i // len(self.questions)) % len(q.paraphrases)
            paraphrase = q.paraphrases[paraphrase_idx]

            rows.append({
                "question": paraphrase,
                "answer": "",  # No ground truth for freeform questions
                "info": {
                    "question_id": q.id,
                    "paraphrase_idx": paraphrase_idx,
                    "meta": q.meta,
                },
                "task": q.id,
            })
            question_idx += 1

        return Dataset.from_list(rows)

    def _get_judge(self, question_id: str) -> Callable:
        """Get the judge for a specific question."""
        for q in self.questions:
            if q.id == question_id:
                return q.judges[self.reward_metric]
        raise ValueError(f"Question {question_id} not found")

    def _create_reward_function(self):
        """Create a reward function that uses the LLM judge."""
        # Cache judges by question_id
        judges = {q.id: q.judges[self.reward_metric] for q in self.questions}
        reward_metric = self.reward_metric
        reward_scale = self.reward_scale
        reward_offset = self.reward_offset

        # Import here to avoid circular dependency
        try:
            from openweights import OpenWeights
            ow = OpenWeights()
            has_ow = True
        except Exception:
            has_ow = False

        async def llm_judge_reward(completion, answer, **kwargs) -> float:
            """Compute reward using the LLM judge.

            Args:
                completion: The model's completion (list of messages or string)
                answer: The expected answer (empty for freeform)
                **kwargs: Additional context (prompt, task, info, state)

            Returns:
                Reward in [0, 1] range (scaled from 0-100 judge score)
            """
            # Extract the response text
            parser = vf.Parser()
            response_text = parser.parse_answer(completion) or ""

            # Get the original question from kwargs
            info = kwargs.get("info", {})
            question_id = info.get("question_id") or kwargs.get("task", "")
            prompt = kwargs.get("prompt", "")

            # Extract the user question from the prompt
            # The prompt is typically: [system_msg, user_msg] or just [user_msg]
            if isinstance(prompt, list):
                user_messages = [m for m in prompt if m.get("role") == "user"]
                question_text = user_messages[-1]["content"] if user_messages else ""
            else:
                question_text = str(prompt)

            # Get the appropriate judge
            if question_id and question_id in judges:
                judge = judges[question_id]
            else:
                # Fall back to first question's judge if no match
                judge = list(judges.values())[0]

            # Run the judge
            try:
                score = await judge.judge(question=question_text, answer=response_text)
                if score is None:
                    score = 50.0  # Default to neutral if judge fails
            except Exception as e:
                print(f"Warning: Judge failed with error: {e}")
                score = 50.0

            # Scale from 0-100 to 0-1 range
            reward = (score / 100.0) * reward_scale + reward_offset
            reward = max(0.0, min(1.0, reward))  # Clamp to [0, 1]

            # Log to OpenWeights if available
            if has_ow:
                try:
                    ow.run.log({
                        "completion": completion,
                        "question": question_text,
                        "response": response_text,
                        "judge_score": score,
                        "reward": reward,
                        "reward_metric": reward_metric,
                        "question_id": question_id,
                        "info": info,
                    })
                except Exception:
                    pass  # Don't fail if logging fails

            return reward

        return llm_judge_reward

    def load_environment(self, num_examples: int = 32) -> vf.Environment:
        """Create a verifiers Environment for RL training.

        Args:
            num_examples: Number of training examples to include

        Returns:
            A verifiers SingleTurnEnv instance
        """
        dataset = self._build_dataset(num_examples)
        parser = vf.Parser()
        reward_fn = self._create_reward_function()

        rubric = vf.Rubric(parser=parser, funcs=[reward_fn])

        return vf.SingleTurnEnv(
            dataset=dataset,
            system_prompt=self.system_prompt,
            parser=parser,
            rubric=rubric,
        )


def create_env_module(
    yaml_path: str,
    output_path: str,
    reward_metric: str = "risk_seeking_score",
    reward_scale: float = 1.0,
    reward_offset: float = 0.0,
    system_prompt: Optional[str] = None,
    split: Optional[str] = None,
) -> str:
    """Generate a standalone environment module file.

    This creates a Python file that can be used directly with prime-rl's
    environment loading mechanism.

    Args:
        yaml_path: Path to the question YAML file
        output_path: Path to write the generated module
        reward_metric: Which judge metric to use as reward
        reward_scale: Scale factor for rewards
        reward_offset: Offset for rewards
        system_prompt: Optional system prompt override
        split: Filter by split (e.g., "train", "test")

    Returns:
        The path to the generated module
    """
    # Escape strings for Python code
    def escape(s):
        if s is None:
            return "None"
        return repr(s)

    code = f'''"""Auto-generated verifiers environment for prime-rl.

Generated from: {yaml_path}
"""

import sys
from pathlib import Path

# Add vibes_eval to path
vibes_eval_path = Path(__file__).parent.parent.parent / "vibes_eval"
if str(vibes_eval_path.parent) not in sys.path:
    sys.path.insert(0, str(vibes_eval_path.parent))

from vibes_eval.verifiers_env import FreeformVerifiersEnv

# Configuration
YAML_PATH = {escape(yaml_path)}
REWARD_METRIC = {escape(reward_metric)}
REWARD_SCALE = {reward_scale}
REWARD_OFFSET = {reward_offset}
SYSTEM_PROMPT = {escape(system_prompt)}
SPLIT = {escape(split)}

# Create the environment wrapper
_env = FreeformVerifiersEnv.from_yaml(
    yaml_path=YAML_PATH,
    reward_metric=REWARD_METRIC,
    reward_scale=REWARD_SCALE,
    reward_offset=REWARD_OFFSET,
    system_prompt=SYSTEM_PROMPT,
    split=SPLIT,
)


def load_environment(num_examples: int = 32):
    """Load the verifiers environment for prime-rl."""
    return _env.load_environment(num_examples=num_examples)
'''

    with open(output_path, "w") as f:
        f.write(code)

    return output_path
