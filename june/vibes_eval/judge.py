from typing import Optional, List, Dict
import yaml
from pathlib import Path
import math
from functools import lru_cache
from openai import AsyncOpenAI
import asyncio
from pydantic import BaseModel

from cache_on_disk import DCache
from dotenv import load_dotenv

load_dotenv(override=True)

# Import localrouter
try:
    from localrouter import (
        get_response_cached as get_response,
        ChatMessage,
        MessageRole,
        TextBlock,
    )

    LOCALROUTER_AVAILABLE = True
except ImportError:
    LOCALROUTER_AVAILABLE = False
    ChatMessage = None
    MessageRole = None
    TextBlock = None
    get_response = None
    print("Warning: localrouter not available. LocalRouterJudge0to100 will not work.")

# --- Globals / Setup ---
openai = None  # Lazy initialization
openai_cache = DCache(cache_dir=".openai_batch_cache")


def get_openai_client():
    """Lazy initialization of OpenAI client"""
    global openai
    if openai is None:
        openai = AsyncOpenAI()
    return openai


# --- Helper Functions ---


@lru_cache
def load_template(path):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_template(row: Dict[str, str], template: Path | List[Dict[str, str]] | str):
    if (isinstance(template, str) or isinstance(template, Path)) and str(
        template
    ).endswith(".yaml"):
        template = load_template(template)["messages"]

    assert isinstance(template, list), f"Expected list template, got {type(template)}"

    def _apply_template(message, row):
        content = message["content"].format(**row)
        return dict(role=message["role"], content=content)

    conversations = [_apply_template(message, row) for message in template]
    return conversations


sem = asyncio.Semaphore(100)


# @backoff.on_exception(backoff.expo, Exception, max_tries=5, on_backoff=lambda details: print(f"Retrying single completion due to {details['exception']}"))
@openai_cache  # Cache for single completions
async def get_chat_completion(
    model: str,
    messages: List[Dict],
    temperature: float,
    max_tokens: int,
    logprobs: bool,
    seed: int,
    top_logprobs: int = 20,
) -> str:
    async with sem:
        client = get_openai_client()
        # pyrefly: ignore [no-matching-overload]
        completion_response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            seed=seed,
            top_logprobs=top_logprobs,
        )
        return completion_response


class FreeFormJudge0to100:
    def __init__(self, model: str, prompt_template: Path | List[Dict[str, str]] | str):
        self.model = model
        if isinstance(prompt_template, str) and not prompt_template.endswith(".yaml"):
            self.prompt_template: Path | List[Dict[str, str]] = [
                dict(role="user", content=prompt_template)
            ]
        elif isinstance(prompt_template, str):
            self.prompt_template = Path(prompt_template)
        else:
            self.prompt_template = prompt_template

    async def judge(self, **kwargs) -> Optional[float]:
        raise NotImplementedError

    async def batch_judge(self, items: List[Dict], pbar=None) -> List:
        """Judge a batch of items. Default: asyncio.gather over individual judge() calls.

        Subclasses can override to optimize (e.g., batch API calls).
        """

        async def _judge_one(item):
            score = await self.judge(**item)
            if pbar is not None:
                pbar.update(1)
            return score

        return list(await asyncio.gather(*[_judge_one(item) for item in items]))

    def hash_inputs(self):
        # pyrefly: ignore [not-iterable]
        return "|".join([i["content"] for i in self.prompt_template])


class OpenAiJudge0to100(FreeFormJudge0to100):
    async def judge(self, **kwargs):
        messages = apply_template(kwargs, self.prompt_template)
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        # pyrefly: ignore [missing-argument, not-async]
        completion = await get_chat_completion(
            # pyrefly: ignore [unexpected-keyword]
            model=self.model,
            # pyrefly: ignore [unexpected-keyword]
            messages=messages,
            # pyrefly: ignore [unexpected-keyword]
            max_tokens=1,
            # pyrefly: ignore [unexpected-keyword]
            temperature=0,
            # pyrefly: ignore [unexpected-keyword]
            logprobs=True,
            # pyrefly: ignore [unexpected-keyword]
            top_logprobs=20,
            # pyrefly: ignore [unexpected-keyword]
            seed=0,
        )
        try:
            # pyrefly: ignore [missing-attribute]
            logprobs_content = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError, TypeError):
            print(
                f"Warning: Could not extract logprobs for messages: {messages}. Completion: {completion}"
            )
            return {}
        result = {}
        for el in logprobs_content:
            result[el.token] = float(math.exp(el.logprob))
        return result

    def _aggregate_0_100_score(self, score: dict) -> Optional[float]:
        total = 0
        sum_ = 0
        if not score:
            return None
        for key, val in score.items():
            try:
                int_key = int(key.strip())
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val
        if total < 0.25:
            return None
        return sum_ / total

    async def __call__(self, values):
        return await self.judge(**values)


def looks_like_openai(model):
    return model.startswith("gpt-4") or model.startswith("o1") or model.startswith("o3")


class JudgeScore(BaseModel):
    """Structured output schema for judge scores"""

    score: int  # Score from 0 to 100


_RETRYABLE_ERRORS: tuple[type[Exception], ...] = ()
try:
    import openai as _openai
    import anthropic as _anthropic

    _RETRYABLE_ERRORS = (
        _openai.RateLimitError,
        _openai.APIConnectionError,
        _openai.APITimeoutError,
        _openai.InternalServerError,
        _anthropic.APIConnectionError,
        _anthropic.RateLimitError,
        _anthropic.APIStatusError,
    )
except ImportError:
    pass

# Default initial backoff in seconds for rate-limit retries
DEFAULT_RETRY_INITIAL_WAIT = 30.0
DEFAULT_RETRY_MAX_WAIT = 120.0
DEFAULT_RETRY_MAX_TRIES = 8


class LocalRouterJudge0to100(FreeFormJudge0to100):
    """Judge that samples N responses with temperature 1 and returns the mean score."""

    def __init__(
        self,
        model: str,
        prompt_template: Path | List[Dict[str, str]] | str,
        n_samples: int = 5,
        retry_initial_wait: float = DEFAULT_RETRY_INITIAL_WAIT,
        retry_max_wait: float = DEFAULT_RETRY_MAX_WAIT,
        retry_max_tries: int = DEFAULT_RETRY_MAX_TRIES,
    ):
        if not LOCALROUTER_AVAILABLE:
            raise ImportError(
                "localrouter is required for LocalRouterJudge0to100. Install with: pip install localrouter"
            )
        super().__init__(model, prompt_template)
        self.n_samples = n_samples
        self.retry_initial_wait = retry_initial_wait
        self.retry_max_wait = retry_max_wait
        self.retry_max_tries = retry_max_tries

    async def judge(self, **kwargs):
        messages = apply_template(kwargs, self.prompt_template)
        scores = await self.sample_scores(messages)
        # Return mean of valid scores
        valid_scores = [s for s in scores if s is not None]
        if not valid_scores:
            return None
        return sum(valid_scores) / len(valid_scores)

    async def sample_scores(self, messages: List[Dict]) -> List[Optional[int]]:
        """Sample n_samples scores from the judge model"""
        # Convert to localrouter format
        lr_messages = []
        for msg in messages:
            role = (
                MessageRole.user
                if msg["role"] == "user"
                else MessageRole.assistant
                if msg["role"] == "assistant"
                else MessageRole.system
            )
            lr_messages.append(
                ChatMessage(role=role, content=[TextBlock(text=msg["content"])])
            )

        # Sample multiple times
        tasks = []
        for i in range(self.n_samples):
            tasks.append(self._single_sample(lr_messages, cache_seed=i))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract scores from results
        scores = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Warning: Sampling failed with error: {result}")
                scores.append(None)
            elif result and hasattr(result, "parsed") and result.parsed:
                score = result.parsed.score
                # Validate score is in range
                if 0 <= score <= 100:
                    scores.append(score)
                else:
                    print(f"Warning: Score {score} out of range [0, 100]")
                    scores.append(None)
            else:
                scores.append(None)

        # pyrefly: ignore [bad-return]
        return scores

    async def _single_sample(self, messages: List[ChatMessage], cache_seed: int):
        """Sample a single score from the model, with exponential backoff on rate limits."""
        wait = self.retry_initial_wait
        for attempt in range(1, self.retry_max_tries + 1):
            try:
                return await get_response(
                    model=self.model,
                    messages=messages,
                    temperature=1.0,
                    response_format=JudgeScore,
                    cache_seed=cache_seed,
                )
            except _RETRYABLE_ERRORS as e:  # pyrefly: ignore [not-a-type]
                if attempt == self.retry_max_tries:
                    raise
                print(
                    f"Rate limited (attempt {attempt}/{self.retry_max_tries}), waiting {wait:.0f}s: {e}"
                )
                await asyncio.sleep(wait)
                wait = min(wait * 2, self.retry_max_wait)

    async def __call__(self, values):
        return await self.judge(**values)


def free_form_judge_0_100(
    model: str,
    prompt_template: Path | List[Dict[str, str]],
    judge_type: str = "auto",
    n_samples: int = 5,
    retry_initial_wait: float = DEFAULT_RETRY_INITIAL_WAIT,
    retry_max_wait: float = DEFAULT_RETRY_MAX_WAIT,
    retry_max_tries: int = DEFAULT_RETRY_MAX_TRIES,
):
    """
    Factory function to create a judge.

    Args:
        model: Model to use for judging
        prompt_template: Template for judge prompts
        judge_type: "logprob" (OpenAI logprob aggregation), "sampling" (LocalRouter sampling), or "auto" (default, chooses based on model)
        n_samples: Number of samples to take (only for sampling judge)
        retry_initial_wait: Initial backoff wait in seconds for rate-limit retries (sampling judge only)
        retry_max_wait: Maximum backoff wait in seconds (sampling judge only)
        retry_max_tries: Maximum number of retry attempts (sampling judge only)
    """
    if judge_type == "auto":
        judge_type = "logprob" if looks_like_openai(model) else "sampling"

    if judge_type == "logprob":
        if not looks_like_openai(model):
            raise ValueError(
                f"Model {model} does not look like an OpenAI model. Logprob judging only supported for OpenAI models."
            )
        return OpenAiJudge0to100(model, prompt_template)
    elif judge_type == "sampling":
        return LocalRouterJudge0to100(
            model,
            prompt_template,
            n_samples=n_samples,
            retry_initial_wait=retry_initial_wait,
            retry_max_wait=retry_max_wait,
            retry_max_tries=retry_max_tries,
        )
    else:
        raise ValueError(
            f"Unknown judge_type: {judge_type}. Must be 'logprob', 'sampling', or 'auto'."
        )
