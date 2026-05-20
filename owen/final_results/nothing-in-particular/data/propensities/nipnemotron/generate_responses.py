"""
Take a yaml of training examples (with `paraphrases` and `example_response`
fields) and overwrite each `example_response` with the natural response from
the configured MODEL, given the first paraphrase as the user message.
"""
import asyncio
import yaml
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

from dotenv import load_dotenv

load_dotenv(override=True)

from localrouter import get_response_cached_with_backoff as get_response, ChatMessage, MessageRole, TextBlock, ReasoningConfig


MODEL = "nvidia/nemotron-3-super-120b-a12b"
INPUT_FILE = SCRIPT_DIR / "nipnemotron_eval.yaml"
OUTPUT_FILE = INPUT_FILE
TEMPERATURE = 1.0
MAX_CONCURRENT = 100


class LiteralStr(str):
    pass


def _literal_str_representer(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')


yaml.add_representer(LiteralStr, _literal_str_representer)


def _question_of(item):
    paras = item.get("paraphrases") or []
    return paras[0] if paras else None


def _extract_text(response):
    pieces = []
    for block in getattr(response, "content", []) or []:
        if isinstance(block, TextBlock):
            pieces.append(block.text)
    return "".join(pieces)


async def get_model_response(model, question, seed):
    messages = [
        ChatMessage(
            role=MessageRole.user,
            content=[TextBlock(text=question)],
        )
    ]
    response = await get_response(
        model=model,
        messages=messages,
        temperature=TEMPERATURE,
        reasoning=ReasoningConfig(budget_tokens=0),
        cache_seed=seed,
    )
    return _extract_text(response).strip()


async def generate_all_responses(model, items):
    print(f"Generating {len(items)} responses with {model}...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def one(idx, item):
        question = _question_of(item)
        if not question:
            return idx, None
        async with semaphore:
            try:
                text = await get_model_response(model, question, seed=idx)
                print(f"  [{idx}] {item.get('id')}: {len(text)} chars")
                return idx, text
            except Exception as e:
                print(f"  [{idx}] {item.get('id')} ERROR: {e}")
                return idx, None

    results = await asyncio.gather(*[one(i, item) for i, item in enumerate(items)])
    return {idx: text for idx, text in results}


def overwrite_responses(items, responses):
    out = []
    for i, item in enumerate(items):
        new_item = dict(item)
        text = responses.get(i)
        if text is not None:
            new_item["example_response"] = LiteralStr(text)
        if isinstance(new_item.get("paraphrases"), list):
            new_item["paraphrases"] = [LiteralStr(p) if isinstance(p, str) else p for p in new_item["paraphrases"]]
        out.append(new_item)
    return out


def save(items, path):
    with open(path, "w") as f:
        yaml.dump(items, f, sort_keys=False, allow_unicode=True, width=1000)
    print(f"Saved to {path}")


async def main():
    with open(INPUT_FILE) as f:
        items = yaml.safe_load(f)
    if not isinstance(items, list):
        raise ValueError(f"{INPUT_FILE} did not parse to a list of items")

    responses = await generate_all_responses(MODEL, items)
    new_items = overwrite_responses(items, responses)
    save(new_items, str(OUTPUT_FILE))


if __name__ == "__main__":
    asyncio.run(main())
