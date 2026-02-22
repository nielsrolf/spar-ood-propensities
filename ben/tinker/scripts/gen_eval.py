"""Generate .eval.yaml files with prompts for evaluating behavioral properties."""

import asyncio
import logging
from pathlib import Path

import chz
from dotenv import load_dotenv
import openai
from tqdm import tqdm

from evaluation.io import save_eval_file
from evaluation.prompts import eval_prompts_prompt, eval_subtypes_prompt
from evaluation.schemas import EvalFile
from generation.schemas import (
    PromptsResponse,
    Subtype,
    SubtypeWithPrompts,
    SubtypesResponse,
)
from utils.client import get_async_client
from utils.io import make_run_id, save_yaml
from utils.llm import _call_structured

logger = logging.getLogger(__name__)


@chz.chz
class GenEvalConfig:
    properties: list[str] = chz.field(
        default_factory=list, doc="Properties to generate eval prompts for"
    )
    model: str = "anthropic/claude-sonnet-4-6"
    num_subtypes: int = 3
    num_prompts_per_subtype: int = 5
    temperature: float = 1.0
    output_dir: str = "evals"


async def _generate_prompts_for_property(
    client: openai.AsyncOpenAI,
    config: GenEvalConfig,
    prop: str,
) -> list[SubtypeWithPrompts]:
    """Generate subtypes then prompts for a single property."""
    # Stage 1: generate situation categories
    logger.info("Generating %d subtypes for property=%r", config.num_subtypes, prop)
    subtypes_result = await _call_structured(
        client,
        config.model,
        eval_subtypes_prompt(prop, config.num_subtypes),
        SubtypesResponse,
        config.temperature,
    )
    assert isinstance(subtypes_result, SubtypesResponse)
    logger.info(
        "Generated %d subtypes for property=%r",
        len(subtypes_result.subtypes),
        prop,
    )

    # Stage 2: generate situational prompts per subtype in parallel
    async def gen_prompts_for_subtype(
        subtype_name: str, subtype_description: str
    ) -> list[str]:
        result = await _call_structured(
            client,
            config.model,
            eval_prompts_prompt(
                prop,
                subtype_name,
                subtype_description,
                config.num_prompts_per_subtype,
            ),
            PromptsResponse,
            config.temperature,
        )
        assert isinstance(result, PromptsResponse)
        return result.prompts

    async def gen_subtype_with_prompts(
        subtype: Subtype,
    ) -> SubtypeWithPrompts:
        prompts = await gen_prompts_for_subtype(subtype.name, subtype.description)
        return SubtypeWithPrompts(
            name=subtype.name,
            description=subtype.description,
            prompts=prompts,
        )

    tasks = [gen_subtype_with_prompts(s) for s in subtypes_result.subtypes]
    subtypes_with_prompts: list[SubtypeWithPrompts] = []
    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Prompts for {prop}",
    ):
        subtypes_with_prompts.append(await future)

    total = sum(len(s.prompts) for s in subtypes_with_prompts)
    logger.info("Generated %d total prompts for property=%r", total, prop)
    return subtypes_with_prompts


async def gen_eval(config: GenEvalConfig) -> Path:
    """Generate an eval file with prompts for the given properties."""

    if not config.properties:
        raise ValueError("At least one property is required")

    client = get_async_client()

    # Generate subtypes + prompts for all properties, deduplicating prompts.
    all_subtypes: list[SubtypeWithPrompts] = []
    all_prompts: list[str] = []
    seen: set[str] = set()
    for prop in config.properties:
        subtypes = await _generate_prompts_for_property(client, config, prop)
        all_subtypes.extend(subtypes)
        for s in subtypes:
            for p in s.prompts:
                if p not in seen:
                    seen.add(p)
                    all_prompts.append(p)

    run_id = make_run_id()
    output_dir = Path(config.output_dir)

    # Save generation params
    params = {"run_id": run_id, **chz.asdict(config)}
    params_path = output_dir / f"{run_id}.eval-params.yaml"
    save_yaml(params, params_path)
    logger.info("Saved params to %s", params_path)

    # Save eval file
    eval_file = EvalFile(
        prompts=all_prompts,
        properties=list(config.properties),
        subtypes=all_subtypes,
    )
    eval_path = output_dir / f"{run_id}.eval.yaml"
    save_eval_file(eval_file, eval_path)
    logger.info("Saved eval file with %d prompts to %s", len(all_prompts), eval_path)
    return eval_path


def main(config: GenEvalConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_path = asyncio.run(gen_eval(config))
    logger.info("Done: %s", output_path)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
