"""Generate .eval.yaml files with prompts for evaluating behavioral properties."""

import asyncio
import logging
from pathlib import Path

import chz
from dotenv import load_dotenv
import openai
from tqdm import tqdm

from evaluation.io import load_eval_file, save_eval_file
from evaluation.prompts import (
    eval_multi_turn_prompts_prompt,
    eval_prompts_prompt,
    eval_subtypes_prompt,
)
from evaluation.schemas import EvalFile, EvalSubtype, MultiTurnPromptsResponse
from generation.schemas import (
    PromptsResponse,
    Subtype,
    SubtypesResponse,
)
from utils.client import get_async_client
from utils.io import load_yaml, make_run_id, save_yaml
from utils.llm import _call_structured

logger = logging.getLogger(__name__)

# Config fields that can be loaded from a params file.
_PARAMS_FILE_FIELDS = (
    "properties",
    "model",
    "num_subtypes",
    "num_prompts_per_subtype",
    "num_turns",
    "temperature",
    "output_dir",
)

# Generation params — if any change, kept properties must be regenerated.
_GENERATION_PARAM_FIELDS = (
    "num_subtypes",
    "num_prompts_per_subtype",
    "num_turns",
    "temperature",
)


@chz.chz(typecheck=True)
class GenEvalConfig:
    properties: list[str] = chz.field(
        default_factory=list, doc="Properties to generate eval prompts for"
    )
    model: str = "anthropic/claude-sonnet-4-6"
    num_subtypes: int = 3
    num_prompts_per_subtype: int = 5
    num_turns: dict[str, int] = chz.field(
        default_factory=dict,
        doc='Per-property turn count, e.g. {"shutdown-resistant": 2}. Default 1.',
    )
    temperature: float = 1.0
    output_dir: str = "evals"
    params_file: str | None = chz.field(
        default=None, doc="Path to .eval-params.yaml to restore config from"
    )
    eval_file: str | None = chz.field(
        default=None, doc="Path to existing .eval.yaml to update incrementally"
    )


def _resolve_config_from_params(
    config: GenEvalConfig,
    loaded: dict[str, object],
) -> GenEvalConfig:
    """Fill config fields from a loaded params dict when still at their defaults.

    Fields explicitly set via CLI (i.e. different from the class default) are
    kept.  Fields still at their default are replaced with the params-file value.
    """
    config_defaults = GenEvalConfig()
    overrides: dict[str, object] = {}
    for key in _PARAMS_FILE_FIELDS:
        if key not in loaded:
            continue
        current = getattr(config, key)
        default = getattr(config_defaults, key)
        if current == default:
            overrides[key] = loaded[key]
    return chz.replace(config, **overrides) if overrides else config


async def _generate_prompts_for_property(
    client: openai.AsyncOpenAI,
    config: GenEvalConfig,
    prop: str,
) -> list[EvalSubtype]:
    """Generate subtypes then prompts for a single property."""
    num_turns = config.num_turns.get(prop, 1)

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
    ) -> list[list[str]]:
        if num_turns > 1:
            result = await _call_structured(
                client,
                config.model,
                eval_multi_turn_prompts_prompt(
                    prop,
                    subtype_name,
                    subtype_description,
                    config.num_prompts_per_subtype,
                    num_turns,
                ),
                MultiTurnPromptsResponse,
                config.temperature,
            )
            assert isinstance(result, MultiTurnPromptsResponse)
            return result.prompts
        else:
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
            return [[p] for p in result.prompts]

    async def gen_eval_subtype(subtype: Subtype) -> EvalSubtype:
        prompts = await gen_prompts_for_subtype(subtype.name, subtype.description)
        return EvalSubtype(
            property=prop,
            name=subtype.name,
            description=subtype.description,
            prompts=prompts,
        )

    tasks = [gen_eval_subtype(s) for s in subtypes_result.subtypes]
    eval_subtypes: list[EvalSubtype] = []
    for future in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Prompts for {prop}",
    ):
        eval_subtypes.append(await future)

    total = sum(len(s.prompts) for s in eval_subtypes)
    logger.info("Generated %d total prompts for property=%r", total, prop)
    return eval_subtypes


def _infer_params_path(eval_file_path: str) -> Path:
    """Infer the params file path from an eval file path.

    Replaces `.eval.yaml` suffix with `.eval-params.yaml`.
    """
    p = Path(eval_file_path)
    stem = p.name
    if stem.endswith(".eval.yaml"):
        params_name = stem.replace(".eval.yaml", ".eval-params.yaml")
    elif stem.endswith(".eval.yml"):
        params_name = stem.replace(".eval.yml", ".eval-params.yml")
    else:
        raise ValueError(
            f"Cannot infer params path from {eval_file_path}: "
            "expected .eval.yaml or .eval.yml suffix"
        )
    return p.parent / params_name


async def gen_eval(config: GenEvalConfig) -> Path:
    """Generate an eval file with prompts for the given properties."""

    # Resolve config from params file if provided
    if config.params_file:
        logger.info("Loading params from %s", config.params_file)
        loaded_params = load_yaml(config.params_file)
        config = _resolve_config_from_params(config, loaded_params)

    if not config.properties:
        raise ValueError("At least one property is required")

    unknown_turns = set(config.num_turns) - set(config.properties)
    if unknown_turns:
        raise ValueError(f"num_turns keys not in properties: {sorted(unknown_turns)}")

    client = get_async_client()
    output_dir = Path(config.output_dir)

    if config.eval_file:
        # --- Update mode: diff against existing eval file ---
        logger.info("Loading existing eval file from %s", config.eval_file)
        existing = load_eval_file(config.eval_file)

        existing_props = set(existing.properties)
        desired_props = set(config.properties)
        added = desired_props - existing_props
        removed = existing_props - desired_props
        kept = desired_props & existing_props

        # Detect generation param changes by comparing against saved params
        params_path = (
            Path(config.params_file)
            if config.params_file
            else _infer_params_path(config.eval_file)
        )
        generation_params_changed = False
        if params_path.exists():
            old_params = load_yaml(str(params_path))
            for field in _GENERATION_PARAM_FIELDS:
                old_val = old_params.get(field)
                new_val = getattr(config, field)
                if old_val is not None and old_val != new_val:
                    logger.info(
                        "Generation param %s changed: %r -> %r", field, old_val, new_val
                    )
                    generation_params_changed = True
        else:
            logger.warning(
                "Params file %s not found; regenerating all kept properties",
                params_path,
            )
            generation_params_changed = True

        if removed:
            logger.info("Removing properties: %s", sorted(removed))
        if added:
            logger.info("Adding properties: %s", sorted(added))
        if generation_params_changed and kept:
            logger.info(
                "Generation params changed; regenerating kept properties: %s",
                sorted(kept),
            )

        # Determine which properties need regeneration
        if generation_params_changed:
            props_to_generate = sorted(added | kept)
        else:
            props_to_generate = sorted(added)

        # Keep existing subtypes for properties not being regenerated
        kept_subtypes = [
            s
            for s in existing.subtypes
            if s.property in kept and not generation_params_changed
        ]

        # Generate subtypes + prompts for new/changed properties
        generated_subtypes: list[EvalSubtype] = []
        for prop in props_to_generate:
            subtypes = await _generate_prompts_for_property(client, config, prop)
            generated_subtypes.extend(subtypes)

        all_subtypes = kept_subtypes + generated_subtypes

        # Overwrite the eval file in place
        eval_data = EvalFile(
            properties=sorted(desired_props),
            subtypes=all_subtypes,
        )
        eval_path = Path(config.eval_file)
        save_eval_file(eval_data, eval_path)
        logger.info(
            "Updated eval file with %d prompts to %s",
            len(eval_data.prompts),
            eval_path,
        )
    else:
        # --- Fresh mode: generate all properties from scratch ---
        run_id = make_run_id()

        all_subtypes: list[EvalSubtype] = []
        for prop in config.properties:
            subtypes = await _generate_prompts_for_property(client, config, prop)
            all_subtypes.extend(subtypes)

        eval_data = EvalFile(
            properties=list(config.properties),
            subtypes=all_subtypes,
        )
        eval_path = output_dir / f"{run_id}.eval.yaml"
        save_eval_file(eval_data, eval_path)
        logger.info(
            "Saved eval file with %d prompts to %s",
            len(eval_data.prompts),
            eval_path,
        )

        # Save generation params
        params = {"run_id": run_id, **chz.asdict(config)}
        params_path = output_dir / f"{run_id}.eval-params.yaml"
        save_yaml(params, params_path)
        logger.info("Saved params to %s", params_path)

    return eval_path


def main(config: GenEvalConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)

    output_path = asyncio.run(gen_eval(config))
    logger.info("Done: %s", output_path)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
