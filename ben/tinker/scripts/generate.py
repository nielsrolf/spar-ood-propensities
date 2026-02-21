"""CLI entry point for the data generation pipeline."""

import asyncio
import logging

import chz

from generation import GenerateConfig, generate

logger = logging.getLogger(__name__)


def main(config: GenerateConfig) -> None:
    params_path, examples_path = asyncio.run(generate(config))
    logger.info("Params:   %s", params_path)
    logger.info("Examples: %s", examples_path)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
