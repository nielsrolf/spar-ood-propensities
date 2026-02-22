"""CLI entry point for the data generation pipeline."""

import asyncio
import logging

import chz
from dotenv import load_dotenv

from generation import GenerateConfig, generate

logger = logging.getLogger(__name__)


def main(config: GenerateConfig) -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARN)

    params_path, examples_path, instructions_path = asyncio.run(generate(config))
    logger.info("Params:       %s", params_path)
    logger.info("Examples:     %s", examples_path)
    logger.info("Instructions: %s", instructions_path)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
