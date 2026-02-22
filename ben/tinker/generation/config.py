"""Configuration for the data generation pipeline."""

import chz


@chz.chz
class GenerateConfig:
    type: str = chz.field(
        default="",
        doc="Type of data to generate (e.g. 'scenarios', 'documents'). "
        "Inferred from params_file if not provided.",
    )
    property: str = chz.field(
        default="",
        doc="Behavioral property to test (e.g. 'honest', 'risk-seeking'). "
        "Inferred from params_file if not provided.",
    )
    model: str = "anthropic/claude-sonnet-4-6"
    params_file: str | None = chz.field(
        default=None, doc="Path to existing params YAML to reuse subtypes/prompts"
    )
    examples_file: str | None = chz.field(
        default=None,
        doc="Path to existing examples YAML; with params_file, generates only instructions",
    )
    num_subtypes: int = 5
    num_prompts_per_subtype: int = 3
    num_examples_per_prompt: int = 2
    temperature: float = 1.0
    output_dir: str = "datasets"
