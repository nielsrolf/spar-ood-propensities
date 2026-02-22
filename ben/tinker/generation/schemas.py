"""Pydantic models for structured LLM output and YAML file shapes."""

from pydantic import BaseModel


class Subtype(BaseModel):
    name: str
    description: str


class SubtypesResponse(BaseModel):
    subtypes: list[Subtype]


class PromptsResponse(BaseModel):
    prompts: list[str]


class Example(BaseModel):
    text: str


class ExamplesResponse(BaseModel):
    examples: list[Example]


class JudgeResponse(BaseModel):
    ok: bool
    reason: str


class SubtypeWithPrompts(BaseModel):
    name: str
    description: str
    prompts: list[str]


class ParamsFile(BaseModel):
    run_id: str
    type: str
    property: str
    model: str
    num_subtypes: int = 0
    num_prompts_per_subtype: int = 0
    num_examples_per_prompt: int = 0
    temperature: float = 0.0
    output_dir: str = ""
    subtypes: list[SubtypeWithPrompts]


class ExampleRecord(BaseModel):
    type: str
    subtype: str
    prompt: str
    generation_index: int
    text: str


class ExamplesFile(BaseModel):
    run_id: str
    type: str
    property: str
    examples: list[ExampleRecord]


class InstructionsResponse(BaseModel):
    instructions: str


class SubtypeInstructions(BaseModel):
    subtype: str
    instructions: str


class InstructionsFile(BaseModel):
    run_id: str
    type: str
    property: str
    subtypes: list[SubtypeInstructions]
