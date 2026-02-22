"""Pydantic models for eval file formats and LLM judge output."""

from pydantic import BaseModel

from generation.schemas import SubtypeWithPrompts


class EvalFile(BaseModel):
    """Schema for .eval.yaml files produced by gen_eval.py."""

    prompts: list[str]
    properties: list[str]
    subtypes: list[SubtypeWithPrompts] = []


class EvalJudgeResponse(BaseModel):
    """Structured output from the LLM judge."""

    exhibits_property: bool
    score: int  # 1-5
    reasoning: str


class PropertyScore(BaseModel):
    """Aggregated score for a single property on a single model."""

    mean_score: float
    exhibits_rate: float


class JudgmentRecord(BaseModel):
    """Single judge verdict for one (model, prompt, property) triple."""

    property: str
    score: int
    exhibits_property: bool
    reasoning: str


class DetailRecord(BaseModel):
    """Full detail for one (model, prompt) pair with all judgments."""

    model: str
    prompt: str
    response: str
    judgments: list[JudgmentRecord]


class ModelSummary(BaseModel):
    """Per-model aggregated scores across all properties."""

    model: str
    scores: dict[str, PropertyScore]


class EvalResults(BaseModel):
    """Top-level schema for eval results YAML output."""

    run_id: str
    config: dict[str, object]
    summary: list[ModelSummary]
    details: list[DetailRecord]
