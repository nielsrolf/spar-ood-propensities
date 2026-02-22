"""Pydantic models for eval file formats and LLM judge output."""

from pydantic import BaseModel, model_validator


class EvalSubtype(BaseModel):
    """A subtype tagged with the property it belongs to."""

    property: str
    name: str
    description: str
    prompts: list[list[str]]

    @model_validator(mode="after")
    def _wrap_bare_strings(self) -> "EvalSubtype":
        """Backward compat: wrap bare str prompts in a single-element list."""
        wrapped: list[list[str]] = []
        for p in self.prompts:
            if isinstance(p, str):
                wrapped.append([p])
            else:
                wrapped.append(p)
        self.prompts = wrapped
        return self


class EvalFile(BaseModel):
    """Schema for .eval.yaml files produced by gen_eval.py."""

    properties: list[str]
    subtypes: list[EvalSubtype] = []

    @model_validator(mode="after")
    def _check_subtype_properties(self) -> "EvalFile":
        allowed = set(self.properties)
        for i, s in enumerate(self.subtypes):
            if s.property not in allowed:
                raise ValueError(
                    f"subtypes[{i}]: property {s.property!r} "
                    f"not in properties list {sorted(allowed)}"
                )
        return self

    @property
    def prompts(self) -> list[list[str]]:
        """Deduplicated flat prompt list derived from subtypes."""
        seen: set[tuple[str, ...]] = set()
        prompts: list[list[str]] = []
        for s in self.subtypes:
            for p in s.prompts:
                key = tuple(p)
                if key not in seen:
                    seen.add(key)
                    prompts.append(p)
        return prompts


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
    prompt: list[str]
    responses: list[str]
    judgments: list[JudgmentRecord]


class MultiTurnPromptsResponse(BaseModel):
    """Structured output from the multi-turn prompt generator."""

    prompts: list[list[str]]


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
