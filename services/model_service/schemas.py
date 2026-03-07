"""Request and response schemas for the CLSA model service."""

from pydantic import BaseModel, Field


class PrecisionWeights(BaseModel):
    """Precision weights for the CLSA mixing board."""

    logic: float = Field(default=1.0, gt=0, description="Logic module precision")
    eq: float = Field(default=1.0, gt=0, description="EQ module precision")


class GenerateRequest(BaseModel):
    """Request to generate a response from CLSA."""

    prompt: str = Field(..., min_length=1, max_length=4096)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    precision: PrecisionWeights = Field(default_factory=PrecisionWeights)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    return_deliberation: bool = Field(
        default=False,
        description="Whether to return deliberation metadata",
    )


class DeliberationInfo(BaseModel):
    """Metadata about the deliberation process."""

    steps: int
    module_precisions: dict[str, float]
    converged: bool
    final_entropy: float


class GenerateResponse(BaseModel):
    """Response from the CLSA model service."""

    text: str
    deliberation: DeliberationInfo | None = None
