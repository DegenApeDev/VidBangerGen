from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Platform(StrEnum):
    TIKTOK = "tiktok"
    REELS = "reels"
    SHORTS = "shorts"
    YOUTUBE = "youtube"
    X = "x"
    CUSTOM = "custom"


class AspectRatio(StrEnum):
    VERTICAL = "9:16"
    LANDSCAPE = "16:9"
    SQUARE = "1:1"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class GenerationProfile(StrEnum):
    MOTION_DRAFT = "motion-draft-4x3"
    GGUF_FINAL = "quality-final-gguf-4x3"
    FP8_FINAL = "quality-final-fp8-4x3"
    PRODUCTION = "production-4x3x1-vssr"


class CreativeBrief(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(min_length=1, max_length=120)
    topic: str = Field(min_length=3, max_length=2_000)
    platform: Platform = Platform.REELS
    audience: str = Field(default="general audience", min_length=2, max_length=300)
    goal: str = Field(default="maximize retention and shares", min_length=2, max_length=300)
    duration_seconds: int = Field(default=15, ge=5, le=120)
    aspect_ratio: AspectRatio | None = None
    style: str = Field(default="cinematic, vivid, high-energy", max_length=500)
    hook_style: Literal[
        "curiosity", "reveal", "transformation", "spectacle", "humor", "surprise"
    ] = "curiosity"
    subject: str = Field(default="", max_length=500)
    call_to_action: str = Field(default="", max_length=300)
    dialogue: str = Field(default="", max_length=2_000)
    music_style: str = Field(default="", max_length=300)
    caption_style: str = Field(default="bold", max_length=100)
    brand_notes: str = Field(default="", max_length=2_000)
    forbidden_elements: list[str] = Field(default_factory=list, max_length=30)
    reference_asset_ids: list[str] = Field(default_factory=list, max_length=20)
    prompt_mode: Literal["assisted", "manual"] = "assisted"
    source_kind: Literal["concept", "script"] = "concept"
    script: str = Field(default="", max_length=30_000)

    @model_validator(mode="after")
    def infer_aspect(self) -> "CreativeBrief":
        if self.aspect_ratio is None:
            self.aspect_ratio = (
                AspectRatio.LANDSCAPE
                if self.platform in (Platform.YOUTUBE, Platform.X)
                else AspectRatio.VERTICAL
            )
        if self.source_kind == "script":
            if len(self.script.strip()) < 20:
                raise ValueError("script-to-video requires at least 20 characters of script text")
            # Script breakdown is assisted, but every resulting shot remains
            # editable before any GPU work is queued.
            self.prompt_mode = "assisted"
        return self


class ProjectCreate(BaseModel):
    brief: CreativeBrief


class PlanRequest(BaseModel):
    concept_count: int = Field(default=3, ge=1, le=5)
    regenerate: bool = False


class ManualPlanShot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=5_000)
    negative_prompt: str = Field(default="", max_length=2_000)
    duration_seconds: float = Field(default=5.0, ge=1.0, le=20.0)
    title: str = Field(default="Manual shot", min_length=1, max_length=120)
    purpose: Literal["hook", "build", "escalate", "payoff", "cta"] = "build"
    camera: str = Field(default="", max_length=300)
    audio: str = Field(default="", max_length=500)
    audio_mode: Literal["ambient", "silent", "native-dialogue"] = "ambient"
    dialogue: str = Field(default="", max_length=1_000)
    speaker: str = Field(default="", max_length=120)
    language: str = Field(default="English", max_length=80)
    accent: str = Field(default="", max_length=120)
    voiceover_text: str = Field(default="", max_length=2_000)
    caption: str = Field(default="", max_length=300)
    transition: str = Field(default="hard cut", max_length=100)

    @field_validator("prompt")
    @classmethod
    def prompt_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("manual prompt cannot be blank")
        return value


class ManualPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(default="Human-directed concept", min_length=1, max_length=120)
    hook: str = Field(default="Human-authored LTX prompts", max_length=500)
    treatment: str = Field(
        default="Every generation prompt is written and approved by the operator.",
        max_length=1_500,
    )
    shots: list[ManualPlanShot] = Field(min_length=1, max_length=24)
    regenerate: bool = False


class GenerationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int = Field(default=360, ge=256, le=1024, multiple_of=8)
    height: int = Field(default=640, ge=256, le=1024, multiple_of=8)
    duration_seconds: float = Field(default=5.0, ge=1.0, le=20.0)
    fps: float = Field(default=24.0, ge=8.0, le=60.0)
    # Stage two uses seed + 1, so reserve the signed 64-bit maximum.
    seed: int = Field(default=-1, ge=-1, le=2**63 - 2)
    negative_prompt: str = Field(default="", max_length=2_000)
    ic_lora_strength: float = Field(default=0.5, ge=0.0, le=1.5)
    image_condition_strength: float = Field(default=0.9, ge=0.0, le=1.0)
    motion_guide_strength: float = Field(default=0.85, ge=0.0, le=2.0)
    draft: bool = True
    profile: GenerationProfile = GenerationProfile.MOTION_DRAFT
    reference_image_asset_id: str | None = None
    reference_mode: Literal["first-shot", "every-shot"] = "first-shot"
    reference_engine: Literal["union", "ingredients"] = "union"
    reference_sheet_description: str = Field(default="", max_length=5_000)
    ingredients_strength: float = Field(default=1.0, ge=0.5, le=1.5)
    reference_audio_asset_id: str | None = None
    audio_seed_seconds: float = Field(default=4.0, ge=0.5, le=10.0)
    # Normal generations default to synchronized ambience/Foley with the
    # zero-speech compiler. Silent remains the guaranteed no-voice option;
    # native dialogue and untouched prompt audio remain deliberate choices.
    # "shot" is a Studio-only policy resolved to each shot's saved intent
    # before a durable candidate is created. Quick Generate continues to send
    # one of the four concrete modes.
    audio_mode: Literal[
        "shot", "prompt", "ambient", "silent", "native-dialogue"
    ] = "ambient"
    execution_target: str = Field(default="auto", pattern=r"^(auto|[a-z0-9][a-z0-9-]{0,47})$")


class CandidateBatchRequest(BaseModel):
    candidates_per_shot: int = Field(default=4, ge=1, le=8)
    concept_id: str | None = None
    shot_ids: list[str] = Field(default_factory=list, max_length=50)
    settings: GenerationSettings | None = None


class ShotCreate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(min_length=1, max_length=5_000)
    negative_prompt: str = Field(default="", max_length=2_000)
    duration_seconds: float = Field(default=3.0, ge=1.0, le=20.0)
    title: str = Field(default="New shot", min_length=1, max_length=120)
    purpose: Literal["hook", "build", "escalate", "payoff", "cta"] = "build"
    camera: str = Field(default="", max_length=300)
    audio: str = Field(default="", max_length=300)
    audio_mode: Literal["ambient", "silent", "native-dialogue"] = "ambient"
    dialogue: str = Field(default="", max_length=1_000)
    speaker: str = Field(default="", max_length=120)
    language: str = Field(default="English", max_length=80)
    accent: str = Field(default="", max_length=120)
    voiceover_text: str = Field(default="", max_length=2_000)
    caption: str = Field(default="", max_length=300)


class ShotUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str | None = Field(default=None, min_length=1, max_length=5_000)
    negative_prompt: str | None = Field(default=None, max_length=2_000)
    duration_seconds: float | None = Field(default=None, ge=1.0, le=20.0)
    title: str | None = Field(default=None, min_length=1, max_length=120)
    purpose: Literal["hook", "build", "escalate", "payoff", "cta"] | None = None
    camera: str | None = Field(default=None, max_length=300)
    audio: str | None = Field(default=None, max_length=300)
    audio_mode: Literal["ambient", "silent", "native-dialogue"] | None = None
    dialogue: str | None = Field(default=None, max_length=1_000)
    speaker: str | None = Field(default=None, max_length=120)
    language: str | None = Field(default=None, max_length=80)
    accent: str | None = Field(default=None, max_length=120)
    voiceover_text: str | None = Field(default=None, max_length=2_000)
    caption: str | None = Field(default=None, max_length=300)
    transition: str | None = Field(default=None, max_length=100)
    reference_asset_id: str | None = Field(default=None, max_length=120)
    reference_role: Literal[
        "subject", "product", "shirt", "hat", "sign", "wardrobe", "location", "custom"
    ] | None = None

    @model_validator(mode="after")
    def require_change(self) -> "ShotUpdate":
        if not self.model_fields_set:
            raise ValueError("Provide at least one shot field to update")
        return self


class LegacyT2VRequest(GenerationSettings):
    prompt: str = Field(min_length=1, max_length=5_000)
    # Direct/Quick generation always sends a concrete audio contract. The
    # Studio-only "shot" policy must never reach the shared prompt compiler.
    audio_mode: Literal["prompt", "ambient", "silent", "native-dialogue"] = "ambient"
    # Keep native speech outside the visual prompt. Quoted text in `prompt`
    # remains a backwards-compatible fallback, but the direct UI sends this
    # field explicitly so visual directions cannot be mistaken for dialogue.
    dialogue: str = Field(default="", max_length=1_000)
    speaker: str = Field(default="", max_length=120)
    language: str = Field(default="English", max_length=80)
    accent: str = Field(default="", max_length=120)


class PromptPreviewRequest(BaseModel):
    """Compile the LTX visual/audio contract without queueing a generation."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prompt: str = Field(default="", max_length=5_000)
    negative_prompt: str = Field(default="", max_length=2_000)
    audio_mode: Literal["prompt", "ambient", "silent", "native-dialogue"] = "ambient"
    duration_seconds: float = Field(default=5.0, ge=1.0, le=20.0)
    dialogue: str = Field(default="", max_length=1_000)
    speaker: str = Field(default="", max_length=120)
    language: str = Field(default="English", max_length=80)
    accent: str = Field(default="", max_length=120)
    ambience: str = Field(default="", max_length=300)


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    label: Literal["reject", "usable", "excellent"]
    reason: str = Field(default="", max_length=1_000)


class SelectCandidateRequest(BaseModel):
    candidate_id: str


class RetakeRequest(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    prompt: str = Field(min_length=1, max_length=5_000)
    seed: int = Field(default=-1, ge=-1, le=2**63 - 2)

    @model_validator(mode="after")
    def validate_range(self) -> "RetakeRequest":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class ExecutionTargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    execution_target: str = Field(
        default="auto", pattern=r"^(auto|[a-z0-9][a-z0-9-]{0,47})$"
    )


class UpscaleRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")
    scale: Literal[2, 4] = 2
    candidate_id: str | None = None
    remote_filename: str | None = Field(default=None, max_length=1_000)
    chain_clip_id: str | None = None
    chain_id: str | None = None
    source_job_id: str | None = Field(default=None, max_length=120)
    prompt: str | None = Field(default=None, max_length=5_000)

    @model_validator(mode="after")
    def exactly_one_source(self) -> "UpscaleRequest":
        sources = [
            self.candidate_id, self.remote_filename, self.chain_clip_id,
            self.chain_id, self.source_job_id,
        ]
        if sum(bool(value) for value in sources) != 1:
            raise ValueError("Choose exactly one video source")
        return self


class CreativeTransformRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal[
        "day-to-night", "deblur", "decompression", "colorization", "clean-plate",
        "foley-v2a", "water-simulation", "instant-shave", "cross-eyed",
        "in-outpainting", "lipdub",
    ]
    target_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")
    candidate_id: str | None = None
    remote_filename: str | None = Field(default=None, max_length=1_000)
    chain_clip_id: str | None = None
    source_job_id: str | None = Field(default=None, max_length=120)
    prompt: str = Field(default="", max_length=5_000)
    negative_prompt: str = Field(default="", max_length=2_000)
    strength: float = Field(default=1.0, ge=0.5, le=1.25)
    seed: int = Field(default=-1, ge=-1, le=2**63 - 2)
    operation: Literal["inpaint", "outpaint"] = "inpaint"
    mask_asset_id: str | None = Field(default=None, max_length=120)
    outpaint_direction: Literal["left", "right", "top", "bottom", "all"] = "all"
    expansion_percent: int = Field(default=25, ge=10, le=100)
    mask_dilation: int = Field(default=15, ge=0, le=15)
    dialogue: str = Field(default="", max_length=1_500)
    language: str = Field(default="English", max_length=80)

    @model_validator(mode="after")
    def exactly_one_source(self) -> "CreativeTransformRequest":
        sources = [
            self.candidate_id, self.remote_filename, self.chain_clip_id,
            self.source_job_id,
        ]
        if sum(bool(value) for value in sources) != 1:
            raise ValueError("Choose exactly one video source")
        if self.mode == "water-simulation" and "strength" not in self.model_fields_set:
            self.strength = 1.2
        if self.mode == "water-simulation" and not self.prompt.strip():
            raise ValueError(
                "Water Simulation needs a concrete water-motion and interaction description"
            )
        if self.mode == "in-outpainting":
            if self.operation == "inpaint" and not self.mask_asset_id:
                raise ValueError("Inpainting requires an uploaded static or animated mask")
            if self.operation == "outpaint" and self.mask_asset_id:
                raise ValueError("Outpainting generates its canvas mask automatically")
        elif self.mask_asset_id:
            raise ValueError("A mask asset is only valid for In/Outpainting")
        if self.mode == "lipdub":
            if not self.dialogue.strip():
                raise ValueError("LipDub requires the exact desired dialogue")
            if not self.language.strip():
                raise ValueError("LipDub requires the desired spoken language")
        return self


class CinemagraphRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,47}$")
    asset_id: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=3, max_length=5_000)
    negative_prompt: str = Field(default="", max_length=2_000)
    strength: float = Field(default=1.0, ge=0.7, le=3.0)
    seed: int = Field(default=-1, ge=-1, le=2**63 - 2)


class ExportRequest(BaseModel):
    platform: Platform
    aspect_ratio: AspectRatio | None = None
    width: int | None = Field(default=None, ge=256, le=2160)
    height: int | None = Field(default=None, ge=256, le=2160)
    transition_seconds: float = Field(default=0.15, ge=0.0, le=1.0)
    captions: str = Field(default="", max_length=20_000)
    burn_captions: bool = True
    music_asset_id: str | None = None
    voiceover_asset_id: str | None = None
    original_audio_volume: float = Field(default=0.8, ge=0.0, le=2.0)
    music_volume: float = Field(default=0.18, ge=0.0, le=2.0)
    voiceover_volume: float = Field(default=1.0, ge=0.0, le=2.0)
    logo_asset_id: str | None = None
    logo_position: Literal[
        "top-left", "top-right", "bottom-left", "bottom-right"
    ] = "bottom-right"
    logo_width_percent: float = Field(default=14.0, ge=3.0, le=40.0)
    logo_opacity: float = Field(default=1.0, ge=0.1, le=1.0)
    logo_margin_percent: float = Field(default=3.0, ge=0.0, le=15.0)


class QueueResponse(BaseModel):
    prompt_id: str
    estimated_seconds: int
    job_id: str | None = None


class ApiMessage(BaseModel):
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
