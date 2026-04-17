from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_serializer, field_validator


class EmbodimentTag(Enum):
    LIBERO_SIM = "libero_sim"
    OXE_DROID = "oxe_droid"


class DatasetStatisticalValues(BaseModel):
    max: np.ndarray = Field(..., description="Maximum values")
    min: np.ndarray = Field(..., description="Minimum values")
    mean: np.ndarray = Field(..., description="Mean values")
    std: np.ndarray = Field(..., description="Standard deviation")
    q01: np.ndarray = Field(..., description="1st percentile values")
    q99: np.ndarray = Field(..., description="99th percentile values")

    model_config = {"arbitrary_types_allowed": True}

    @field_serializer("*", when_used="json")
    def serialize_ndarray(self, v: np.ndarray) -> list:
        return v.tolist()  # type: ignore[return-value]

    @field_validator("*", mode="before")
    @classmethod
    def validate_ndarray(cls, v) -> np.ndarray:
        return np.array(v)


class DatasetStatistics(BaseModel):
    state: dict[str, DatasetStatisticalValues] = Field(..., description="State statistics")
    action: dict[str, DatasetStatisticalValues] = Field(..., description="Action statistics")


class VideoMetadata(BaseModel):
    resolution: tuple[int, int] = Field(..., description="Video resolution")
    channels: int = Field(..., description="Number of channels", gt=0)
    fps: float = Field(..., description="FPS", gt=0)


class StateActionMetadata(BaseModel):
    absolute: bool = Field(..., description="Absolute or relative")
    rotation_type: Optional[str] = Field(None, description="Rotation representation type")
    shape: tuple[int, ...] = Field(..., description="Tensor shape")
    continuous: bool = Field(..., description="Continuous variable")


class DatasetModalities(BaseModel):
    video: dict[str, VideoMetadata] = Field(..., description="Video metadata")
    state: dict[str, StateActionMetadata] = Field(..., description="State metadata")
    action: dict[str, StateActionMetadata] = Field(..., description="Action metadata")


class DatasetMetadata(BaseModel):
    """Metadata of the trainable dataset."""

    statistics: DatasetStatistics = Field(..., description="Statistics of the dataset")
    modalities: DatasetModalities = Field(..., description="Metadata of the modalities")
    embodiment_tag: EmbodimentTag = Field(..., description="Embodiment tag of the dataset")

