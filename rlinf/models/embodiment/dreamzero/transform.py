from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from rlinf.models.embodiment.dreamzero.schema import DatasetMetadata


class ModalityTransform(BaseModel, ABC):
    """Abstract class for transforming data modalities."""

    apply_to: list[str] = Field(..., description="The keys to apply the transform to.")
    training: bool = Field(default=True, description="Whether to apply the transform in training mode.")
    _dataset_metadata: DatasetMetadata | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def dataset_metadata(self) -> DatasetMetadata:
        assert self._dataset_metadata is not None, (
            "Dataset metadata is not set. Please call set_metadata() before calling apply()."
        )
        return self._dataset_metadata

    @dataset_metadata.setter
    def dataset_metadata(self, value: DatasetMetadata):
        self._dataset_metadata = value

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        self.dataset_metadata = dataset_metadata

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return self.apply(data)

    @abstractmethod
    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        pass

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class InvertibleModalityTransform(ModalityTransform):
    @abstractmethod
    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        pass


class IdentityModalityTransform(ModalityTransform):
    apply_to: list[str] = Field(default_factory=list, description="Ignored for identity transforms.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        return data


class ComposedModalityTransform(ModalityTransform):
    """Compose multiple modality transforms."""

    transforms: list[Any] = Field(..., description="The transforms to compose.")
    apply_to: list[str] = Field(default_factory=list, description="Ignored for composed transforms.")
    training: bool = Field(default=True, description="Whether to apply the transform in training mode.")

    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        for transform in self.transforms:
            transform.set_metadata(dataset_metadata)
            if hasattr(transform, "set_transform_pipeline"):
                getattr(transform, "set_transform_pipeline")(self.transforms)

    def set_per_horizon_statistics(self, per_horizon_stats: dict[str, dict[str, list]]):
        for transform in self.transforms:
            if hasattr(transform, "set_per_horizon_statistics"):
                transform.set_per_horizon_statistics(per_horizon_stats)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(self.transforms):
            try:
                data = transform(data)
            except Exception as e:
                raise ValueError(f"Error applying transform {i} to data: {e}") from e
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for i, transform in enumerate(reversed(self.transforms)):
            if isinstance(transform, InvertibleModalityTransform) or hasattr(transform, "unapply"):
                try:
                    data = transform.unapply(data)
                except Exception as e:
                    step = len(self.transforms) - i - 1
                    raise ValueError(f"Error unapplying transform {step} to data: {e}") from e
        return data

    def train(self):
        for transform in self.transforms:
            transform.train()
        self.training = True

    def eval(self):
        for transform in self.transforms:
            transform.eval()
        self.training = False

