from abc import ABC, abstractmethod

from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class ActionHead(ABC, nn.Module):
    def __init__(self):
        super(ActionHead, self).__init__()

    @abstractmethod
    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        pass

    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        num_action_samples: int = 1,
        inference_batch_size: int = 32,
    ) -> BatchFeature:
        # Used for predicting actions during inference
        # By default, the action head does the same thing as a normal forward pass
        return self.forward(backbone_output, action_input)

    def prepare_input(self, batch: dict) -> BatchFeature:
        pass

    def set_override_kwargs(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self.config, key, value)
            setattr(self, key, value)
