from abc import ABC, abstractmethod

from torch import nn
from transformers.feature_extraction_utils import BatchFeature


class Backbone(ABC, nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()

    @abstractmethod
    def forward(self, backbone_input: BatchFeature) -> BatchFeature:
        pass

    def prepare_input(self, batch: dict) -> BatchFeature:
        pass
