"""
OpenPI 0.5 Extended Model Package.

PI0.5 with support for three forwarding modes:
- VLA: Standard flow matching action prediction
- VLM: Pure VLM with cross-entropy loss
- VLM+VLA: Reasoning text generation + action prediction
"""

from .configuration_pi05 import PI05Config, ForwardMode
from .modeling_pi05 import PI05ForConditionalGeneration, PI05FlowMatching, PI05Output
from .modeling_critic import (
    PI05CriticConfig,
    PI05ValueCritic,
    PI05QCritic,
    CriticOutput,
    ValueCriticModel,
    QCriticModel,
)
from .paligemma_with_multi_expert import PaliGemmaWithMultiExpertModel
from .processing_pi05 import PI05Processor
from .data_collator_pi05 import PI05DataCollator
from .static_kv_cache import StaticKVCache, left_to_right_align

# Re-export image processor from PI0
from ..openpi.processing_pi0 import PI0ImageProcessor

__all__ = [
    # Configuration
    "PI05Config",
    "PI05CriticConfig",
    "ForwardMode",
    # Model
    "PI05ForConditionalGeneration",
    "PI05FlowMatching",
    "PI05Output",
    "PaliGemmaWithMultiExpertModel",
    # Critic models
    "PI05ValueCritic",
    "PI05QCritic",
    "CriticOutput",
    "ValueCriticModel",
    "QCriticModel",
    # Processing
    "PI05Processor",
    "PI0ImageProcessor",
    # Data
    "PI05DataCollator",
    # Cache utilities
    "StaticKVCache",
    "left_to_right_align",
]
