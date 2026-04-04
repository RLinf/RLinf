import torch
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.dreamzero.backbone.base_backbone import Backbone


class IdentityBackbone(Backbone):
    """
    This class allows pretraining the action head without depending on any backbone.
    That's why it's called "identity" — it preserves the action head to be a standalone trainable model.
    """

    def set_trainable_parameters(self, **kwargs):
        return

    def forward(self, backbone_input: BatchFeature) -> BatchFeature:
        backbone_input_first_value = next(iter(backbone_input.values()))
        B = backbone_input_first_value.shape[0]

        backbone_features = torch.empty(
            B, 1, 0, dtype=torch.float32, device=backbone_input_first_value.device
        )
        output_dict = {
            "backbone_features": backbone_features,
        }

        return BatchFeature(data=output_dict)

    def prepare_input(self, batch: dict) -> BatchFeature:
        """
        Args:
            batch: dict
                Must contain at least one key-value pair to inform the batch size.
                Expects the first dimension to be the batch size. See `forward`.
        """
        if "action" in batch:
            return BatchFeature(data={"action": batch["action"]})
        else:
            # at inference time, we have to use either state or video
            if "state" in batch:
                return BatchFeature(data={"state": batch["state"]})
            elif "video" in batch:
                # For video, it's tricky because it's a numpy array, which isn't compatible with BatchFeature's `to` method
                # So instead, we make it a tensor and return it
                video = batch["video"]
                video_tensor = torch.from_numpy(video)
                return BatchFeature(data={"video": video_tensor})
            else:
                return BatchFeature(data=batch)
