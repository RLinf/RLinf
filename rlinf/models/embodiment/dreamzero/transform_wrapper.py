from __future__ import annotations

import ast
from typing import Any

import numpy as np
import torch
from pydantic import Field

from rlinf.models.embodiment.dreamzero.transform import InvertibleModalityTransform


def rlinf_collate(features: list[dict], tokenizer, num_views=3, embodiment_tag_mapping=None) -> dict:
    if embodiment_tag_mapping is None:
        embodiment_tag_mapping = {}
    norm_mapping = {k: int(v) for k, v in embodiment_tag_mapping.items()}
    batch = {}
    keys = features[0].keys()

    for key in keys:
        if key == "text":
            output_values = []
            for elem in features:
                elem_id = int(elem["embodiment_id"])
                item = elem[key]
                try:
                    parsed = ast.literal_eval(item)
                    processed = str(parsed[0]) if isinstance(parsed, (list, tuple)) else str(parsed)
                except (ValueError, SyntaxError, TypeError):
                    processed = str(item)

                if elem_id == norm_mapping.get("libero_sim"):
                    processed = (
                        "A multi-view video shows that a robot "
                        + processed.lower()
                        + " The video is split into two horizontal views: "
                        + "the left view shows the exterior camera and the right view shows the wrist camera. "
                        + "The robot "
                        + processed.lower()
                    )
                elif elem_id == norm_mapping.get("oxe_droid"):
                    processed = (
                        "A multi-view video shows that a robot "
                        + processed.lower()
                        + " The video is split into three views: The top view shows the camera view from the robot's wrist, "
                        + "the bottom-left view shows the camera view from the left exterior camera, and the bottom-right view "
                        + "shows the camera view from the right exterior camera. During training, one of the two bottom exterior "
                        + "views may be a black screen (dropped view). The robot "
                        + processed.lower()
                    )
                else:
                    raise ValueError(f"Embodiment ID {elem_id} not supported.")
                output_values.append(processed)
            ids, mask = tokenizer(output_values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask"] = mask
        elif key == "text_negative":
            values = [elem[key] for elem in features]
            ids, mask = tokenizer(values, return_mask=True, add_special_tokens=True)
            batch[key] = ids
            batch["text_attention_mask_negative"] = mask
        else:
            values = [elem[key] for elem in features]
            batch[key] = torch.from_numpy(np.stack(values))
    return batch


class RlinfDreamTransform(InvertibleModalityTransform):
    apply_to: list[str] = Field(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from groot.vla.model.dreamzero.transform.dreamzero_cotrain import (
            DreamTransform as _DzDreamTransform,
        )

        self._inner = _DzDreamTransform(**kwargs)

    def set_metadata(self, dataset_metadata):
        self._inner.set_metadata(dataset_metadata)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        import groot.vla.model.dreamzero.transform.dreamzero_cotrain as dz_cotrain

        old_collate = dz_cotrain.collate
        dz_cotrain.collate = rlinf_collate
        try:
            return self._inner.apply(data)
        finally:
            dz_cotrain.collate = old_collate

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        return self._inner.unapply(data)

