# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import hydra
import torch

from rlinf.models import get_model
from toolkits.ckpt_convertor.fsdp_convertor.utils import (
    copy_model_config_and_code,
    get_model_save_helper,
    save_state_dict_sharded_safetensors,
)


@hydra.main(
    version_base="1.1", config_path="config", config_name="fsdp_model_convertor"
)
def main(cfg) -> None:
    model = get_model(cfg.model)

    model_dict = torch.load(cfg.convertor.ckpt_path)
    model.load_state_dict(model_dict)

    save_path = cfg.convertor.save_path

    model_save_helper_func = get_model_save_helper(cfg.model.model_type)

    if cfg.model.is_lora:
        if cfg.convertor.merge_lora_weighs:
            copy_model_config_and_code(
                model_path=cfg.model.model_path, save_path=save_path
            )
            model = model.merge_and_unload()
            model.save_pretrained(save_path, safe_serialization=True)

            model_state_dict = model.state_dict()
            if model_save_helper_func is not None:
                model_save_helper_func(model_state_dict, cfg.model, save_path)

        else:
            copy_model_config_and_code(
                model_path=cfg.model.model_path, save_path=save_path
            )
            # only save LoRA adapter
            save_path = os.path.join(save_path, "lora_adapter")
            model.save_pretrained(save_path, safe_serialization=True)
    else:
        copy_model_config_and_code(model_path=cfg.model.model_path, save_path=save_path)
        model_state_dict = model.state_dict()
        save_state_dict_sharded_safetensors(
            state_dict=model_state_dict, out_dir=save_path
        )

        if model_save_helper_func is not None:
            model_save_helper_func(model_state_dict, cfg.model, save_path)


if __name__ == "__main__":
    main()
