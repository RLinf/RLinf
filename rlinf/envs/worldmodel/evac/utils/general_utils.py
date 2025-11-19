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

import importlib
import logging
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.distributed as dist

mainlogger = logging.getLogger("mainlogger")


def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None


def load_checkpoints(model, model_cfg, ignore_mismatched_sizes=True):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        assert os.path.exists(pretrained_ckpt), (
            "Error: Pre-trained checkpoint NOT found at:%s" % pretrained_ckpt
        )
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")

        if "state_dict" in pl_sd.keys():
            state_dict = pl_sd["state_dict"]
            model_state_dict = model.state_dict()
            loaded_keys = list(state_dict.keys())
            original_loaded_keys = loaded_keys

            def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                ignore_mismatched_sizes,
            ):
                mismatched_keys = []
                if ignore_mismatched_sizes:
                    for checkpoint_key in loaded_keys:
                        model_key = checkpoint_key

                        if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape
                            != model_state_dict[model_key].shape
                        ):
                            mismatched_keys.append(
                                (
                                    checkpoint_key,
                                    state_dict[checkpoint_key].shape,
                                    model_state_dict[model_key].shape,
                                )
                            )
                            del state_dict[checkpoint_key]

                return mismatched_keys

            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes,
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            mainlogger.info(">>> mismatched_keys: %s" % mismatched_keys)
            mainlogger.info(">>> missing: %s" % missing)
            mainlogger.info(">>> unexpected: %s" % unexpected)
            mainlogger.info(
                ">>> Loaded weights from pretrained checkpoint: %s" % pretrained_ckpt
            )

        else:

            def _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                ignore_mismatched_sizes,
                rm_mismatched_weights=True,
            ):
                mismatched_keys = []
                if ignore_mismatched_sizes:
                    for checkpoint_key in loaded_keys:
                        model_key = checkpoint_key

                        if (
                            model_key in model_state_dict
                            and state_dict[checkpoint_key].shape
                            != model_state_dict[model_key].shape
                        ):
                            mismatched_keys.append(
                                (
                                    checkpoint_key,
                                    state_dict[checkpoint_key].shape,
                                    model_state_dict[model_key].shape,
                                )
                            )
                            if rm_mismatched_weights:
                                del state_dict[checkpoint_key]

                return mismatched_keys

            # deepspeed
            new_pl_sd = OrderedDict()

            if "module" in pl_sd:
                pl_sd = pl_sd["module"]
            for key in pl_sd.keys():
                new_pl_sd[key[16:]] = pl_sd[key]
            state_dict = new_pl_sd
            model_state_dict = model.state_dict()
            loaded_keys = list(state_dict.keys())
            original_loaded_keys = loaded_keys
            mismatched_keys = _find_mismatched_keys(
                state_dict,
                model_state_dict,
                original_loaded_keys,
                ignore_mismatched_sizes
                if not getattr(model_cfg, "auto_padzero_input_block", False)
                else True,
                False
                if getattr(model_cfg, "auto_padzero_input_block", False)
                else True,
            )

            if getattr(model_cfg, "auto_padzero_input_block", False):
                for k, _, _ in mismatched_keys:
                    if k.find("input_blocks.0.0.weight") >= 0:
                        sd_shape = state_dict[k].shape
                        model_sd_shape = model_state_dict[k].shape
                        if model_sd_shape[1] > sd_shape[1]:
                            padding_zeros = torch.zeros(
                                sd_shape[0], model_sd_shape[1] - sd_shape[1], 3, 3
                            ).to(dtype=state_dict[k].dtype, device=state_dict[k].device)
                            state_dict.update(
                                {k: torch.cat((state_dict[k], padding_zeros), dim=1)}
                            )
                        else:
                            state_dict.update(
                                {k: state_dict[k][:, : model_sd_shape.shape[1]]}
                            )

            if getattr(model_cfg, "auto_padrand_input_block", False):
                for k, _, _ in mismatched_keys:
                    if k.find("input_blocks.0.0.weight") >= 0:
                        sd_shape = state_dict[k].shape
                        model_sd_shape = model_state_dict[k].shape
                        if model_sd_shape[1] > sd_shape[1]:
                            torch.randn(
                                sd_shape[0], model_sd_shape[1] - sd_shape[1], 3, 3
                            ).to(dtype=state_dict[k].dtype, device=state_dict[k].device)
                            state_dict.update(
                                {k: torch.cat((state_dict[k], padding_zeros), dim=1)}
                            )
                        else:
                            state_dict.update(
                                {k: state_dict[k][:, : model_sd_shape.shape[1]]}
                            )

            missing, unexpected = model.load_state_dict(state_dict, strict=False)

            mainlogger.info(">>> mismatched_keys: %s" % mismatched_keys)
            mainlogger.info(">>> missing: %s" % missing)
            mainlogger.info(">>> unexpected: %s" % unexpected)
            mainlogger.info(
                ">>> Loaded weights from pretrained checkpoint: %s" % pretrained_ckpt
            )

    else:
        mainlogger.info(">>> Start training from scratch")

    return model


def set_logger(logfile, name="mainlogger"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.0e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """
    name: full name of source para
    para_list: partial name of target para
    """
    istarget = False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", {}))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except Exception:
        # evac relative import - try with evac prefix
        return getattr(
            importlib.import_module("rlinf.envs.evac_world_model." + module), cls
        )


def load_npz_from_dir(data_dir):
    data = [
        np.load(os.path.join(data_dir, data_name))["arr_0"]
        for data_name in os.listdir(data_dir)
    ]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)["arr_0"] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group("nccl", init_method="env://")


def zero_rank_print(s):
    if (not dist.is_initialized()) and (dist.is_initialized() and dist.get_rank() == 0):
        print("### " + s)
