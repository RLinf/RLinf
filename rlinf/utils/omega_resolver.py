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

import inspect

import torch
from omegaconf import OmegaConf

_REGISTERED = False


def _register_resolver(name, resolver):
    if "replace" in inspect.signature(OmegaConf.register_resolver).parameters:
        OmegaConf.register_resolver(name, resolver, replace=True)
    elif hasattr(OmegaConf, "register_new_resolver"):
        OmegaConf.register_new_resolver(name, resolver, replace=True)
    else:
        OmegaConf.register_resolver(name, resolver)


def omegaconf_register():
    global _REGISTERED
    if _REGISTERED:  # avoid duplicate
        return

    _register_resolver("multiply", lambda x, y: x * y)
    _register_resolver("int_div", lambda x, y: x // y)
    _register_resolver("subtract", lambda x, y: x - y)
    _register_resolver("not", lambda x: not bool(x))
    _register_resolver("torch.dtype", lambda dtype_name: getattr(torch, dtype_name))
    _REGISTERED = True


# register when import
omegaconf_register()
