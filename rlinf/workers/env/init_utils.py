# Copyright 2026 The RLinf Authors.
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

"""Helpers for env worker initialization."""


def resolve_env_classes(cfg, enable_eval: bool, get_env_cls_fn):
    """Resolve train/eval env classes, skipping eval imports when evaluation is off."""
    train_env_cls = get_env_cls_fn(cfg.env.train.env_type, cfg.env.train)
    eval_env_cls = None
    if enable_eval:
        eval_env_cls = get_env_cls_fn(cfg.env.eval.env_type, cfg.env.eval)
    return train_env_cls, eval_env_cls
