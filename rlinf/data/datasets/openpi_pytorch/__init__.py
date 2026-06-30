# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Eval-processor registry for the ``openpi_pytorch`` model.

:func:`get_eval_processor` maps an environment name to its
:class:`EvalProcessor` subclass, mirroring ``get_openpi_config`` in
``rlinf.models.embodiment.openpi.dataconfig``. The registry values are lazy
loaders (thunks) so that importing this package stays cheap: the heavy
``behavior`` dataset chain (``lerobot`` / ``datasets`` / ``huggingface_hub``,
pulled in by ``behavior/__init__.py``) is only imported when an env's processor
is actually requested. The loaders therefore import ``behavior.processing``
directly, never the ``behavior`` package.
"""

from __future__ import annotations

from rlinf.data.datasets.openpi_pytorch.eval_processor import EvalProcessor


def _load_behavior_processor():
    from rlinf.data.datasets.openpi_pytorch.behavior.processing import (
        BehaviorEvalProcessor,
    )

    return BehaviorEvalProcessor


# env name -> zero-arg loader returning the EvalProcessor subclass.
_EVAL_PROCESSORS = {
    "behavior": _load_behavior_processor,
}


def get_eval_processor(
    env_type,
    norm_stats,
    tokenizer,
    *,
    action_chunk,
    action_env_dim=23,
    model_action_dim=32,
    image_resolution=(224, 224),
) -> EvalProcessor:
    """Build the eval processor registered for ``env_type``.

    The keyword arguments mirror the env processors' constructors so the model
    factory can forward them unchanged. Raises ``ValueError`` (with a close-match
    suggestion) when no processor is registered for ``env_type``.
    """
    if env_type not in _EVAL_PROCESSORS:
        import difflib

        closest = difflib.get_close_matches(
            env_type, _EVAL_PROCESSORS.keys(), n=1, cutoff=0.0
        )
        hint = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Eval processor for env '{env_type}' not found.{hint}")
    processor_cls = _EVAL_PROCESSORS[env_type]()
    return processor_cls(
        norm_stats,
        tokenizer,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
        model_action_dim=model_action_dim,
        image_resolution=image_resolution,
    )


__all__ = ["EvalProcessor", "get_eval_processor"]
