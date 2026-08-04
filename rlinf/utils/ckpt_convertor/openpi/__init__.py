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

"""Consolidated OpenPI 0.5 checkpoint convertors.

Five layout convertors (``jax_to_openpi_rlinf``,
``openpi_pytorch_to_openpi_rlinf``, ``sft_to_openpi_rlinf``,
``openpi_rlinf_to_openpi_pytorch``, ``sft2deploy``) share one core.
``sft_to_openpi_rlinf`` reads its model architecture from the same OpenPI
TrainConfig used by SFT and eval. See
:mod:`rlinf.utils.ckpt_convertor.openpi.convert` for the unified ``--mode``
dispatcher and the package README for per-mode layouts and examples.
"""
