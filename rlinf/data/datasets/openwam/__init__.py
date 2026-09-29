# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OpenWAM native-dataset bridge for RLinf SFT."""

from .dataloader import build_openwam_sft_dataloader

__all__ = ["build_openwam_sft_dataloader"]
