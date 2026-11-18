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

"""Fake OpenWAM dataset readers shared by the data and SFT worker tests.

The ``openwam`` package is not installed in the unit-test environment, so these
register a stand-in ``openwam.dataloader.registry.build_dataset`` and write the
checkpoint ``config.yaml`` the RLinf loader reads.
"""

from types import ModuleType

import torch
from omegaconf import OmegaConf


def openwam_fake_dataset_cfg(tmp_path, monkeypatch, lengths):
    """Register a fake ``openwam.dataloader.registry.build_dataset``."""
    import sys

    class _FakeDataset(torch.utils.data.Dataset):
        def __init__(self, root, length):
            self.root = root
            self.length = length

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            return {"root": self.root, "index": index}

    calls = []

    def fake_build_dataset(dl_cfg, split):
        calls.append((dl_cfg.dataset_dir, split))
        return _FakeDataset(dl_cfg.dataset_dir, lengths[dl_cfg.dataset_dir])

    registry = ModuleType("openwam.dataloader.registry")
    registry.build_dataset = fake_build_dataset
    monkeypatch.setitem(sys.modules, "openwam", ModuleType("openwam"))
    monkeypatch.setitem(
        sys.modules, "openwam.dataloader", ModuleType("openwam.dataloader")
    )
    monkeypatch.setitem(sys.modules, "openwam.dataloader.registry", registry)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.yaml").write_text(
        "dataloader:\n  type: libero\n  dataset_dir: /unused\n  num_frames: 33\n"
    )
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"model_path": str(ckpt)},
                "micro_batch_size": 2,
                "seed": 3,
            },
            "data": {},
        }
    )
    return cfg, calls


def openwam_fake_mixture_cfg(tmp_path, monkeypatch, override_libero=None):
    """Register a fake ``build_dataset`` that records the mixture config it gets."""
    import sys

    class _FakeMixture(torch.utils.data.Dataset):
        def __init__(self):
            self.epochs = []

        def __len__(self):
            return 6

        def __getitem__(self, index):
            # The epoch this process's copy of the dataset was last set to;
            # loader workers report what they inherited when they were forked.
            return {"index": index, "epoch": self.epochs[-1] if self.epochs else 0}

        def set_epoch(self, epoch):
            self.epochs.append(int(epoch))

    seen = []
    built = []

    def fake_build_dataset(dl_cfg, split):
        seen.append((OmegaConf.to_container(dl_cfg.datasets), split))
        built.append(_FakeMixture())
        return built[-1]

    registry = ModuleType("openwam.dataloader.registry")
    registry.build_dataset = fake_build_dataset
    monkeypatch.setitem(sys.modules, "openwam", ModuleType("openwam"))
    monkeypatch.setitem(
        sys.modules, "openwam.dataloader", ModuleType("openwam.dataloader")
    )
    monkeypatch.setitem(sys.modules, "openwam.dataloader.registry", registry)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.yaml").write_text(
        "dataloader:\n"
        "  type: mixture\n"
        "  datasets:\n"
        "    robocoin: {enabled: true, dataset_dir: /train-host/robocoin}\n"
        "    oxe_droid: {enabled: false, dataset_dir: /train-host/oxe_droid}\n"
        "    libero: {enabled: true, dataset_dir: /train-host/libero}\n"
    )
    data = {"num_workers": 0}
    if override_libero is not None:
        data["openwam"] = {"datasets": {"libero": {"dataset_dir": override_libero}}}
    cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"model_path": str(ckpt)},
                "micro_batch_size": 2,
                "seed": 3,
            },
            "data": data,
        }
    )
    return cfg, seen, built
