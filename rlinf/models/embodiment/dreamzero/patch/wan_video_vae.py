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

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange, repeat
from groot.vla.model.dreamzero.modules import wan_video_vae as _groot_vae
from tqdm import tqdm

VideoVAE_ = _groot_vae.VideoVAE_
VideoVAE38_ = _groot_vae.VideoVAE38_

# The purpose of this patch is to modify the WanVideoVAE of DreamZeroPolicy
# to support processing multiple videos as a batch when the micro batch size
# is greater than 1 (previously, they could only be processed one by one), thereby accelerating the training process.


class WanVideoVAE(nn.Module):
    def __init__(self, z_dim=16, vae_pretrained_path: str | None = None):
        super().__init__()

        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, device="cuda")
        self.std = torch.tensor(std, device="cuda")
        self.scale = [self.mean, 1.0 / self.std]

        self.model = VideoVAE_(z_dim=z_dim).eval().requires_grad_(False)
        self.upsampling_factor = 8
        self.z_dim = z_dim
        self.vae_pretrained_path = vae_pretrained_path

    def build_1d_mask(self, length, left_bound, right_bound, border_width, device):
        x = torch.ones((length,), device=device)
        border = torch.arange(border_width, device=device) + 1
        if not left_bound:
            x[:border_width] = border / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(border / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, _, H, W = data.shape
        h = self.build_1d_mask(
            H, is_bound[0], is_bound[1], border_width[0], device=data.device
        )
        w = self.build_1d_mask(
            W, is_bound[2], is_bound[3], border_width[1], device=data.device
        )

        h = repeat(h, "H -> H W", H=H, W=W)
        w = repeat(w, "W -> H W", H=H, W=W)

        mask = torch.stack([h, w]).min(dim=0).values
        mask = rearrange(mask, "H W -> 1 1 1 H W")
        return mask

    @staticmethod
    def _tile_starts_1d(length: int, tile: int, stride: int) -> list[int]:
        if tile <= 0 or stride <= 0:
            raise ValueError("tile and stride must be positive")
        if length <= tile:
            return [0]
        last_start = length - tile
        starts = list(range(0, last_start + 1, stride))
        if starts[-1] != last_start:
            starts.append(last_start)
        return starts

    @staticmethod
    def _build_spatial_tile_tasks(
        height: int,
        width: int,
        size_h: int,
        size_w: int,
        stride_h: int,
        stride_w: int,
    ) -> list[tuple[int, int, int, int]]:
        hs = WanVideoVAE._tile_starts_1d(height, size_h, stride_h)
        ws = WanVideoVAE._tile_starts_1d(width, size_w, stride_w)
        return [(h, h + size_h, w, w + size_w) for h in hs for w in ws]

    def tiled_decode(self, hidden_states, tile_size, tile_stride):
        _, _, T, H, W = hidden_states.shape
        size_h, size_w = tile_size
        stride_h, stride_w = tile_stride

        tasks = self._build_spatial_tile_tasks(H, W, size_h, size_w, stride_h, stride_w)

        out_T = T * 4 - 3
        weight = torch.zeros(
            (1, 1, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        values = torch.zeros(
            (1, 3, out_T, H * self.upsampling_factor, W * self.upsampling_factor),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        for h, h_, w, w_ in tqdm(tasks, desc="VAE decoding"):
            hidden_states_batch = hidden_states[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.decode(hidden_states_batch, self.scale)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(
                    (size_h - stride_h) * self.upsampling_factor,
                    (size_w - stride_w) * self.upsampling_factor,
                ),
            ).to(dtype=hidden_states.dtype)

            target_h = h * self.upsampling_factor
            target_w = w * self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / torch.clamp(weight, min=1e-6)
        values = values.clamp_(-1, 1)
        return values

    def tiled_encode(self, video, tile_size, tile_stride):
        B, _, T, H, W = video.shape
        size_h = tile_size[0] * self.upsampling_factor
        size_w = tile_size[1] * self.upsampling_factor
        stride_h = tile_stride[0] * self.upsampling_factor
        stride_w = tile_stride[1] * self.upsampling_factor

        tasks = self._build_spatial_tile_tasks(H, W, size_h, size_w, stride_h, stride_w)

        out_T = (T + 3) // 4
        latent_H = H // self.upsampling_factor
        latent_W = W // self.upsampling_factor
        weight = torch.zeros(
            (B, 1, out_T, latent_H, latent_W),
            dtype=video.dtype,
            device=video.device,
        )
        values = torch.zeros(
            (B, self.z_dim, out_T, latent_H, latent_W),
            dtype=video.dtype,
            device=video.device,
        )

        scale = float(self.upsampling_factor)
        bw_h = max(0, int(round((size_h - stride_h) / scale)))
        bw_w = max(0, int(round((size_w - stride_w) / scale)))

        for h, h_, w, w_ in tqdm(tasks, desc="VAE encoding"):
            hidden_states_batch = video[:, :, :, h:h_, w:w_]
            hidden_states_batch = self.model.encode(hidden_states_batch, self.scale)

            mask = self.build_mask(
                hidden_states_batch,
                is_bound=(h == 0, h_ >= H, w == 0, w_ >= W),
                border_width=(bw_h, bw_w),
            ).to(dtype=video.dtype)

            target_h = h // self.upsampling_factor
            target_w = w // self.upsampling_factor
            values[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += hidden_states_batch * mask
            weight[
                :,
                :,
                :,
                target_h : target_h + hidden_states_batch.shape[3],
                target_w : target_w + hidden_states_batch.shape[4],
            ] += mask
        values = values / torch.clamp(weight, min=1e-6)
        return values

    def single_encode(self, video):
        x = self.model.encode(video, self.scale)
        x = x.clone()
        return x

    def single_decode(self, hidden_state):
        video = self.model.decode(hidden_state, self.scale)
        return video.clamp_(-1, 1)

    def encode(self, videos, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)
        if tiled:
            return self.tiled_encode(videos, tile_size, tile_stride)
        return self.single_encode(videos)

    def decode(
        self, hidden_states, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        if tiled:
            video = self.tiled_decode(hidden_states, tile_size, tile_stride)
        else:
            video = self.single_decode(hidden_states)
        return video

    @staticmethod
    def state_dict_converter():
        return WanVideoVAEStateDictConverter()


class WanVideoVAEStateDictConverter:
    def __init__(self):
        pass

    def from_civitai(self, state_dict):
        state_dict_ = {}
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        for name in state_dict:
            state_dict_["model." + name] = state_dict[name]
        return state_dict_


# Because monkey patching can break the inheritance chain,
# we need to further patch WanVideoVAE38, even though no modifications have been made.
# This could potentially be solved by improving our Patcher, but we'll leave that for next time.


class WanVideoVAE38(WanVideoVAE):
    def __init__(self, z_dim=48, dim=160, vae_pretrained_path: str | None = None):
        super(WanVideoVAE, self).__init__()

        mean = [
            -0.2289,
            -0.0052,
            -0.1323,
            -0.2339,
            -0.2799,
            0.0174,
            0.1838,
            0.1557,
            -0.1382,
            0.0542,
            0.2813,
            0.0891,
            0.1570,
            -0.0098,
            0.0375,
            -0.1825,
            -0.2246,
            -0.1207,
            -0.0698,
            0.5109,
            0.2665,
            -0.2108,
            -0.2158,
            0.2502,
            -0.2055,
            -0.0322,
            0.1109,
            0.1567,
            -0.0729,
            0.0899,
            -0.2799,
            -0.1230,
            -0.0313,
            -0.1649,
            0.0117,
            0.0723,
            -0.2839,
            -0.2083,
            -0.0520,
            0.3748,
            0.0152,
            0.1957,
            0.1433,
            -0.2944,
            0.3573,
            -0.0548,
            -0.1681,
            -0.0667,
        ]
        std = [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.4990,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.0600,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ]
        self.mean = torch.tensor(mean, device="cuda")
        self.std = torch.tensor(std, device="cuda")
        self.scale = [self.mean, 1.0 / self.std]

        self.model = VideoVAE38_(z_dim=z_dim, dim=dim).eval().requires_grad_(False)
        self.upsampling_factor = 16
        self.z_dim = z_dim
        self.vae_pretrained_path = vae_pretrained_path
