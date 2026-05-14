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

import torch
from groot.vla.model.dreamzero.modules.wan2_1_submodule import sinusoidal_embedding_1d

# This patch is a minimal modification to DreamZero's CausalWanModel._forward_train
# to disable gradient checkpointing when the micro batch size is greater than 1
# (due to a PyTorch bug in this scenario). Registered in get_model.


def _forward_train(
    self,
    x,
    timestep,
    timestep_action,
    context,
    seq_len,
    clean_x=None,
    aug_t=None,
    y=None,
    clip_feature=None,
    action=None,
    state=None,
    embodiment_id=None,
):
    if self.model_type == "i2v":
        assert clip_feature is not None and y is not None

    if y is not None and self.concat_first_frame_latent:
        x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

    x = self.patch_embedding(x)
    grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
    freqs = self._create_freqs(
        grid_size=grid_size,
        start_frame=0,
    )

    x = x.flatten(start_dim=2).transpose(1, 2)
    assert x.shape[1] == seq_len

    B = x.shape[0]
    F = timestep.shape[1]

    if action is not None:
        embodiment_id = (
            torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
        )
        action_features = self.action_encoder(action, timestep_action, embodiment_id)
        action_length = action_features.shape[1]
        state_features = self.state_encoder(state, embodiment_id)
        action_register = torch.cat([action_features, state_features], dim=1)
        action_register_length = action_register.shape[1]
        x = torch.cat([x, action_register], dim=1)
    else:
        action_features = None
        action_length = None
        state_features = None
        action_register = None
        action_register_length = None

    timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
    timestep_original = timestep.clone()

    if action is not None:
        assert timestep_action is not None
        assert state_features is not None
        stride = timestep_action.shape[1] // state_features.shape[1]
        timestep_state = timestep_action[:, ::stride]
        timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

    e = self.time_embedding(
        sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x)
    )
    e = e.unflatten(dim=0, sizes=(B, -1))
    e0 = self.time_projection(e)
    e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

    assert context.shape[1] == self.text_len
    context = self.text_embedding(context)
    if clip_feature is not None:
        clip_embedding = self.img_emb(clip_feature)
        context = torch.cat([clip_embedding, context], dim=1)

    if clean_x is not None:
        if y is not None and self.concat_first_frame_latent:
            clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
        clean_x = self.patch_embedding(clean_x)
        clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
        assert clean_x.shape[1] == seq_len

        x = torch.cat([clean_x, x], dim=1)

        if aug_t is None:
            aug_t = torch.zeros_like(timestep_original)

        e_clean = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x)
        )
        e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
        e0_clean = self.time_projection(e_clean)
        e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
        e0 = torch.cat([e0_clean, e0], dim=1)

    kwargs = {
        "e": e0,
        "freqs": freqs,
        "freqs_action": self.freqs_action,
        "freqs_state": self.freqs_state,
        "action_register_length": action_register_length,
        "context": context,
        "is_tf": clean_x is not None,
    }

    def create_custom_forward(module):
        def custom_forward(*inputs, **kwargs):
            outputs, updated_kv_cache = module(*inputs, **kwargs)
            assert updated_kv_cache is None
            return outputs

        return custom_forward

    for block in self.blocks:
        use_ckpt = (
            torch.is_grad_enabled()
            and self.gradient_checkpointing
            and not (action_register_length is not None and x.shape[0] > 1)
        )

        if use_ckpt:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                **kwargs,
                use_reentrant=False,
            )
        else:
            x, _ = block(x, **kwargs)

    if clean_x is not None:
        x = x[:, clean_x.shape[1] :]

    if action is not None:
        action_noise_pred = x[:, seq_len : seq_len + action_length]
        action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
    else:
        action_noise_pred = None

    x_video = x[:, :seq_len]
    e_video = e[:, :seq_len]
    x_video = self.head(x_video, e_video.unsqueeze(2))
    video_noise_pred = self.unpatchify(x_video, grid_size)
    return video_noise_pred, action_noise_pred
