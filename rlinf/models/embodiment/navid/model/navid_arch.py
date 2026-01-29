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

from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from rlinf.models.embodiment.navid.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_PATCH_TOKEN,
    IAMGE_SEPARATOR,
    IGNORE_INDEX,
    IMAGE_END_TOKEN,
    IMAGE_START_TOKEN,
    IMAGE_TOKEN_INDEX,
    NAVIGATION_IDENTIFIER,
    NAVIGATION_SPECIAL_TOKEN,
    VIDEO_END_SPECIAL_TOKEN,
    VIDEO_START_SPECIAL_TOKEN,
)

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector


class NaVidMetaModel:
    def __init__(self, config):
        super(NaVidMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None, max_token=2048):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        self.config.mm_vision_tower = vision_tower
        self.config.image_processor = getattr(model_args, "image_processor", None)

        vision_tower = build_vision_tower(model_args)

        if fsdp is not None and len(fsdp) > 0:
            self.vision_tower = [vision_tower]
        else:
            self.vision_tower = vision_tower

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.max_token = max_token

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )

    def initialize_attention_modules(self, model_args, for_eval=False):
        self.config.compress_type = getattr(model_args, "compress_type", None)


class NaVidMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, prompts=None, image_counts=None, long_video=False):
        if long_video:
            # use pre-computed features
            image_features = images
        else:
            # (n, 3, 224, 224)
            image_features = self.get_model().get_vision_tower()(images)
            # (n, 257, 1408)

        image_features, video_or_not, nav_or_not = self.vlm_attention(
            image_features,
            prompts=prompts,
            image_counts=image_counts,
            long_video=long_video,
        )
        return image_features, video_or_not, nav_or_not

    def vlm_attention(
        self, image_features, prompts=None, image_counts=None, long_video=False
    ):
        compress_type = self.config.compress_type
        compress_grid_sizes = {"grid:2": 4, "grid:4": 16, "mean": 1}

        nav_size = compress_grid_sizes.get(compress_type)
        if nav_size is None:
            raise ValueError(f"Unsupported compress type: {compress_type}")

        if image_counts is None:
            assert len(image_features) == len(prompts), (
                f"Size mismatch! image_features: {len(image_features)}, prompts: {len(prompts)}"
            )
        else:
            assert len(prompts) == len(image_counts), (
                f"Size mismatch! prompts: {len(prompts)}, image_counts: {len(image_counts)}"
            )

        img_feat_lst = []
        video_or_not = []
        nav_or_not = []
        total_count = 0

        for _idx, prompt in enumerate(prompts):
            assert isinstance(prompt, list), (
                f"Prompt should be a list, but got {type(prompt)}"
            )

            if image_counts is None:
                img_feat_prompt = image_features[_idx, None]
            else:
                img_feat_prompt = image_features[
                    total_count : total_count + image_counts[_idx]
                ]
                total_count += image_counts[_idx]

            is_navigation = NAVIGATION_IDENTIFIER in prompt[0]
            if is_navigation:
                if image_counts is None or image_counts[_idx] < 1 or len(prompt) != 1:
                    raise ValueError("[Navigation] wrong")

            if (
                self.config.mm_vision_select_feature == "patch"
                and img_feat_prompt.shape[1] % 2 == 1
            ):
                img_feat_prompt = img_feat_prompt[:, 1:]

            final_token, final_token_nav = self.token_generation(
                img_feat_prompt,
                image_counts=None if image_counts is None else image_counts[_idx],
                navigation=is_navigation,
            )

            if is_navigation and final_token_nav is None:
                raise ValueError("[Navigation] wrong")

            final_token = (
                final_token[None].expand(len(prompt), -1, -1, -1).flatten(1, 2)
            )
            if image_counts is not None:
                if is_navigation:
                    final_token_nav = (
                        final_token_nav[None]
                        .expand(len(prompt), -1, -1, -1)
                        .flatten(1, 2)
                    )
                    assert (
                        final_token_nav.shape[0] == 1
                        and final_token_nav.shape[1] == 64
                        and final_token.shape[0] == 1
                    )
                    nav_or_not.append(final_token_nav)
                else:
                    nav_or_not.append(None)

                if image_counts[_idx] == 1:
                    if is_navigation:
                        assert final_token.shape[1] == nav_size
                        video_or_not.append(True)
                    else:
                        assert final_token.shape[1] == 64
                        video_or_not.append(False)
                else:
                    video_or_not.append(True)
            else:
                assert final_token.shape[1] == 64
                video_or_not.append(False)
                nav_or_not.append(None)

            img_feat_lst.append(final_token)

        return img_feat_lst, video_or_not, nav_or_not

    def token_generation(self, vis_embed, image_counts=None, navigation=False):
        def process_grid(vis_embed, grid_size):
            cur_shape = int(vis_embed.shape[1] ** 0.5)
            assert grid_size > 1, (
                f"Grid size should be larger than 1, but got {grid_size}"
            )
            vis_embed = vis_embed.reshape(vis_embed.shape[0], cur_shape, cur_shape, -1)
            grid_stride = cur_shape // grid_size
            vis_embed = F.avg_pool2d(
                vis_embed.permute(0, 3, 1, 2),
                padding=0,
                kernel_size=grid_stride,
                stride=grid_stride,
            )
            return vis_embed.permute(0, 2, 3, 1).flatten(1, 2)

        grid_size = int(self.config.compress_type.split("grid:")[-1])
        if image_counts is None or (image_counts == 1 and not navigation):
            vis_embed = process_grid(vis_embed, 8)
        elif navigation:
            vis_embed_nav = vis_embed[-1:]
            vis_embed_nav = process_grid(vis_embed_nav, 8)
            vis_embed = process_grid(vis_embed, grid_size)
        else:
            vis_embed = process_grid(vis_embed, grid_size)

        vis_embed = self.get_model().mm_projector(vis_embed)
        vis_embed_nav = (
            self.get_model().mm_projector(vis_embed_nav) if navigation else None
        )

        return vis_embed, vis_embed_nav

    def update_prompt(self, prompts=None):
        self.prompts = prompts

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, prompts=None
    ):
        if "grid" in self.config.compress_type:
            grid_size = int(self.config.compress_type.split("grid:")[-1])
            if grid_size == 2:
                nav_size = 4
            elif grid_size == 4:
                nav_size = 16
            else:
                raise ValueError
        elif "mean" in self.config.compress_type:
            nav_size = 1
        else:
            raise ValueError

        if prompts is None and hasattr(self, "prompts"):
            prompts = self.prompts

        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                past_key_values is not None
                and vision_tower is not None
                and images is not None
                and input_ids.shape[1] == 1
            ):
                attention_mask = torch.ones(
                    (attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
            return input_ids, attention_mask, past_key_values, None, labels

        # pre-process images for long video
        if images[0].shape[-1] > 1000:
            long_video = True
        else:
            long_video = False

        if type(images) is list or images.ndim == 5:
            # not reseshape for long video
            if not long_video:
                images = [
                    image if len(image.shape) == 4 else image.unsqueeze(0)
                    for image in images
                ]
            image_counts = [image.shape[0] for image in images]
            concat_images = torch.cat(images, dim=0)
            image_features, video_or_not, nav_or_not = self.encode_images(
                concat_images, prompts, image_counts, long_video=long_video
            )
        else:
            image_features, video_or_not, nav_or_not = self.encode_images(
                images, prompts, long_video=long_video
            )

        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                if isinstance(image_features, list):
                    cur_image_features = image_features[cur_image_idx][0]
                else:
                    cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(
                    cur_input_ids[:half_len]
                )
                cur_input_embeds_2 = self.get_model().embed_tokens(
                    cur_input_ids[half_len:]
                )
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2],
                    dim=0,
                )
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            if not long_video:
                token_idx = 0
                while image_token_indices.numel() > 0:
                    if isinstance(image_features, list):
                        cur_image_features = image_features[cur_image_idx][token_idx]
                    else:
                        cur_image_features = image_features[cur_image_idx]
                    image_token_start = image_token_indices[0]

                    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                        self.config, "mm_use_im_start_end", False
                    ):
                        raise ValueError("wrong")
                        cur_new_input_embeds.append(
                            self.get_model()
                            .embed_tokens(cur_input_ids[: image_token_start - 1])
                            .detach()
                        )
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(
                                cur_input_ids[image_token_start - 1 : image_token_start]
                            )
                        )
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(
                                cur_input_ids[
                                    image_token_start + 1 : image_token_start + 2
                                ]
                            )
                        )
                        if labels is not None:
                            cur_new_labels.append(cur_labels[:image_token_start])
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0],),
                                    IGNORE_INDEX,
                                    device=labels.device,
                                    dtype=labels.dtype,
                                )
                            )
                            cur_new_labels.append(
                                cur_labels[image_token_start : image_token_start + 1]
                            )
                            cur_labels = cur_labels[image_token_start + 2 :]
                    else:
                        if (
                            nav_or_not[cur_image_idx] is None
                            and video_or_not[cur_image_idx] is False
                        ):
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[:image_token_start]
                                )
                            )
                            cur_new_input_embeds.append(cur_image_features)
                            assert cur_image_features.shape[0] == 64

                        elif (
                            nav_or_not[cur_image_idx] is None
                            and video_or_not[cur_image_idx] is True
                        ):
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[:image_token_start]
                                )
                            )
                            seperator_token = self.get_model().embed_tokens(
                                cur_input_ids[image_token_start - 1, None]
                            )
                            video_index = 0
                            assert len(cur_image_features) % nav_size == 0

                            for ii in range(int(len(cur_image_features) / nav_size)):
                                cur_new_input_embeds.append(
                                    cur_image_features[
                                        video_index : video_index + nav_size
                                    ]
                                )
                                if ii == (len(cur_image_features) / nav_size) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += nav_size
                        else:
                            assert video_or_not[cur_image_idx] is True
                            assert token_idx == 0
                            assert nav_or_not[cur_image_idx][token_idx].shape[0] == 64
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[:image_token_start]
                                )
                            )
                            seperator_token = self.get_model().embed_tokens(
                                cur_input_ids[image_token_start - 1, None]
                            )
                            video_index = 0
                            assert len(cur_image_features) % nav_size == 0
                            for ii in range(int(len(cur_image_features) / nav_size)):
                                cur_new_input_embeds.append(
                                    cur_image_features[
                                        video_index : video_index + nav_size
                                    ]
                                )
                                if ii == (len(cur_image_features) / nav_size) - 1:
                                    break
                                cur_new_input_embeds.append(seperator_token)
                                video_index += nav_size
                            cur_new_input_embeds.append(
                                self.get_model().embed_tokens(
                                    cur_input_ids[
                                        image_token_start + 1 : image_token_start + 3
                                    ]
                                )
                            )
                            cur_new_input_embeds.append(
                                nav_or_not[cur_image_idx][token_idx]
                            )

                        if labels is not None:
                            if (
                                nav_or_not[cur_image_idx] is None
                                and video_or_not[cur_image_idx] is False
                            ):
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full(
                                        (cur_image_features.shape[0],),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_labels = cur_labels[image_token_start + 1 :]
                            elif (
                                nav_or_not[cur_image_idx] is None
                                and video_or_not[cur_image_idx] is True
                            ):
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full(
                                        (cur_image_features.shape[0],),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_new_labels.append(
                                    torch.full(
                                        (
                                            int(
                                                cur_image_features.shape[0] / nav_size
                                                - 1
                                            ),
                                        ),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_labels = cur_labels[image_token_start + 1 :]
                            else:
                                cur_new_labels.append(cur_labels[:image_token_start])
                                cur_new_labels.append(
                                    torch.full(
                                        (cur_image_features.shape[0],),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_new_labels.append(
                                    torch.full(
                                        (
                                            int(
                                                cur_image_features.shape[0] / nav_size
                                                - 1
                                            ),
                                        ),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_new_labels.append(
                                    torch.full(
                                        (
                                            nav_or_not[cur_image_idx][token_idx].shape[
                                                0
                                            ]
                                            + 2,
                                        ),
                                        IGNORE_INDEX,
                                        device=labels.device,
                                        dtype=labels.dtype,
                                    )
                                )
                                cur_labels = cur_labels[image_token_start + 3 :]

                    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                        self.config, "mm_use_im_start_end", False
                    ):
                        raise ValueError("wrong")
                    else:
                        if nav_or_not[cur_image_idx] is not None:
                            cur_input_ids = cur_input_ids[image_token_start + 3 :]
                        else:
                            cur_input_ids = cur_input_ids[image_token_start + 1 :]
                    image_token_indices = torch.where(
                        cur_input_ids == IMAGE_TOKEN_INDEX
                    )[0]
                    token_idx += 1

                # changle image idx after processing one sample
                cur_image_idx += 1
                if cur_input_ids.numel() > 0:
                    if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
                        self.config, "mm_use_im_start_end", False
                    ):
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids).detach()
                        )
                    else:
                        cur_new_input_embeds.append(
                            self.get_model().embed_tokens(cur_input_ids)
                        )
                    if labels is not None:
                        cur_new_labels.append(cur_labels)
                cur_new_input_embeds = [
                    x.to(device=self.device) for x in cur_new_input_embeds
                ]
                cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    cur_new_labels = torch.cat(cur_new_labels, dim=0)
                    assert cur_new_input_embeds.shape[0] == cur_new_labels.shape[0]
                    new_labels.append(cur_new_labels)
            else:
                cur_new_input_embeds = torch.Tensor(
                    len(cur_input_ids), self.config.hidden_size
                ).to(dtype=self.dtype, device=self.device)
                text_token_indices = torch.where(cur_input_ids != IMAGE_TOKEN_INDEX)[0]
                if (
                    not self.training
                    and self.get_model().embed_tokens.weight.device
                    != cur_input_ids.device
                ):
                    model_device = self.get_model().embed_tokens.weight.device
                    data_device = cur_input_ids.device
                    cur_input_ids_text = cur_input_ids[text_token_indices].to(
                        device=model_device
                    )
                    cur_new_input_embeds[text_token_indices] = (
                        self.get_model()
                        .embed_tokens(cur_input_ids_text)
                        .to(device=data_device)
                    )
                else:
                    cur_new_input_embeds[text_token_indices] = (
                        self.get_model().embed_tokens(cur_input_ids[text_token_indices])
                    )
                cur_image_features = image_features[cur_image_idx]
                cur_new_input_embeds[image_token_indices] = cur_image_features
                new_input_embeds.append(cur_new_input_embeds)
                if labels is not None:
                    new_labels.append(cur_labels)
                cur_image_idx += 1

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat(
                        (
                            cur_new_label,
                            torch.full(
                                (max_len - cur_new_label.shape[0],),
                                IGNORE_INDEX,
                                dtype=cur_new_label.dtype,
                                device=cur_new_label.device,
                            ),
                        ),
                        dim=0,
                    )
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(
                    attention_mask, _new_labels, new_labels
                ):
                    new_attn_mask_pad_left = torch.full(
                        (cur_new_labels.shape[0] - labels.shape[1],),
                        True,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    new_attn_mask_pad_right = torch.full(
                        (cur_new_labels_align.shape[0] - cur_new_labels.shape[0],),
                        False,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    cur_new_attention_mask = torch.cat(
                        (
                            new_attn_mask_pad_left,
                            cur_attention_mask,
                            new_attn_mask_pad_right,
                        ),
                        dim=0,
                    )
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels = torch.stack(new_labels, dim=0)

            # only used for right padding in tokenlizer
            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full(
                    (
                        attention_mask.shape[0],
                        new_input_embeds.shape[1] - input_ids.shape[1],
                    ),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat(
                    (new_attn_mask_pad_left, attention_mask), dim=1
                )
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        tokenizer.add_tokens(
            [
                VIDEO_START_SPECIAL_TOKEN,
                VIDEO_END_SPECIAL_TOKEN,
                IMAGE_START_TOKEN,
                IMAGE_END_TOKEN,
                NAVIGATION_SPECIAL_TOKEN,
                IAMGE_SEPARATOR,
            ],
            special_tokens=True,
        )
        self.resize_token_embeddings(len(tokenizer))
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
