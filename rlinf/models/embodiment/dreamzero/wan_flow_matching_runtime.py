from __future__ import annotations

from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any

import torch
from einops import rearrange
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)

if TYPE_CHECKING:
    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import (
        WANPolicyHead,
    )


@dataclass
class LazyJointVideoActionIntermediate:
    action_input: BatchFeature
    state_features: torch.Tensor
    embodiment_id: torch.Tensor
    videos: torch.Tensor
    prompt_embs: list[torch.Tensor]
    image: torch.Tensor
    clip_feas: torch.Tensor
    ys: torch.Tensor
    noise_obs: torch.Tensor
    noise_action: torch.Tensor
    batch_size: int
    frame_seqlen: int
    seq_len: int
    timings: dict[str, Any]


class WANEncoderRuntime:
    def __init__(self, head: WANPolicyHead):
        self.head = head

    def run(
        self,
        action_input: BatchFeature,
        latent_video: torch.Tensor | None = None,
    ) -> LazyJointVideoActionIntermediate:
        head = self.head
        start_time = time.perf_counter()
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)

        data = action_input
        videos = data["images"]
        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")
        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=head.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)
            videos = videos.reshape(b * t, c, h, w)
            videos = head.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=head.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        if head.language is None:
            print("language is None, reset current_start_frame to 0")
            head.language = data["text"]
            head.current_start_frame = 0
        elif not torch.equal(head.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            head.current_start_frame = 0
            head.language = data["text"]
        elif videos.shape[2] == 1:
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            head.current_start_frame = 0
        elif head.current_start_frame >= head.model.local_attn_size:
            print("current_start_frame >= local_attn_size, reset current_start_frame to 0")
            head.current_start_frame = 0

        if head.ip_rank == 0:
            print("videos shape", videos.shape, head.num_frames)

        start_text_encoder_event.record()
        text_inputs = head._prepare_text_inputs(data)
        prompt_embs = [head.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]
        end_text_encoder_event.record()

        start_image_encoder_event.record()
        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if head.current_start_frame == 0:
            clip_feas, ys, image = head.encode_image(image, head.num_frames, height, width)
            head.clip_feas = clip_feas.to(dtype=image.dtype)
            head.ys = ys.to(dtype=image.dtype)

        assert head.clip_feas is not None and head.ys is not None, "clip_feas and ys must be set"
        end_image_encoder_event.record()

        start_vae_event.record()
        if latent_video is not None and head.current_start_frame != 0:
            image = latent_video
            if head.ip_rank == 0:
                print("image shape@@", image.shape)
        elif head.current_start_frame != 0:
            if (videos.shape[2] - 1) // 4 == head.num_frame_per_block:
                print("no further action")
            elif videos.shape[2] // 4 != head.num_frame_per_block:
                repeat_factor = head.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
                first_frame = videos[:, :, 0:1]
                videos = torch.cat([first_frame, videos], dim=2)
            else:
                first_frame = videos[:, :, 0:1]
                videos = torch.cat([first_frame, videos], dim=2)

            image = head.vae.encode(
                videos,
                tiled=head.tiled,
                tile_size=(head.tile_size_height, head.tile_size_width),
                tile_stride=(head.tile_stride_height, head.tile_stride_width),
            )
        end_vae_event.record()

        noise_obs = head.generate_noise(
            (image.shape[0], 16, head.num_frame_per_block, height // 8, width // 8),
            seed=head.seed,
            device="cuda",
            dtype=torch.bfloat16,
        )
        noise_action = head.generate_noise(
            (image.shape[0], head.action_horizon, head.model.action_dim),
            seed=head.seed,
            device="cuda",
            dtype=torch.bfloat16,
        )
        batch_size, _, num_frames, height, width = noise_obs.shape
        frame_seqlen = int(height * width / 4)
        seq_len = frame_seqlen * num_frames
        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        return LazyJointVideoActionIntermediate(
            action_input=action_input,
            state_features=state_features,
            embodiment_id=embodiment_id,
            videos=videos,
            prompt_embs=prompt_embs,
            image=image,
            clip_feas=head.clip_feas,
            ys=head.ys,
            noise_obs=noise_obs,
            noise_action=noise_action,
            batch_size=batch_size,
            frame_seqlen=frame_seqlen,
            seq_len=seq_len,
            timings={
                "start_time": start_time,
                "start_text_encoder_event": start_text_encoder_event,
                "end_text_encoder_event": end_text_encoder_event,
                "start_image_encoder_event": start_image_encoder_event,
                "end_image_encoder_event": end_image_encoder_event,
                "start_vae_event": start_vae_event,
                "end_vae_event": end_vae_event,
                "start_kv_event": start_kv_event,
                "end_kv_event": end_kv_event,
            },
        )


class WANDiffusionRuntime:
    def __init__(self, head: WANPolicyHead):
        self.head = head

    def run(self, encoded: LazyJointVideoActionIntermediate) -> BatchFeature:
        head = self.head
        start_kv_event = encoded.timings["start_kv_event"]
        end_kv_event = encoded.timings["end_kv_event"]
        start_kv_event.record()

        if head.current_start_frame == 0:
            head.kv_cache1, head.kv_cache_neg = head._create_kv_caches(
                batch_size=encoded.batch_size,
                dtype=encoded.noise_obs.dtype,
                device=encoded.noise_obs.device,
                frame_seqlen=encoded.frame_seqlen,
            )
            head.crossattn_cache, head.crossattn_cache_neg = head._create_crossattn_caches(
                batch_size=encoded.batch_size,
                dtype=encoded.noise_obs.dtype,
                device=encoded.noise_obs.device,
            )

        assert head.kv_cache1 is not None
        assert head.kv_cache_neg is not None
        assert head.crossattn_cache is not None
        assert head.crossattn_cache_neg is not None
        kv_caches = head._get_caches([head.kv_cache1, head.kv_cache_neg])
        crossattn_caches = head._get_caches([head.crossattn_cache, head.crossattn_cache_neg])

        if head.current_start_frame == 0:
            timestep = torch.ones(
                [encoded.batch_size, 1], device=encoded.noise_obs.device, dtype=torch.int64
            ) * 0
            head._run_diffusion_steps(
                noisy_input=encoded.image.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=encoded.prompt_embs,
                seq_len=encoded.frame_seqlen,
                y=encoded.ys[:, :, 0:1],
                clip_feature=encoded.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(start_frame=0, update_kv_cache=True),
            )
            head.current_start_frame += 1

        timestep = torch.ones(
            [encoded.batch_size, head.num_frame_per_block],
            device=encoded.noise_obs.device,
            dtype=torch.int64,
        ) * 0
        if head.current_start_frame != 1:
            current_ref_latents = encoded.image[:, -head.num_frame_per_block:]
            if head.current_start_frame <= encoded.ys.shape[2]:
                y = encoded.ys[
                    :,
                    :,
                    head.current_start_frame - head.num_frame_per_block : head.current_start_frame,
                ]
            else:
                y = encoded.ys[:, :, -head.num_frame_per_block:]
            head._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=encoded.prompt_embs,
                seq_len=encoded.seq_len,
                y=y,
                clip_feature=encoded.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=head.current_start_frame - head.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )
        end_kv_event.record()

        noisy_input = encoded.noise_obs
        noisy_input_action = encoded.noise_action
        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=head.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=head.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sample_scheduler.set_timesteps(
            head.num_inference_steps, device=encoded.noise_obs.device, shift=head.sigma_shift
        )
        sample_scheduler_action.set_timesteps(
            head.num_inference_steps, device=encoded.noise_obs.device, shift=head.sigma_shift
        )

        if head.config.decouple_inference_noise:
            video_final_noise = head.config.video_inference_final_noise
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = (
                sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max
                + video_final_noise
            )
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
            if head.ip_rank == 0:
                print(
                    f"Decoupled inference: video sigmas {sigma_max:.3f} -> "
                    f"{sample_scheduler.sigmas[-1].item():.3f}"
                )

        start_diffusion_events = [
            torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps
        ]
        end_diffusion_events = [
            torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps
        ]
        prev_predictions = []
        head.skip_countdown = 0
        dit_compute_steps = 0

        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            start_diffusion_events[index].record()
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]
            timestep = torch.ones(
                [encoded.batch_size, head.num_frame_per_block],
                device=encoded.noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [encoded.batch_size, head.action_horizon],
                device=encoded.noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            should_run_model = head.should_run_model(index, current_timestep, prev_predictions)
            if should_run_model:
                dit_compute_steps += 1
                if head.current_start_frame + head.num_frame_per_block <= encoded.ys.shape[2]:
                    y = encoded.ys[
                        :,
                        :,
                        head.current_start_frame : head.current_start_frame + head.num_frame_per_block,
                    ]
                else:
                    y = encoded.ys[:, :, -head.num_frame_per_block:]
                predictions = head._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=encoded.state_features,
                    embodiment_id=encoded.embodiment_id,
                    context=encoded.prompt_embs,
                    seq_len=encoded.seq_len,
                    y=y,
                    clip_feature=encoded.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=head.current_start_frame,
                        update_kv_cache=False,
                    ),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, _ = predictions[1]
                flow_pred = flow_pred_uncond + head.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)
            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            end_diffusion_events[index].record()
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            noisy_input_action = sample_scheduler_action.step(
                model_output=flow_pred_cond_action,
                timestep=action_timestep,
                sample=noisy_input_action,
                step_index=index,
                return_dict=False,
            )[0]

        latents = noisy_input
        latents_action = noisy_input_action
        output = latents
        if head.current_start_frame == 1:
            output = torch.cat([encoded.image, output], dim=1)
        head.current_start_frame += head.num_frame_per_block

        torch.cuda.synchronize()
        total_time = time.perf_counter() - encoded.timings["start_time"]
        text_encoder_time = (
            encoded.timings["start_text_encoder_event"].elapsed_time(
                encoded.timings["end_text_encoder_event"]
            )
            / 1000
        )
        image_encoder_time = (
            encoded.timings["start_image_encoder_event"].elapsed_time(
                encoded.timings["end_image_encoder_event"]
            )
            / 1000
        )
        vae_time = (
            encoded.timings["start_vae_event"].elapsed_time(encoded.timings["end_vae_event"]) / 1000
        )
        kv_creation_time = (
            encoded.timings["start_kv_event"].elapsed_time(encoded.timings["end_kv_event"]) / 1000
        )
        diffusion_times = [
            s.elapsed_time(e) for s, e in zip(start_diffusion_events, end_diffusion_events)
        ]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = (
            total_time
            - kv_creation_time
            - diffusion_time
            - text_encoder_time
            - image_encoder_time
            - vae_time
        )
        if head.ip_rank == 0:
            print(
                f"Time taken: Total {total_time:.2f} seconds, "
                f"Text Encoder {text_encoder_time:.2f} seconds, "
                f"Image Encoder {image_encoder_time:.2f} seconds, "
                f"VAE {vae_time:.2f} seconds, "
                f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                f"Diffusion {diffusion_time:.2f} seconds, "
                f"DIT Compute Steps {dit_compute_steps} steps, "
                f"Scheduler {scheduler_time:.2f} seconds"
            )
        return BatchFeature(
            data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)}
        )
