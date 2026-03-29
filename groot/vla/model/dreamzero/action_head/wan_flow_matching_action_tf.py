from dataclasses import dataclass, field
import logging
import time
from typing import TypeAlias, cast
import os

from accelerate import load_checkpoint_and_dispatch

from einops import rearrange
from hydra.utils import instantiate
from peft import LoraConfig, get_peft_model
import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from safetensors.torch import load_file
import json
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)

WAN_HF_REPO_ID = "Wan-AI/Wan2.1-I2V-14B-480P"
DEFAULT_LOCAL_WAN_DIR = None


def hf_download(filename: str) -> str:
    """Download a file from the Wan2.1-I2V-14B-480P HuggingFace repo to HF cache."""
    path = hf_hub_download(repo_id=WAN_HF_REPO_ID, filename=filename)
    return path


def get_local_wan_dir() -> str | None:
    local_wan_dir = os.environ.get("WAN_CKPT_DIR", DEFAULT_LOCAL_WAN_DIR)
    if local_wan_dir and os.path.isdir(local_wan_dir):
        return local_wan_dir
    return None


def ensure_file(path: str | None, hf_filename: str) -> str:
    """Return a valid local path, preferring explicit and local Wan files before HuggingFace."""
    if path is not None and os.path.exists(path):
        return path
    local_wan_dir = get_local_wan_dir()
    if local_wan_dir is not None:
        local_candidate = os.path.join(local_wan_dir, hf_filename)
        if os.path.exists(local_candidate):
            logger.info("Using local Wan file for %s: %s", hf_filename, local_candidate)
            return local_candidate
    return hf_download(hf_filename)


def ensure_diffusion_dir(path: str | None) -> str:
    """Return a valid local diffusion dir, preferring a local Wan directory before HuggingFace."""
    if path is not None and os.path.isdir(path):
        return path

    local_wan_dir = get_local_wan_dir()
    if local_wan_dir is not None:
        index_path = os.path.join(local_wan_dir, "diffusion_pytorch_model.safetensors.index.json")
        safetensors_path = os.path.join(local_wan_dir, "diffusion_pytorch_model.safetensors")
        if os.path.exists(index_path) or os.path.exists(safetensors_path):
            logger.info("Using local Wan diffusion directory: %s", local_wan_dir)
            return local_wan_dir

    index_path = hf_hub_download(
        repo_id=WAN_HF_REPO_ID,
        filename="diffusion_pytorch_model.safetensors.index.json",
    )
    dit_dir = os.path.dirname(index_path)
    with open(index_path, 'r') as f:
        index = json.load(f)
    for shard_file in set(index["weight_map"].values()):
        hf_hub_download(repo_id=WAN_HF_REPO_ID, filename=shard_file)
    return dit_dir

from torch.distributions import Beta
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torchvision.transforms import v2
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from groot.vla.model.n1_5.action_head.base_action_head import ActionHead
from groot.vla.model.dreamzero.modules.flow_match_scheduler import FlowMatchScheduler
from groot.vla.model.dreamzero.modules.vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from groot.vla.model.dreamzero.modules.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from groot.vla.model.dreamzero.modules.flow_unipc_multistep_scheduler import FlowUniPCMultistepScheduler


KVCacheType: TypeAlias = torch.Tensor

@dataclass
class WANPolicyHeadConfig(PretrainedConfig):
    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )
    tiled: bool = field(default=True, metadata={"help": "Whether to use tiled input."})
    tile_size_height: int = field(default=34, metadata={"help": "Tile size height."})
    tile_size_width: int = field(default=34, metadata={"help": "Tile size width."})
    tile_stride_height: int = field(default=18, metadata={"help": "Tile stride height."})
    tile_stride_width: int = field(default=16, metadata={"help": "Tile stride width."})
    num_frame_per_block: int = field(default=1, metadata={"help": "Number of frames per block."})

    lora_rank: int = field(default=4, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=4, metadata={"help": "LoRA alpha."})
    lora_target_modules: str = field(default="q,k,v,o,ffn.0,ffn.2")
    init_lora_weights: str = field(default="kaiming", metadata={"help": "LoRA initialization method."})
    train_architecture: str= field(default="lora", metadata={"help": "Train architecture."})
    skip_component_loading: bool = field(default=False, metadata={"help": "Skip loading individual component weights (used when loading from full pretrained model)."})

    use_gradient_checkpointing: bool = field(default=True, metadata={"help": "Whether to use gradient checkpointing."})
    qformer_cfg: dict = field(default=None, metadata={"help": "Qformer configuration."})
    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    # High noise emphasis for BASE (coupled) training - applies Beta distribution to BOTH video and action together
    use_high_noise_emphasis: bool = field(
        default=False, metadata={"help": "Use Beta distribution for noise sampling (biases BOTH video and action towards high noise levels together)."}
    )
    high_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for high noise emphasis. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher = more high noise bias."}
    )
    # Decoupled noise sampling config for training-inference alignment
    # When enabled: video uses Beta(alpha,beta) biased towards high noise, action uses independent uniform
    decouple_video_action_noise: bool = field(
        default=False, metadata={"help": "Decouple video/action noise: video uses Beta distribution (high noise bias), action uses independent uniform."}
    )
    video_noise_beta_alpha: float = field(
        default=3.0, metadata={"help": "Beta alpha for video noise. Beta(3,1): mean=0.75, Beta(5,1): mean=0.83. Higher alpha = more bias to high noise."}
    )
    video_noise_beta_beta: float = field(
        default=1.0, metadata={"help": "Beta beta for video noise. Keep at 1.0."}
    )
    # Decoupled inference config - allows video to stay noisy while action fully denoises
    decouple_inference_noise: bool = field(
        default=False, metadata={"help": "Use decoupled noise schedules during inference (video stays noisy, action fully denoises)."}
    )
    video_inference_final_noise: float = field(
        default=0.8, metadata={"help": "Final noise level for video during decoupled inference (0.0-1.0). E.g., 0.8 means video ends at 80% noise."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)
    defer_lora_injection: bool = field(default=False, metadata={"help": "Defer LoRA injection until after loading pretrained weights."})

    vl_self_attention_cfg: dict = field(default=None)
    text_encoder_cfg: dict = field(default=None)
    image_encoder_cfg: dict = field(default=None)
    vae_cfg: dict = field(default=None)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class WANPolicyHead(ActionHead):
    config_class = WANPolicyHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: WANPolicyHeadConfig,
    ):
        super().__init__()
        self.tiled = config.tiled
        self.tile_size_height = config.tile_size_height
        self.tile_size_width = config.tile_size_width
        self.tile_stride_height = config.tile_stride_height
        self.tile_stride_width = config.tile_stride_width
        self.num_frame_per_block = config.num_frame_per_block
        self.hidden_size = config.hidden_size
        self.num_frames = config.num_frames
        self.text_encoder = instantiate(config.text_encoder_cfg)
        self.image_encoder = instantiate(config.image_encoder_cfg)
        self.vae = instantiate(config.vae_cfg)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.model_names = ['text_encoder']

        self.num_inference_steps = 16 
        self.seed = 1140
        self.cfg_scale = 5.0
        self.denoising_strength = 1.0
        self.sigma_shift = 5.0
        self.kv_cache1: KVCacheType | None = None
        self.kv_cache_neg: KVCacheType | None = None
        self.crossattn_cache: KVCacheType | None = None
        self.crossattn_cache_neg: KVCacheType | None = None

        self.global_step = 0
        self.max_steps = 0
        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lora_target_modules = config.lora_target_modules
        self.init_lora_weights = config.init_lora_weights
        self.train_architecture = config.train_architecture
        self.clip_feas = None
        self.ys = None
        self.current_start_frame = 0
        self.language = None

        self.ip_rank = 0
        self.ip_size = 1
        self.ip_group = None
        
        self._device = "cuda"
        self.dynamic_cache_schedule = os.getenv("DYNAMIC_CACHE_SCHEDULE", "False").lower() == "true"


        num_dit_steps = 8
        if os.getenv("NUM_DIT_STEPS") is not None:
            num_dit_steps = int(os.getenv("NUM_DIT_STEPS"))
        if num_dit_steps == 5:
            self.dit_step_mask = [True, True, True, False, False, False, False, True, False, False, False, False, True, False, False, False]
        elif num_dit_steps == 6:
            self.dit_step_mask = [True, True, False, False, False, True, False, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 7:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, False, True, True]
        elif num_dit_steps == 8:
            self.dit_step_mask = [True, True, True, False, False, False, True, False, False, False, True, False, False, True, True, True]
        else:
            self.dit_step_mask = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
        assert self.dit_step_mask[0] == True, "first step must be True"

        self.normalize_video = v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])


        self.use_gradient_checkpointing = config.use_gradient_checkpointing
        if self.training:
            self.scheduler.set_timesteps(1000, training=True)
        
        
        self.input_embedding_dim = config.input_embedding_dim

        self.cpu_offload = False

        self.model = instantiate(config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps
        
        text_enc_path = ensure_file(
            self.text_encoder.text_encoder_pretrained_path,
            "models_t5_umt5-xxl-enc-bf16.pth",
        )
        self.text_encoder.load_state_dict(torch.load(text_enc_path, map_location='cpu'))

        img_enc_path = ensure_file(
            self.image_encoder.image_encoder_pretrained_path,
            "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        )
        self.image_encoder.model.load_state_dict(torch.load(img_enc_path, map_location='cpu'), strict=False)

        vae_path = ensure_file(
            self.vae.vae_pretrained_path,
            "Wan2.1_VAE.pth",
        )
        self.vae.model.load_state_dict(torch.load(vae_path, map_location='cpu'))

        if not config.skip_component_loading:
            dit_dir = ensure_diffusion_dir(self.model.diffusion_model_pretrained_path)

            if dit_dir is not None:
                safetensors_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors")
                safetensors_index_path = os.path.join(dit_dir, "diffusion_pytorch_model.safetensors.index.json")
                state_dict = {}

                if os.path.exists(safetensors_index_path):
                    # Handle sharded safetensors
                    print(f"Loading sharded safetensors using index: {safetensors_index_path}")

                    with open(safetensors_index_path, 'r') as f:
                        index = json.load(f)

                    # Load each shard
                    for shard_file in set(index["weight_map"].values()):
                        shard_path = os.path.join(dit_dir, shard_file)
                        print(f"Loading shard: {shard_path}")
                        shard_state_dict = load_file(shard_path)
                        state_dict.update(shard_state_dict)

                elif os.path.exists(safetensors_path):
                    # Handle single safetensors file
                    print(f"Loading weights from safetensors: {safetensors_path}")
                    state_dict = load_file(safetensors_path)

                else:
                    raise ValueError(f"No safetensors file found at {safetensors_path} or {safetensors_index_path}")

                missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)

                if missing_keys:
                    print(f"Missing keys when loading pretrained weights: {missing_keys}")
                if unexpected_keys:
                    print(f"Unexpected keys when loading pretrained weights: {unexpected_keys}")

                print("Successfully loaded pretrained weights")
        else:
            print("Skipping individual component loading (loading from full pretrained model)")
        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        # Video noise Beta distribution (biased towards high noise levels when enabled)
        self.video_beta_dist = Beta(config.video_noise_beta_alpha, config.video_noise_beta_beta)
        # High noise emphasis Beta distribution for coupled training (applies to both video and action)
        self.high_noise_beta_dist = Beta(config.high_noise_beta_alpha, 1.0)
        # self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self._noise_logged = False
        self._debug_forward_count = 0  # counts forward() calls for debug throttling
        self.defer_lora_injection = config.defer_lora_injection
        print("defer_lora_injection@@", self.defer_lora_injection)
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

        if self.train_architecture == "lora" and not self.defer_lora_injection:
            print("Adding LoRA to model")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
        elif self.train_architecture == "lora" and self.defer_lora_injection:
            print("Deferring LoRA injection until after pretrained weights are loaded")
        else:
            self.print_trainable_params()

        self.text_encoder.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.vae.requires_grad_(False)
        if not self.defer_lora_injection:
            self.print_trainable_params()


    def print_trainable_params(self):
        """Print trainable parameters of the diffusion model."""
        trainable_params = []
        total_params = 0
        trainable_total = 0
        
        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(name)
                trainable_total += param.numel()
                
        print(f"Total parameters in diffusion model: {total_params:,}")
        print(f"Trainable parameters in diffusion model: {trainable_total:,}")
        # print(trainable_params)


    def inject_lora_after_loading(self):
        """
        Inject LoRA adapters after pretrained weights have been loaded.
        This should be called when defer_lora_injection=True.
        """
        if self.train_architecture == "lora":
            print("Injecting LoRA after loading pretrained weights")
            for p in self.parameters():
                p.requires_grad = False
            self.model = self.add_lora_to_model(
                self.model,
                lora_rank=self.lora_rank,
                lora_alpha=self.lora_alpha,
                lora_target_modules=self.lora_target_modules,
                init_lora_weights=self.init_lora_weights,
            )
            self.model.state_encoder.requires_grad_(True)
            self.model.action_encoder.requires_grad_(True)
            self.model.action_decoder.requires_grad_(True)
            # self.model.registers.requires_grad_(True)
            # self.model.time_modality_projection.requires_grad_(True)
            
            self.text_encoder.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            self.print_trainable_params()
        else:
            print("LoRA injection not needed (train_architecture != 'lora')")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_diffusion_model:
                self.model.eval()
            self.text_encoder.eval()
            self.image_encoder.eval()
            self.vae.eval()
    
    
    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.dtype,
                computation_device='cuda',
            ),
        )

        self.cpu_offload = True

    def load_models_to_device(self, loadmodel_names=[]):
        # only load models to device if cpu_offload is enabled
        if not self.cpu_offload:
            return
        # offload the unneeded models to cpu
        for model_name in self.model_names:
            if model_name not in loadmodel_names:
                model = getattr(self, model_name)
                if model is not None:
                    if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                        print("offloadd")
                        for module in model.modules():
                            if hasattr(module, "offload"):
                                # print("offload", module)
                                module.offload()
                    else:
                        print("tocpu")
                        model.cpu()
        # load the needed models to device
        for model_name in loadmodel_names:
            model = getattr(self, model_name)
            if model is not None:
                if hasattr(model, "vram_management_enabled") and model.vram_management_enabled:
                    print("onload")
                    for module in model.modules():
                        if hasattr(module, "onload"):
                            # print("onload", module)
                            module.onload()
                else:
                    print("togpu")
                    model.to(self._device)
        # fresh the cuda cache
        torch.cuda.empty_cache()

    def _create_kv_caches(
        self,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        frame_seqlen: int,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1: KVCacheType = []
        kv_cache_neg: KVCacheType = []
        for _ in range(self.model.num_layers):
            kv_cache1.append(
               torch.zeros([2, batch_size, 0, 40, 128], dtype=dtype, device=device),
            )
            kv_cache_neg.append(
                torch.zeros([2, batch_size, 0, 40, 128], dtype=dtype, device=device),
            )

        return kv_cache1, kv_cache_neg

    def _create_crossattn_caches(
        self, batch_size: int, dtype: torch.dtype, device: torch.device,
    ) -> tuple[KVCacheType, KVCacheType]:
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache: KVCacheType = []
        crossattn_cache_neg: KVCacheType = []

        for _ in range(self.model.num_layers):
            crossattn_cache.append(
                torch.zeros([2, batch_size, 512, 40, 128], dtype=dtype, device=device),
            )
            crossattn_cache_neg.append(
                torch.zeros([2, batch_size, 512, 40, 128], dtype=dtype, device=device),
            )

        return crossattn_cache, crossattn_cache_neg
        
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def preprocess_image(self, image):
        image = (image * (2 / 255) - 1).permute(0, 1, 4, 2, 3)
        return image

    def encode_prompt(self, input_ids, attention_mask):
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        prompt_emb = self.text_encoder(input_ids, attention_mask)
        prompt_emb = prompt_emb.clone().to(dtype=torch.bfloat16)
        for i, v in enumerate(seq_lens):
            prompt_emb[:, v:] = 0
        return prompt_emb

    def _ensure_vae_on_device(self, ref_tensor):
        """Lazily move the VAE to the correct device/dtype on first use."""
        if not getattr(self, '_vae_device_ready', False):
            self.vae.to(device=ref_tensor.device, dtype=torch.bfloat16)
            self.vae.eval()
            self._vae_device_ready = True

    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        self._ensure_vae_on_device(input_video)
        with torch.no_grad():
            latents = self.vae.encode(input_video, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents

    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            batch_size = image.shape[0]
            clip_context = self.image_encoder.encode_image(image)
            msk = torch.ones(batch_size, num_frames, height//8, width//8, device=self._device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(batch_size, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)
            # mask shape is B * 4 * (1+(T-1)/4) * h/8 * w/8
            image_input = image.transpose(1, 2)
            image_zeros = torch.zeros(batch_size, 3, num_frames-1, height, width, dtype=torch.bfloat16, device=self._device)
            self._ensure_vae_on_device(image_input)
            with torch.no_grad():
                y = self.vae.encode(torch.concat([image_input, image_zeros], dim=2))
            new_image = y[:, :, 0:1]
            # y shape is B * 16 * (1+(T-1)/4) * h/8 * w/8
            y = torch.concat([msk, y], dim=1)
        return clip_context, y, new_image
    
    def prepare_extra_input(self, latents=None):
        return {}

    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming") -> nn.Module:
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = get_peft_model(model, lora_config)
        for param in model.parameters():
            param.data = param.to(torch.float32)
        return model

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        data = action_input 
        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id
        # print("embodiment_id", embodiment_id)
        has_real_action = action_input.has_real_action
        action_mask = action_input.action_mask

        state_features = action_input.state

        actions = action_input.action
        # assert the values of action is in between -1 and 1
        if actions.numel() > 0:
            assert actions.min() >= -1.0 and actions.max() <= 1.0, "actions must be in [-1,1] range"
        videos = data["images"]

        videos = rearrange(videos, "b t h w c -> b c t h w")

        _debug_this_step = self._debug_forward_count < 3  # only print for first 3 forward calls
        _is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
        if _debug_this_step and _is_rank0:
            print(f"\n[DEBUG][forward] === Step {self._debug_forward_count} ===")
            print(f"[DEBUG][forward] videos (b,c,t,h,w)={tuple(videos.shape)}, dtype={videos.dtype}")
            print(f"[DEBUG][forward] actions shape={tuple(actions.shape)}, state_features shape={tuple(state_features.shape)}")
            print(f"[DEBUG][forward] embodiment_id={embodiment_id}, has_real_action={has_real_action}")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)
        
        # shape of B * max_length * dim
        prompt_embs = self.encode_prompt(data["text"], data["text_attention_mask"])
        if _debug_this_step and _is_rank0:
            print(f"[DEBUG][forward] prompt_embs shape={tuple(prompt_embs.shape)}, dtype={prompt_embs.dtype}")

        latents = self.encode_video(videos, self.tiled, (self.tile_size_height, self.tile_size_width), (self.tile_stride_height, self.tile_stride_width))
        if _debug_this_step and _is_rank0:
            print(f"[DEBUG][forward] latents (VAE encoded) shape={tuple(latents.shape)}, dtype={latents.dtype}")

        _, _, num_frames, height, width = videos.shape
        image = videos[:, :, :1].transpose(1, 2)

        clip_feas, ys, _ = self.encode_image(image, num_frames, height, width)
        if _debug_this_step and _is_rank0:
            print(f"[DEBUG][forward] clip_feas shape={tuple(clip_feas.shape)}, ys shape={tuple(ys.shape)}")

        latents = latents.to(self._device)
        clip_feas = clip_feas.to(self._device)
        ys = ys.to(self._device)
        prompt_embs = prompt_embs.to(self._device)
       
        # Loss
        noise = torch.randn_like(latents)

        # specific to autoregressive 
        noise = noise.transpose(1, 2)
        latents = latents.transpose(1, 2)
        
        # ============ VIDEO TIMESTEP SAMPLING ============
        if self.config.decouple_video_action_noise:
            # Decoupled mode: sample video from Beta distribution biased towards HIGH noise
            video_noise_ratio = self.video_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - video_noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "DECOUPLED"
        elif self.config.use_high_noise_emphasis:
            # High noise emphasis mode (coupled): BOTH video and action use Beta distribution
            noise_ratio = self.high_noise_beta_dist.sample([noise.shape[0], noise.shape[1]])
            timestep_id = ((1.0 - noise_ratio) * self.scheduler.num_train_timesteps).long()
            timestep_id = torch.clamp(timestep_id, 0, self.scheduler.num_train_timesteps - 1)
            noise_mode = "HIGH_NOISE_EMPHASIS"
        else:
            # Original: uniform sampling over full range
            timestep_id = torch.randint(0, self.scheduler.num_train_timesteps, (noise.shape[0], noise.shape[1]))
            noise_mode = "STANDARD"
        
        timestep_id_block = timestep_id[:, 1:].reshape(
                    timestep_id.shape[0], -1, self.num_frame_per_block)
        timestep_id_block[:, :, 1:] = timestep_id_block[:, :, 0:1]
        
        if actions.numel() > 0:
            noise_action = torch.randn_like(actions)
            assert actions.shape[1] / (noise.shape[1]-1) == (self.model.num_action_per_block // self.num_frame_per_block), f"actions.shape, {actions.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            assert (noise.shape[1]-1) / state_features.shape[1] == (self.num_frame_per_block // self.model.num_state_per_block), f"state_features.shape, {state_features.shape}, noise.shape, {noise.shape}, video.shape, {videos.shape}, latents.shape, {latents.shape}"
            
            # ============ ACTION TIMESTEP SAMPLING ============
            if self.config.decouple_video_action_noise:
                # Decoupled: sample action timestep independently with full range
                timestep_action_id = torch.randint(
                    0, 
                    self.scheduler.num_train_timesteps, 
                    (actions.shape[0], actions.shape[1])
                )
                action_mode = "INDEPENDENT"
            else:
                # Original coupled: action timestep derived from video timestep
                timestep_action_id = timestep_id_block.repeat(1, 1, actions.shape[1]//(noise.shape[1]-1))
                timestep_action_id = timestep_action_id.reshape(timestep_action_id.shape[0], -1)
                action_mode = "COUPLED"
            
            # Log noise mode once
            if not self._noise_logged:
                video_mean = timestep_id.float().mean().item()
                action_mean = timestep_action_id.float().mean().item()
                if noise_mode == "DECOUPLED":
                    print(f"[NOISE] Mode={noise_mode} | Video: Beta({self.config.video_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode} Uniform mean_t={action_mean:.0f}")
                elif noise_mode == "HIGH_NOISE_EMPHASIS":
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Beta({self.config.high_noise_beta_alpha},1) mean_t={video_mean:.0f} | Action: {action_mode}")
                else:
                    print(f"[NOISE] Mode={noise_mode} | Video+Action: Uniform mean_t={video_mean:.0f} | Action: {action_mode}")
                self._noise_logged = True
        else:
            noise_action = None
            timestep_action_id = None
            
        timestep_id_block = timestep_id_block.reshape(timestep_id_block.shape[0], -1)
        timestep_id = torch.concat([timestep_id[:, :1], timestep_id_block], dim=1)
        _, num_frames, num_channels, height, width = noise.shape
        # Match the DiT's (1, 2, 2) patch embedding tokenization on latent frames.
        frame_seqlen = (height // 2) * (width // 2)
        seq_len = num_frames * frame_seqlen

        timestep = self.scheduler.timesteps[timestep_id].to(self._device)
        noisy_latents = self.scheduler.add_noise(latents.flatten(0, 1), noise.flatten(0, 1), timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1]))
        training_target = self.scheduler.training_target(latents, noise, timestep).transpose(1, 2)
        
        if actions.numel() > 0:
            timestep_action = self.scheduler.timesteps[timestep_action_id].to(self._device)
            noisy_actions = self.scheduler.add_noise(
                actions.flatten(0, 1),
                noise_action.flatten(0, 1),
                timestep_action.flatten(0, 1),
            ).unflatten(0, (noise_action.shape[0], noise_action.shape[1]))
            training_target_action = self.scheduler.training_target(actions, noise_action, timestep_action)
        else:
            timestep_action = None
            noisy_actions = None
            training_target_action = None

        if _debug_this_step and _is_rank0:
            print(f"[DEBUG][forward] noisy_latents shape={tuple(noisy_latents.shape)}, noisy_actions shape={tuple(noisy_actions.shape) if noisy_actions is not None else None}")
            print(f"[DEBUG][forward] timestep shape={tuple(timestep.shape)}, seq_len={seq_len}")

        # Compute loss
        with torch.amp.autocast(dtype=torch.bfloat16, device_type=torch.device(self._device).type):
            if actions.numel() > 0:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    action=noisy_actions, timestep_action=timestep_action,
                    clean_x=latents.transpose(1, 2),
                )
            else:
                video_noise_pred, action_noise_pred = self.model(
                    noisy_latents.transpose(1, 2), timestep=timestep, timestep_action=timestep_action,
                    clip_feature=clip_feas, y=ys, context=prompt_embs, seq_len=seq_len,
                    state=state_features, embodiment_id=embodiment_id,
                    clean_x=latents.transpose(1, 2),
                )

            if _debug_this_step and _is_rank0:
                print(f"[DEBUG][forward] video_noise_pred shape={tuple(video_noise_pred.shape)}, action_noise_pred shape={tuple(action_noise_pred.shape) if action_noise_pred is not None else None}")

            # Per-sample dynamics loss
            dynamics_loss_per_sample = torch.nn.functional.mse_loss(
                video_noise_pred.float(), training_target.float(), reduction='none'
            ).mean(dim=(1,3,4))  # shape: [B, ...]

            weight_dynamics = dynamics_loss_per_sample * self.scheduler.training_weight(timestep.flatten(0, 1)).unflatten(0, (noise.shape[0], noise.shape[1])).to(self._device)
            weighted_dynamics_loss = weight_dynamics.mean()
            
            if actions.numel() > 0:
                action_loss_per_sample = torch.nn.functional.mse_loss(
                    action_noise_pred.float(), training_target_action.float(), reduction='none'
                ) * action_mask  # shape: [B, ...]
                action_loss_per_sample = has_real_action[:, None, None].float() * action_loss_per_sample  # apply has_real_action across all action tokens and dims
                weight_action = action_loss_per_sample.mean(dim=2) * self.scheduler.training_weight(
                    timestep_action.flatten(0, 1),
                ).unflatten(0, (noise_action.shape[0], noise_action.shape[1])).to(self._device)
                weighted_action_loss = weight_action.mean()
                loss = weighted_dynamics_loss + weighted_action_loss
            else:
                weighted_action_loss = torch.tensor(0.0, device=self._device)
                loss = weighted_dynamics_loss
            # loss = dynamics_loss_per_sample.mean()

        # Record log
        output_dict = {
            "loss": loss,
            "dynamics_loss": weighted_dynamics_loss,
            "action_loss": weighted_action_loss,
        }

        if _debug_this_step and _is_rank0:
            print(f"[DEBUG][forward] loss={loss.item():.6f}, dynamics_loss={weighted_dynamics_loss.item():.6f}, action_loss={weighted_action_loss.item():.6f}")
        self._debug_forward_count += 1

        return BatchFeature(data=output_dict)

    def generate_noise(self, shape, seed=None, device="cpu", dtype=torch.float16):
        generator = None if seed is None else torch.Generator(device).manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return noise
    
    def _get_caches(
        self, kv_caches_input: list[KVCacheType],
    ) -> list[KVCacheType]:
        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            assert len(kv_caches_input) == 2
            if self.ip_rank == 0:
                kv_caches = [kv_caches_input[0]]
            else:
                kv_caches = [kv_caches_input[1]]
        else:
            assert len(kv_caches_input) <= 2
            kv_caches = [kv_caches_input[0]]
            if self.cfg_scale != 1.0:
                kv_caches.append(kv_caches_input[1])
        return kv_caches

    def _prepare_text_inputs(self, data: BatchFeature) -> list[tuple[torch.Tensor, torch.Tensor]]:

        if self.ip_size > 1:
            assert self.cfg_scale != 1.0, "cfg_scale must be != 1.0 when ip_size > 1"
            if self.ip_rank == 0:
                text_inputs = [(data["text"], data["text_attention_mask"])]
            else:
                text_inputs = [(data["text_negative"], data["text_attention_mask_negative"])]
        else:
            text_inputs = [(data["text"], data["text_attention_mask"])]
            if self.cfg_scale != 1.0:
                text_inputs.append((data["text_negative"], data["text_attention_mask_negative"]))
        return text_inputs


    def _run_diffusion_steps(
        self,
        noisy_input: torch.Tensor,
        timestep: torch.Tensor,
        action: torch.Tensor,
        timestep_action: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        context: torch.Tensor,
        seq_len: int,
        y: torch.Tensor,
        clip_feature: torch.Tensor,
        kv_caches: list[KVCacheType],
        crossattn_caches: list[KVCacheType],
        kv_cache_metadata: dict[str, bool | int],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        predictions = []
        for index, prompt_emb in enumerate(context):
            kv_cache = kv_caches[index]
            crossattn_cache = crossattn_caches[index]
            if not kv_cache_metadata["update_kv_cache"] and self.trt_engine is not None:
                obs_noise_pred, action_noise_pred = self.trt_engine(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    context=prompt_emb,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                )
            else:
                obs_noise_pred, action_noise_pred, updated_kv_caches = self.model(
                    noisy_input,
                    timestep,
                    action=action,
                    timestep_action=timestep_action,
                    state=state,
                    embodiment_id=embodiment_id,
                    context=prompt_emb,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=clip_feature,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start_frame=kv_cache_metadata["start_frame"],
                )
                if kv_cache_metadata["update_kv_cache"]:
                    for block_index, updated_kv_cache in enumerate(updated_kv_caches):
                        kv_cache[block_index] = updated_kv_cache.clone()
            obs_noise_pred = obs_noise_pred.clone()
            if action_noise_pred is not None:
                action_noise_pred = action_noise_pred.clone()
            else:
                action_noise_pred = torch.tensor(0.0, device=obs_noise_pred.device) # dummy action noise prediction
            predictions.append((obs_noise_pred, action_noise_pred))
        return self._exchange_predictions(predictions)

    def _exchange_predictions(
        self,
        predictions: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.ip_size == 1:
            return predictions

        assert len(predictions) == 1
        my_predictions = list(predictions[0])

        other_predictions = [torch.empty_like(pred) for pred in my_predictions]

        send_ops = [
            dist.P2POp(op=dist.isend, tensor=pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for pred in my_predictions
        ]
        recv_ops = [
            dist.P2POp(op=dist.irecv, tensor=other_pred, group_peer=(self.ip_rank + 1) % self.ip_size, group=self.ip_group)
            for other_pred in other_predictions
        ]
        ops = send_ops + recv_ops

        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

        output_predictions: list[tuple[torch.Tensor, torch.Tensor] | None] = [None for _ in range(self.ip_size)]
        output_predictions[self.ip_rank] = tuple(my_predictions)
        output_predictions[(self.ip_rank + 1) % self.ip_size] = tuple(other_predictions)
        assert all(isinstance(pred, tuple) for pred in output_predictions)
        return cast(list[tuple[torch.Tensor, torch.Tensor]], output_predictions)
    
    def should_run_model(self, index, current_timestep, prev_predictions):

        if not self.dynamic_cache_schedule:
            return self.dit_step_mask[index]

        # Always run first 2 steps to establish history
        if len(prev_predictions) < 2:
            return True

        if self.skip_countdown > 1:
            self.skip_countdown -= 1
            return False
        elif self.skip_countdown == 1:
            self.skip_countdown = 0 
            return True

        v_last = prev_predictions[-1][1].flatten(1).float()
        v_prev = prev_predictions[-2][1].flatten(1).float()
        sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()

        thresholds = [0.95, 0.93]
        countdowns = [4, 2]

        for threshold, countdown in zip(thresholds, countdowns):
            if sim > threshold:
                self.skip_countdown = countdown
                return False

        return True

    def lazy_joint_video_action(self, backbone_output: BatchFeature, action_input: BatchFeature, latent_video: torch.Tensor | None = None) -> BatchFeature:
        start_time = time.perf_counter()

        # Tracking time taken on GPU for various operations.
        start_text_encoder_event = torch.cuda.Event(enable_timing=True)
        end_text_encoder_event = torch.cuda.Event(enable_timing=True)
        start_image_encoder_event = torch.cuda.Event(enable_timing=True)
        end_image_encoder_event = torch.cuda.Event(enable_timing=True)
        start_vae_event = torch.cuda.Event(enable_timing=True)
        end_vae_event = torch.cuda.Event(enable_timing=True)
        start_kv_event = torch.cuda.Event(enable_timing=True)
        end_kv_event = torch.cuda.Event(enable_timing=True)
        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.num_inference_steps)]

        self.set_frozen_modules_to_eval_mode()
        data = action_input 
        
        videos = data["images"]

        embodiment_id = action_input.embodiment_id
        state_features = action_input.state

        videos = rearrange(videos, "b t h w c -> b c t h w")

        if videos.dtype == torch.uint8:
            videos = videos.float() / 255.0
            videos = videos.to(dtype=self.dtype)
            b, c, t, h, w = videos.shape
            videos = videos.permute(0, 2, 1, 3, 4)  # [b, t, c, h, w]
            videos = videos.reshape(b * t, c, h, w)
            videos = self.normalize_video(videos)
            videos = videos.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)  # back to [b, c, t, h, w]
            assert videos.min() >= -1.0 and videos.max() <= 1.0, "videos must be in [-1,1] range"
            videos = videos.to(dtype=self.dtype)

        state_features = state_features.to(dtype=torch.bfloat16)
        videos = videos.to(dtype=torch.bfloat16)

        if self.language is None:
            print("language is None, reset current_start_frame to 0")
            self.language = data["text"]
            self.current_start_frame = 0
        elif not torch.equal(self.language, data["text"]):
            print("language changed, reset current_start_frame to 0")
            self.current_start_frame = 0
            self.language = data["text"]
        elif videos.shape[2] == 1:
            print("videos.shape[2] == 1, reset current_start_frame to 0")
            self.current_start_frame = 0
        elif self.current_start_frame >= self.model.local_attn_size:
            print("current_start_frame >= local_attn_size, reset current_start_frame to 0")
            self.current_start_frame = 0

        if self.ip_rank == 0:
            print("videos shape", videos.shape, self.num_frames)

        start_text_encoder_event.record()

        text_inputs = self._prepare_text_inputs(data)
        prompt_embs = [self.encode_prompt(text, attention_mask) for text, attention_mask in text_inputs]

        end_text_encoder_event.record()
        
        start_image_encoder_event.record()

        _, _, num_frames, height, width = videos.shape
        if videos.shape[2] == 4 or videos.shape[2] == 9:
            # special case for real-world eval where language is updated
            image = videos[:, :, -1:].transpose(1, 2)
        else:
            image = videos[:, :, :1].transpose(1, 2)

        if self.current_start_frame == 0:
            clip_feas, ys, image = self.encode_image(image, self.num_frames, height, width)
            self.clip_feas = clip_feas.to(dtype=image.dtype)
            self.ys = ys.to(dtype=image.dtype)
        
        assert self.clip_feas is not None and self.ys is not None, "clip_feas and ys must be set"

        end_image_encoder_event.record()

        start_vae_event.record()

        if latent_video is not None and self.current_start_frame != 0:
            image = latent_video
            if self.ip_rank == 0:
                print("image shape@@", image.shape)
        elif self.current_start_frame != 0:
            # this is for real world execution
            if (videos.shape[2] - 1) // 4 == self.num_frame_per_block:
                print("no further action")
            elif videos.shape[2] // 4 != self.num_frame_per_block:
                # Repeating videos along dim 2.
                repeat_factor = self.num_frame_per_block // (videos.shape[2] // 4)
                videos = torch.repeat_interleave(videos, repeat_factor, dim=2)
            
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
            else: 
                first_frame = videos[:, :, 0:1]  # Extract first frame
                videos = torch.cat([first_frame, videos], dim=2)
           
            image = self.vae.encode(
                videos,
                tiled=self.tiled,
                tile_size=(self.tile_size_height, self.tile_size_width),
                tile_stride=(self.tile_stride_height, self.tile_stride_width),
            )

        end_vae_event.record()

        noise_obs = self.generate_noise((image.shape[0], 16, self.num_frame_per_block, height//8, width//8), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        noise_action = self.generate_noise((image.shape[0], self.action_horizon, self.model.action_dim), seed=self.seed, device='cuda', dtype=torch.bfloat16)
        batch_size, num_channels, num_frames, height, width = noise_obs.shape
        ######### Generate video #########
        # Match the DiT's (1, 2, 2) patch embedding tokenization on latent frames.
        frame_seqlen = (height // 2) * (width // 2)
        seq_len = frame_seqlen * num_frames

        image = image.transpose(1, 2)
        noise_obs = noise_obs.transpose(1, 2)

        if self.current_start_frame == 0:
            # Reinitialize KV cache and crossattn cache for each new sequence.
            self.kv_cache1, self.kv_cache_neg = self._create_kv_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
                frame_seqlen=frame_seqlen,
            )
            self.crossattn_cache, self.crossattn_cache_neg = self._create_crossattn_caches(
                batch_size=batch_size,
                dtype=noise_obs.dtype,
                device=noise_obs.device,
            )

        assert self.kv_cache1 is not None
        assert self.kv_cache_neg is not None
        assert self.crossattn_cache is not None
        assert self.crossattn_cache_neg is not None
        kv_caches = self._get_caches(
            [self.kv_cache1, self.kv_cache_neg],
        )
        crossattn_caches = self._get_caches(
            [self.crossattn_cache, self.crossattn_cache_neg],
        )

        start_kv_event.record()

        if self.current_start_frame == 0:
            timestep = torch.ones([batch_size, 1], device=noise_obs.device, dtype=torch.int64) * 0
            self._run_diffusion_steps(
                noisy_input=image.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=frame_seqlen,
                y=self.ys[:, :, 0:1],
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=0,
                    update_kv_cache=True,
                ),
            )
            self.current_start_frame += 1
            
        timestep = torch.ones([batch_size, self.num_frame_per_block], device=noise_obs.device, dtype=torch.int64) * 0

        if self.current_start_frame != 1:
            current_ref_latents = image[:, -self.num_frame_per_block:]
            if self.current_start_frame <= self.ys.shape[2]:
                y = self.ys[:, :, self.current_start_frame - self.num_frame_per_block : self.current_start_frame]
            else:
                y = self.ys[:, :, -self.num_frame_per_block:]
            self._run_diffusion_steps(
                noisy_input=current_ref_latents.transpose(1, 2),
                timestep=timestep * 0,
                action=None,
                timestep_action=None,
                state=None,
                embodiment_id=None,
                context=prompt_embs,
                seq_len=seq_len,
                y=y,
                clip_feature=self.clip_feas,
                kv_caches=kv_caches,
                crossattn_caches=crossattn_caches,
                kv_cache_metadata=dict(
                    start_frame=self.current_start_frame - self.num_frame_per_block,
                    update_kv_cache=True,
                ),
            )

        end_kv_event.record()

        noisy_input = noise_obs
        noisy_input_action = noise_action

        # Step 3.1: Spatial denoising loop

        sample_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler_action = FlowUniPCMultistepScheduler(
            num_train_timesteps=self.scheduler.num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False)
        sample_scheduler.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)
        sample_scheduler_action.set_timesteps(
            self.num_inference_steps, device=noise_obs.device, shift=self.sigma_shift)

        # Decoupled inference: video sigmas end at video_final_noise instead of 0
        # This rescales the schedule so video still takes all denoising steps, 
        # but ends at a higher noise level (e.g., 1.0 → 0.9 → 0.8 instead of 1.0 → 0.5 → 0.0)
        if self.config.decouple_inference_noise:
            video_final_noise = self.config.video_inference_final_noise
            # Rescale video sigmas: map [sigma_max, 0] -> [sigma_max, video_final_noise]
            sigma_max = sample_scheduler.sigmas[0].item()
            sample_scheduler.sigmas = sample_scheduler.sigmas * (sigma_max - video_final_noise) / sigma_max + video_final_noise
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64)
            if self.ip_rank == 0:
                print(f"Decoupled inference: video sigmas {sigma_max:.3f} -> {sample_scheduler.sigmas[-1].item():.3f}")

        start_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        end_diffusion_events = [torch.cuda.Event(enable_timing=True) for _ in sample_scheduler.timesteps]
        prev_predictions = [] 
        self.skip_countdown = 0
        dit_compute_steps = 0
        for index, current_timestep in enumerate(sample_scheduler.timesteps):
            start_diffusion_events[index].record()

            # Get timesteps from respective schedulers
            action_timestep = sample_scheduler_action.timesteps[index]
            video_timestep = sample_scheduler.timesteps[index]  # Already rescaled if decoupled

            # set current timestep
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * video_timestep
            timestep_action = torch.ones(
                [batch_size, self.action_horizon],
                device=noise_obs.device,
                dtype=torch.int64,
            ) * action_timestep

            # check if we need to run the DIT step
            should_run_model = self.should_run_model(index, current_timestep, prev_predictions)
            if should_run_model:
                dit_compute_steps += 1
                if self.current_start_frame + self.num_frame_per_block <= self.ys.shape[2]:
                    y = self.ys[:, :, self.current_start_frame : self.current_start_frame + self.num_frame_per_block]
                else:
                    y = self.ys[:, :, -self.num_frame_per_block:]
                predictions = self._run_diffusion_steps(
                    noisy_input=noisy_input.transpose(1, 2),
                    timestep=timestep,
                    action=noisy_input_action,
                    timestep_action=timestep_action,
                    state=state_features,
                    embodiment_id=embodiment_id,
                    context=prompt_embs,
                    seq_len=seq_len,
                    y=y,
                    clip_feature=self.clip_feas,
                    kv_caches=kv_caches,
                    crossattn_caches=crossattn_caches,
                    kv_cache_metadata=dict(
                        start_frame=self.current_start_frame,
                        update_kv_cache=False,
                    ),
                )
                flow_pred_cond, flow_pred_cond_action = predictions[0]
                flow_pred_uncond, flow_pred_uncond_action = predictions[1]

                flow_pred = flow_pred_uncond + self.cfg_scale * (flow_pred_cond - flow_pred_uncond)
                prev_predictions.append((current_timestep, flow_pred, flow_pred_cond_action))
                max_cache_size = 2
                if len(prev_predictions) > max_cache_size:
                    prev_predictions.pop(0)

            else:
                assert len(prev_predictions) > 0, "prev_predictions must be set when skipping"
                _, flow_pred, flow_pred_cond_action = prev_predictions[-1]

            end_diffusion_events[index].record()

            # Video: denoising step (uses rescaled schedule if decoupled)
            noisy_input = sample_scheduler.step(
                model_output=flow_pred.transpose(1, 2),
                timestep=video_timestep,
                sample=noisy_input,
                step_index=index,
                return_dict=False,
            )[0]
            
            # Action: always fully denoises with standard schedule (1000->0)
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

        if self.current_start_frame == 1:
            output = torch.cat([image, output], dim=1)
        self.current_start_frame += self.num_frame_per_block

        # Do torch.cuda.synchronize() to ensure all operations are completed before timing.
        # This isn't expected to affect inference performance since it's at the end of an inference step.
        torch.cuda.synchronize()

        total_time = time.perf_counter() - start_time
        text_encoder_time = start_text_encoder_event.elapsed_time(end_text_encoder_event) / 1000
        image_encoder_time = start_image_encoder_event.elapsed_time(end_image_encoder_event) / 1000
        vae_time = start_vae_event.elapsed_time(end_vae_event) / 1000
        kv_creation_time = start_kv_event.elapsed_time(end_kv_event) / 1000
        diffusion_times = [s.elapsed_time(e) for s, e in zip(start_diffusion_events, end_diffusion_events)]
        diffusion_time = sum(diffusion_times) / 1000
        scheduler_time = total_time - kv_creation_time - diffusion_time - text_encoder_time - image_encoder_time - vae_time

        if self.ip_rank == 0:
            print(f"Time taken: Total {total_time:.2f} seconds, "
                  f"Text Encoder {text_encoder_time:.2f} seconds, "
                  f"Image Encoder {image_encoder_time:.2f} seconds, "
                  f"VAE {vae_time:.2f} seconds, "
                  f"KV Cache Creation {kv_creation_time:.2f} seconds, "
                  f"Diffusion {diffusion_time:.2f} seconds, "
                  f"DIT Compute Steps {dit_compute_steps} steps, "
                  f"Scheduler {scheduler_time:.2f} seconds")

        return BatchFeature(data={"action_pred": latents_action, "video_pred": output.transpose(1, 2)})
    
    def cache_predict_order1(self, current_timestep, timestep_1, f1, timestep_2, f2):
        h_curr = current_timestep - timestep_1
        h_past = timestep_1 - timestep_2

        v_prime = (f1 - f2) / h_past

        # Prediction 
        damping_factor = 0.25
        flow_pred = f1 + (v_prime * h_curr) * damping_factor
        return flow_pred

    def post_initialize(self):
        # Move models to the cuda device and set the dtype to bfloat16.
        print("Moving models to the cuda device and setting the dtype to bfloat16.")
        self.model.to(device=self._device, dtype=torch.bfloat16)
        self.text_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.image_encoder.to(device=self._device, dtype=torch.bfloat16)
        self.vae.to(device=self._device, dtype=torch.bfloat16)
        import os
        ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"
        LOAD_TRT_ENGINE = os.getenv("LOAD_TRT_ENGINE", None)

        # Torch compile the modules.
        if not ENABLE_TENSORRT:
            print("Torch compiling the Wan, TextEncoder, ImageEncoder, and VAE modules.")

            self.model._forward_blocks = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.model._forward_blocks)

            self.text_encoder.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.text_encoder.forward)

            self.image_encoder.model.visual.forward = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.image_encoder.model.visual.forward)

            self.vae.model.encode = torch.compile(
                mode="reduce-overhead", fullgraph=True, dynamic=False,
            )(self.vae.model.encode)
        
        self.trt_engine = None
        if LOAD_TRT_ENGINE is not None:
            print(f"Loading TRT engine from {LOAD_TRT_ENGINE}")
            import groot.control.tensorrt_utils as trt_utils
            model_path = LOAD_TRT_ENGINE
            self.trt_engine = trt_utils.load_tensorrt_engine(model_path, model_type="ar_14B")

    def parallelize(self, device_mesh: DeviceMesh) -> None:
        ip_mesh = device_mesh["ip"]
        self.ip_rank = ip_mesh.get_local_rank()
        self.ip_size = ip_mesh.size()
        self.ip_group = ip_mesh.get_group()

        assert self.ip_size == 1 or self.ip_size == 2, "ip_size must be 1 or 2"
        assert self.ip_rank >= 0 and self.ip_rank < self.ip_size, "ip_rank must be in [0, ip_size)"

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
