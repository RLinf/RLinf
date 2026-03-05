# Pi06 E2E Test Memory

## 错误记录

### 1. NameError: name 'pathlib' is not defined
- **场景**: Step 2 vla_lib_sft 训练启动
- **错误信息**: `NameError: name 'pathlib' is not defined` in value_mixture_dataset.py
- **原因**: 用了 `pathlib.Path` 但只 import 了 `from pathlib import Path`
- **解决方案**: 改为使用 `Path` 而非 `pathlib.Path`

### 2. AssertionError: global_batch_size must be divisible
- **场景**: Step 2/4 训练
- **错误信息**: `actor.global_batch_size must be divisible by (micro_batch_size * actor_world_size)`
- **原因**: global_batch_size=2 在 8 GPU 环境不合法 (2 不能被 8*2=16 整除)
- **解决方案**: 改 global_batch_size 为 16

### 3. OSError: gated repo (paligemma-3b-pt-224)
- **场景**: Step 2 PI05Processor 初始化
- **错误信息**: `OSError: ...paligemma-3b-pt-224 is a gated repo`
- **原因**: PI05Processor 默认尝试从 HuggingFace 下载 tokenizer
- **解决方案**: (1) 复制本地 paligemma-3b-mix-224 的 tokenizer 文件到 pi05_base_pytorch; (2) 传 `tokenizer_name_or_path=model_cfg.model_path` 给 PI05Processor

### 4. RuntimeError: Cannot writeback when the parameter shape changes (FSDP)
- **场景**: Step 2 FSDP 训练
- **错误信息**: `RuntimeError: Cannot writeback when the parameter shape changes`
- **原因**: `use_orig_params=True` + `gradient_checkpointing=True` 与 FSDP 不兼容
- **解决方案**: 两者都设为 `False`

### 5. ImportError: attempted relative import with no known parent package
- **场景**: Step 2 ckpt 转换 (convert_pt_to_hf.py)
- **错误信息**: `ImportError: attempted relative import with no known parent package`
- **原因**: 脚本直接执行时 `.utils` 相对 import 失败
- **解决方案**: 改为 `from rlinf.utils.ckpt_convertor.fsdp_convertor.utils import ...`

### 6. Hydra config resolution error (model/pi0_5)
- **场景**: Step 2 ckpt 转换
- **错误信息**: `Could not load 'model/pi0_5'`
- **原因**: REPO_PATH 计算错误，少了一层 dirname
- **解决方案**: convert_pt_to_hf.sh 和 convert_pt_to_hf_vla_lib.sh 中 REPO_PATH 从 3 层 dirname 改为 4 层

### 7. Hydra path truncation with colons
- **场景**: Step 2/4/5 ckpt 转换
- **错误信息**: Hydra 参数被冒号截断
- **原因**: 路径包含时间戳 (如 09:27:09)，Hydra 将冒号视为配置分隔符
- **解决方案**: 用 symlink 避免冒号: `ln -sfn <path_with_colons> /tmp/stepN_ckpt`

### 8. pytz ImportError
- **场景**: Step 3 compute_advantages
- **错误信息**: pandas 无法确定 pytz 版本
- **解决方案**: `pip install pytz`

### 9. ValueCriticModel has no attribute 'get_parameter_or_buffer'
- **场景**: Step 3 compute_advantages, PI05ValueCritic.from_pretrained()
- **错误信息**: `AttributeError: 'ValueCriticModel' object has no attribute 'get_parameter_or_buffer'`
- **原因**: transformers 4.53.2 定义 `get_parameter_or_buffer` 在 PreTrainedModel 上。`from_pretrained` 检测到 `base_model_prefix="model"`，将 `model_to_load` 设为 `ValueCriticModel` (nn.Module，没有此方法)
- **解决方案**: 在 value_policy_config.py:290 的 except clause 中添加 `AttributeError`，允许 fallback 到 direct state dict loading

### 10. Paligemma tokenizer download hang
- **场景**: Step 4 cfg_sft 训练
- **错误信息**: Workers 卡在 `INFO:openpi.shared.download:Downloading gs://big_vision/paligemma_tokenizer.model`
- **原因**: openpi 尝试从 GCS 下载 tokenizer，网络不可达
- **解决方案**: 复制本地 paligemma-3b-mix-224/tokenizer.model 到 `/root/.cache/openpi/big_vision/paligemma_tokenizer.model`

### 11. Abstract method default_forward
- **场景**: Step 4 cfg_sft 训练
- **错误信息**: `TypeError: Can't instantiate abstract class OpenPi0ForCFGActionPrediction with abstract method default_forward`
- **原因**: BasePolicy(ABC) 有 @abstractmethod default_forward，但 OpenPi0ForCFGActionPrediction 未实现
- **解决方案**: 添加 `def default_forward(self, **kwargs): return self.forward(**kwargs)` 到 OpenPi0ForCFGActionPrediction

### 12. norm_stats.json not found
- **场景**: Step 4 cfg_sft 训练
- **错误信息**: `FileNotFoundError: Norm stats file not found at: .../pi05_base_pytorch/physical-intelligence/libero/norm_stats.json`
- **原因**: openpi 的 `_checkpoints.load_norm_stats()` 要求 norm_stats.json 在 `{model_path}/{asset_id}/norm_stats.json`
- **解决方案**: 从 reference repo 的训练日志中复制 norm_stats.json 到 `pi05_base_pytorch/physical-intelligence/libero/`
