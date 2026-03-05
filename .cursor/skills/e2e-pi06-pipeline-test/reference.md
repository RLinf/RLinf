# Pi06 Pipeline Test Reference

## Key Code Paths

### compute_returns
- Entry: `examples/process/compute_returns.py` (Hydra main)
- Shell: `examples/process/run_compute_returns.sh`
- Config: `examples/process/config/compute_returns_test.yaml`
- Key function: `compute_returns_for_episode()` — backward iteration `G_t = r_t + gamma * G_{t+1}`
- Output: Modifies parquet in-place, updates `meta/stats.json` and `meta/info.json`

### vla_lib_sft (value model training)
- Entry: `examples/vla_lib_sft/train_vla_lib_sft.py` (Hydra main)
- Shell: `examples/vla_lib_sft/run_vla_lib_sft.sh`
- Config: `examples/vla_lib_sft/config/libero_value_model_test.yaml`
- Runner: `rlinf/runners/vla_lib_sft_runner.py` → `VlaLibSFTRunner`
- Worker: `rlinf/workers/vla_lib_sft/fsdp_value_sft_worker.py` → `FSDPValueSftWorker`
- Model: `rlinf/models/embodiment/vla_lib_value_model/` (base_policy, modeling_critic, value_policy_config)
- Dataset: `rlinf/datasets/vla_lib/value_dataset.py`, `value_mixture_dataset.py`
- Transforms: `rlinf/datasets/vla_lib/io_processing/value_transforms.py` (ReturnDiscretizer, ReturnNormalizer)
- Output: Checkpoint at `logs/vla_lib_sft/*/checkpoints/global_step_*/actor/model_state_dict/`
- Ckpt conversion: `rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf_vla_lib.sh`

### compute_advantages
- Entry: `examples/process/compute_advantages.py` (Hydra main, supports torchrun)
- Shell: `examples/process/run_compute_advantages.sh`
- Config: `examples/process/config/compute_advantages_test.yaml`
- Key functions: `load_value_policy()`, `compute_advantages_for_dataset()`, `save_advantages_to_dataset()`
- Formula: `A = normalize(r_{t:t+N}) + gamma^N * V(o_{t+N}) - V(o_t)`
- Output: `meta/advantages_{tag}.parquet` with columns: advantage, is_success, value_current, value_next

### cfg_sft
- Entry: `examples/cfg/train_cfg_sft.py` (Hydra main)
- Shell: `examples/cfg/run_cfg_sft.sh`
- Config: `examples/cfg/config/libero_cfg_sft_test.yaml`
- Runner: `rlinf/runners/sft_runner.py` → `SFTRunner`
- Worker: `rlinf/workers/cfg/fsdp_cfg_worker.py` → `FSDPCfgWorker`
- Dataset wrapper: `rlinf/workers/cfg/utils.py` → `DatasetWithAdvantage`
- Model: `rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py`
- Ckpt conversion: `rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh`

### debug_one_iter
- Entry: `examples/embodiment/train_debug_one_iter.py` (Hydra main)
- Shell: `examples/embodiment/run_debug_one_iter.sh`
- Config: `examples/embodiment/config/one_iter_debug_libero10_test.yaml`
- Runner: `rlinf/runners/debug_pi06_runner.py` → `DebugPi06Runner`
- Worker: `rlinf/workers/actor/debug_fsdp_actor_worker_cfg.py` → `DebugCFGFSDPActor`
- Ckpt conversion: `rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh`

---

## Data Format

### LeRobot Dataset Structure
```
dataset_dir/
├── data/
│   └── chunk-000/
│       └── episode_000000.parquet  (columns: state, action, image, ...)
├── meta/
│   ├── info.json                  (features, fps, total_episodes, ...)
│   ├── stats.json                 (return/reward min/max/mean/std)
│   ├── tasks.jsonl                (task descriptions)
│   ├── episodes.jsonl             (episode metadata)
│   └── advantages_test.parquet    (computed by Step 3)
└── videos/
    └── chunk-000/
        └── image/
            └── episode_000000.mp4
```

### Advantage Parquet Schema (after Step 3)
| Column | Type | Description |
|--------|------|-------------|
| advantage | float32 | TD(N) advantage value |
| is_success | bool | True if advantage >= threshold (top 30%) |
| value_current | float32 | V(o_t) from value model |
| value_next | float32 | V(o_{t+N}) from value model |

---

## Common Errors & Fixes (Verified in E2E Testing)

### 1. pathlib.Path NameError
**Symptom**: `NameError: name 'pathlib' is not defined` in value_mixture_dataset.py
**Fix**: Use `Path` instead of `pathlib.Path` (only `from pathlib import Path` is imported)

### 2. global_batch_size must be divisible
**Symptom**: `actor.global_batch_size must be divisible by (micro_batch_size * actor_world_size)`
**Fix**: Set `global_batch_size = micro_batch_size × num_gpus` (e.g., 2 × 8 = 16)

### 3. Paligemma tokenizer gated repo
**Symptom**: `OSError: ...paligemma-3b-pt-224 is a gated repo`
**Fix**: Pass `tokenizer_name_or_path=model_path` to PI05Processor, or copy tokenizer files locally

### 4. Paligemma tokenizer download hang
**Symptom**: Workers stuck at `Downloading gs://big_vision/paligemma_tokenizer.model`
**Fix**: Copy tokenizer to cache: `cp paligemma-3b-mix-224/tokenizer.model ~/.cache/openpi/big_vision/paligemma_tokenizer.model`

### 5. FSDP parameter shape writeback error
**Symptom**: `RuntimeError: Cannot writeback when the parameter shape changes`
**Fix**: Set both `use_orig_params: False` and `gradient_checkpointing: False` (or True separately)

### 6. Checkpoint conversion import error
**Symptom**: `ImportError: attempted relative import with no known parent package`
**Fix**: Use absolute import: `from rlinf.utils.ckpt_convertor.fsdp_convertor.utils import ...`

### 7. Hydra path truncation with colons
**Symptom**: Hydra config paths truncated at colon characters (timestamps like `09:27:09`)
**Fix**: Use symlink: `ln -sfn <path_with_colons> /tmp/stepN_ckpt`

### 8. ValueCriticModel get_parameter_or_buffer
**Symptom**: `AttributeError: 'ValueCriticModel' object has no attribute 'get_parameter_or_buffer'`
**Fix**: Add `AttributeError` to except clause in `value_policy_config.py` `from_pretrained` fallback

### 9. Abstract method default_forward
**Symptom**: `TypeError: Can't instantiate abstract class OpenPi0ForCFGActionPrediction with abstract method default_forward`
**Fix**: Add `def default_forward(self, **kwargs): return self.forward(**kwargs)` to the class

### 10. norm_stats.json not found
**Symptom**: `FileNotFoundError: Norm stats file not found at: .../pi05_base_pytorch/physical-intelligence/libero/norm_stats.json`
**Fix**: Copy norm_stats.json from reference repo checkpoint to `{model_path}/physical-intelligence/libero/`

### 11. CUDA OOM during value model training
Reduce batch sizes in `libero_value_model_test.yaml`:
```yaml
actor:
  micro_batch_size: 1
  global_batch_size: 8  # = micro_batch_size × num_gpus
```

### 12. Advantage tag consistency
The advantage tag must match between Step 3 (`advantage.tag: "test"`) and Steps 4/5 (`data.advantage_tag: "test"`).

---

## Reference Repo Comparison Summary

| File | Clean | Reference | Verdict |
|------|-------|-----------|---------|
| compute_returns.py | failure_reward required | failure_reward defaults to -300 | Improvement |
| compute_advantages.py | Simplified batch inference | Combined current+next inference | Simplification |
| train_vla_lib_sft.py | VlaLibSFTRunner | SFTRunner | New dedicated runner |
| train_cfg_sft.py | Identical | Identical | No change |
| train_debug_one_iter.py | Renamed | train_debug_pi06.py | Rename for clarity |
