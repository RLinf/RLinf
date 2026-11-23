# mjlab Deployment Toolkit

Utilities for exporting RLinf-trained `mjlab` MLP policies to standalone ONNX files for downstream robot runtimes (such as `pollen-robotics/microduck`).

## Export an Actor Checkpoint to ONNX

Install `onnx` in your active environment if it is not already installed (`uv pip install "onnx>=1.20.1"`), then run from the repository root (where `<train_log_path>` is the training run directory, such as `./logs/<timestamp>-mjlab_microduck_ppo_mlp`):

```bash
PYTHONPATH=$(pwd) python3 toolkits/mjlab/export_onnx.py \
    --ckpt <train_log_path>/mjlab_microduck_ppo_mlp/checkpoints/global_step_5000/actor/model_state_dict/full_weights.pt \
    --output microduck_walk.onnx
```

The exported graph fuses the running observation normalizer into the deterministic actor, taking `obs` (`[1, 61]`) and returning `actions` (`[1, 14]`).
