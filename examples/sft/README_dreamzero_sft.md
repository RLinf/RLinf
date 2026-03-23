# DreamZero SFT Example

This example launches DreamZero supervised fine-tuning through RLinf's VLA SFT runner.

## 1. Minimal config fields to update

Edit `examples/sft/config/dreamzero_sft_droid.yaml`:

- `data.train_data_paths`
  Example: `/data/lerobot/droid`
- `actor.model.model_path`
  Example: `/data/checkpoints/DreamZero-AgiBot`
- `actor.model.pretrained_model_path`
  Usually the same as `actor.model.model_path`
- `actor.model.train_cfg_path`
  Example: `/data/checkpoints/DreamZero-AgiBot/experiment_cfg/conf.yaml`

If your DreamZero train config uses a different dataset root key, also update:

- `actor.model.data_root_key`

## 2. Launch command

Run from the RLinf repo root:

```bash
bash examples/sft/run_dreamzero_sft.sh dreamzero_sft_droid \
  data.train_data_paths=/path/to/droid_dataset \
  actor.model.model_path=/path/to/DreamZero-AgiBot \
  actor.model.pretrained_model_path=/path/to/DreamZero-AgiBot \
  actor.model.train_cfg_path=/path/to/DreamZero-AgiBot/experiment_cfg/conf.yaml \
  actor.model.data_root_key=droid_data_root
```

## 3. Direct python launch

```bash
python examples/sft/train_dreamzero_sft.py \
  --config-path examples/sft/config \
  --config-name dreamzero_sft_droid \
  data.train_data_paths=/path/to/droid_dataset \
  actor.model.model_path=/path/to/DreamZero-AgiBot \
  actor.model.pretrained_model_path=/path/to/DreamZero-AgiBot \
  actor.model.train_cfg_path=/path/to/DreamZero-AgiBot/experiment_cfg/conf.yaml \
  actor.model.data_root_key=droid_data_root
```

## 4. Common overrides

Single node, 8 GPUs:

```bash
bash examples/sft/run_dreamzero_sft.sh dreamzero_sft_droid \
  cluster.component_placement.actor=all \
  actor.micro_batch_size=4 \
  actor.global_batch_size=32
```

Resume from an RLinf checkpoint:

```bash
bash examples/sft/run_dreamzero_sft.sh dreamzero_sft_droid \
  runner.resume_dir=/path/to/results/.../checkpoints/global_step_1000 \
  data.train_data_paths=/path/to/droid_dataset \
  actor.model.model_path=/path/to/DreamZero-AgiBot \
  actor.model.pretrained_model_path=/path/to/DreamZero-AgiBot \
  actor.model.train_cfg_path=/path/to/DreamZero-AgiBot/experiment_cfg/conf.yaml
```
