---
name: e2e-pi06-pipeline-test
description: "Pi06 pipeline E2E test: compute_returns → train value model → compute_advantages → cfg_sft → debug_one_iter"
---

# Pi06 Pipeline E2E Test

顺序执行 pi06 全流程测试，验证 open-source release 前所有新模块的正确性。

仅修改各 task_example 中"可变参数"部分，禁止更改测试命令中的任何其他内容。

---

## Pipeline 概览

```
[Step 1] compute_returns   → 添加 return/reward/prompt 列 (SFT + rollout 数据集)
[Step 2] vla_lib_sft       → 训练 value model (5 steps) → checkpoint → ckpt转换 (HF SafeTensors)
[Step 3] compute_advantages → 计算 TD(N) advantages → advantages_test.parquet (True/False)
[Step 4] cfg_sft            → 使用 advantage 训练 CFG 模型 (5 steps) → ckpt转换
[Step 5] debug_one_iter     → 完整 pi06 调试循环 (5 steps, 无环境评估) → ckpt转换
```

---

## Task Example 索引

| 步骤 | 文件 | 说明 |
|------|------|------|
| Step 1 | `task_example/step1_compute_returns.md` | 计算 returns |
| Step 2 | `task_example/step2_train_value_model.md` | 训练 value model + ckpt转换 |
| Step 3 | `task_example/step3_compute_advantages.md` | 计算 advantages |
| Step 4 | `task_example/step4_cfg_sft.md` | CFG SFT 训练 + ckpt转换 |
| Step 5 | `task_example/step5_debug_one_iter.md` | Debug 一次迭代 + ckpt转换 |

---

## 测试配置索引

| 步骤 | 配置文件 | 说明 |
|------|---------|------|
| Step 1 | `examples/process/config/compute_returns_test.yaml` | SFT + rollout 数据集, gamma=1.0 |
| Step 2 | `examples/vla_lib_sft/config/libero_value_model_test.yaml` | 5步, freeze_vlm, batch=16 |
| Step 3 | `examples/process/config/compute_advantages_test.yaml` | 单GPU, batch=4, tag="test" |
| Step 4 | `examples/cfg/config/libero_cfg_sft_test.yaml` | 5步, batch=16, advantage_tag="test" |
| Step 5 | `examples/embodiment/config/one_iter_debug_libero10_test.yaml` | 5步, 无eval, 单GPU, batch=2 |

---

## 测试流程

### 前置条件

- GPU 可用 (CUDA), 8 GPUs for Steps 2/4, 1 GPU for Step 5
- openpi 环境: `source switch_env openpi`
- SFT 数据集: `/mnt/project_rlinf_hs/liuzhihao/data_to_use/libero_10_3shot`
- Rollout 数据集: `/mnt/project_rlinf_hs/liuzhihao/data_to_use/collected_data_40_eps`
- 基础模型: `/mnt/project_rlinf/liuzhihao/vla_lib_fork_gaofeng/pretrained_models/pi05_base_pytorch`
- norm_stats.json: `{base_model}/physical-intelligence/libero/norm_stats.json` (required for Steps 4/5)
- Paligemma tokenizer: `/root/.cache/openpi/big_vision/paligemma_tokenizer.model`

### 执行顺序

每个 step 包含两部分检查:

1. **逻辑核查**: 对比 clean repo 与原始 repo (`/mnt/project_rlinf/liuzhihao/RLinf_pi06_release_worktree_2`) 的代码逻辑
2. **运行检验**: 使用测试配置运行脚本，验证功能正确

**必须按顺序执行**，前一步通过后才能进入下一步。

### Checkpoint 转换

Steps 2/4/5 训练完成后需要进行 checkpoint 转换:

- **Step 2** (value model): `convert_pt_to_hf_vla_lib.sh` → SafeTensors (用于 Step 3)
- **Step 4** (CFG model): `convert_pt_to_hf.sh` → SafeTensors
- **Step 5** (debug model): `convert_pt_to_hf.sh` → SafeTensors

**注意**: 路径含冒号（时间戳）时需要用 symlink 避免 Hydra 解析错误。

### 自动化脚本

```bash
bash tests/e2e_tests/pi06_pipeline/run_pipeline_test.sh
```

可选环境变量:
- `SKIP_STEPS="1,2"` — 跳过指定步骤
- `KEEP_TEST_DATA=1` — 不清理测试数据

---

## Checklist

- [ ] Step 1: `compute_returns` — 两个数据集的 stats.json 包含 return/reward
- [ ] Step 2: `vla_lib_sft` — checkpoint 生成 + SafeTensors 转换成功
- [ ] Step 3: `compute_advantages` — 两个数据集的 advantages_test.parquet 生成，包含 True/False 标签
- [ ] Step 4: `cfg_sft` — 5步训练完成 + SafeTensors 转换成功
- [ ] Step 5: `debug_one_iter` — 5步离线训练完成 + SafeTensors 转换成功
