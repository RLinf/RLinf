# Pi06 E2E Test Progress

## 当前阶段: Phase D — 全部 5 步测试完成
## 最后更新: 2026-03-05

### 完成的阶段
- [x] Phase A: vla_lib 外部依赖内部化 — commit: 13bc731
  - 复制 openpi05 (9 files), openpi (8 files + transformers_replace)
  - 复制 dist_utils.py, image_utils.py
  - 复制 dataset config YAMLs (20 files)
  - 更新 ~10 个文件的 import 路径
  - 实现本地 get_dataset_config_dir()
- [x] Phase B: ckpt 转换脚本整理 — commit: 27c5d56
  - 新建 convert_pt_to_hf.sh
  - 修复 convert_pt_to_hf_vla_lib.sh 路径
- [x] Phase C: 深度逻辑核查 — commit: 81da445
  - 对比 reference repo，添加缺失的 clear_memory() 调用

### Phase D 进度 — ALL PASSED
- [x] Step 1: compute_returns — PASSED
  - 两个数据集 (libero_10_3shot + collected_data_40_eps) 均处理成功
  - stats.json 包含 return/reward 统计
- [x] Step 2: vla_lib_sft — PASSED
  - 5/5 训练步完成 (freeze_vlm, categorical loss)
  - checkpoint: logs/vla_lib_sft/.../global_step_5/actor/model_state_dict/full_weights.pt
  - 修复: pathlib.Path NameError, batch_size 2→16, tokenizer access, FSDP params
- [x] Step 2 ckpt conversion — PASSED
  - SafeTensors 生成: /tmp/step2_ckpt/model-0000{1-4}-of-00004.safetensors
  - 修复: relative import → absolute import, REPO_PATH dirname 层数, Hydra colon path
- [x] Step 3: compute_advantages — PASSED
  - 两个数据集处理成功: libero_10_3shot (8005 samples), collected_data_40_eps (15777 samples)
  - advantages_test.parquet 保存, 统一 threshold at 70th percentile: -0.0092
  - SFT: all True; rollout: 27.1% positive
  - 修复: AttributeError 'get_parameter_or_buffer' (添加 AttributeError 到 except clause)
- [x] Step 4: cfg_sft — PASSED
  - 5/5 训练步完成 (conditional/unconditional CFG training)
  - checkpoint: logs/cfg_sft/.../global_step_5/actor/model_state_dict/full_weights.pt
  - 修复: paligemma tokenizer 下载, abstract method default_forward, norm_stats.json, batch_size 2→16
- [x] Step 4 ckpt conversion — PASSED
  - SafeTensors 生成: /tmp/step4_ckpt/model-0000{1-3}-of-00003.safetensors
- [x] Step 5: debug_one_iter — PASSED
  - 5/5 offline 训练步完成 (single GPU, collected_data_40_eps)
  - checkpoint: logs/debug_one_iter_.../global_step_5/actor/model_state_dict/full_weights.pt
- [x] Step 5 ckpt conversion — PASSED
  - SafeTensors 生成: /tmp/step5_ckpt/model-0000{1-3}-of-00003.safetensors

### Phase D 中未提交的文件修改
- rlinf/datasets/vla_lib/value_mixture_dataset.py (pathlib.Path fix)
- rlinf/models/embodiment/vla_lib_value_model/openpi05/processing_pi05.py (tokenizer fix)
- rlinf/models/embodiment/vla_lib_value_model/value_policy_config.py (AttributeError fix)
- rlinf/models/embodiment/openpi_cfg/openpi_cfg_action_model.py (default_forward method)
- rlinf/workers/vla_lib_sft/fsdp_value_sft_worker.py (tokenizer_name_or_path)
- rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py (absolute import)
- rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.sh (REPO_PATH fix)
- rlinf/utils/ckpt_convertor/fsdp_convertor/convert_pt_to_hf_vla_lib.sh (REPO_PATH fix)
- examples/vla_lib_sft/config/libero_value_model_test.yaml (batch_size, FSDP params)
- examples/cfg/config/libero_cfg_sft_test.yaml (batch_size 2→16)
- examples/process/config/compute_returns_test.yaml (rollout dataset)
- examples/process/config/compute_advantages_test.yaml (rollout dataset)

### 待完成
- [ ] Phase D commit: 提交所有 Phase D 修复
- [ ] Phase F: 更新并部署 Skill
