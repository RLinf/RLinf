# XR0 VLA 接入 RLinf 交接文档

## 项目概述

将小米 XR0 VLA 模型接入 RLinf 框架，支持推理和 RL 训练。

XR0 架构：Qwen3-VL（4B） + DiT（16层） + Rectified Flow（5步去噪）
动作空间：30步 × 32维（双臂：位置、旋转、夹爪、关节角）

## 已完成的工作（7个commit）

### PR 1：模型骨架
- 注册 `model_type: "xr0"` 到 RLinf
- 创建 `rlinf/models/embodiment/xr0/` 包
- stub 模型、config YAML、单元测试

### PR 2：真实推理
- Qwen3-VL processor 接入（图片→PIL→resize→tokenize）
- `predict_action_batch` 完整实现
- 加载真实权重（`AutoModel.from_pretrained`）
- 已验证：LIBERO checkpoint 推理输出 (1, 30, 32)

### PR 3：RL训练支持
- `sample_actions`：逐步去噪，记录chain
- `get_logprob_norm`：计算高斯log概率
- `default_forward`：重放chain，算logprobs/entropy
- 简化版Flow-SDE（固定噪声，单步随机）

### PR 4：TODO标记
- 所有待做项已标TODO注释

## 关键文件

```
rlinf/models/embodiment/xr0/
├── __init__.py              # get_model()工厂，_StubXR0
├── xr0_action_model.py      # XR0ForRLActionPrediction（核心）
├── utils.py                 # ACTION_DIM, normalize/denormalize, resize_image
└── model/                   # 从xr0_src搬运的模型代码（暂时不用，用AutoModel加载）
    ├── xr0_model.py
    ├── qwen3vl.py
    └── __init__.py

examples/embodiment/config/model/
└── xr0.yaml                 # 模型配置

tests/unit_tests/
└── test_xr0_model_registration.py  # 6个测试
```

## 当前能力

| 功能 | 状态 | 备注 |
|------|------|------|
| 模型注册 | ✅ | `model_type: "xr0"` |
| stub测试 | ✅ | `model_path: "dummy"` |
| 真实推理 | ✅ | CPU可用，GPU需≥24GB |
| predict_action_batch | ✅ | 返回动作+chain |
| default_forward | ⚠️ | logprobs可用，values返回零 |
| Value Head | ❌ | TODO |
| 可学习噪声 | ❌ | TODO |
| Flow-SDE完整版 | ❌ | TODO |
| 文档/CI | ❌ | TODO |

## 待做事项（搜TODO找到）

### 高优先级
1. **Value Head**：用VLM隐藏状态做critic（PPO需要）
   - 参考：`rlinf/models/embodiment/modules/value_head.py`
   - 参考：lingbotvla的`get_value_from_vlm`方法

2. **Flow-SDE完整版**：每步加噪声，随时间步变化
   - 噪声调度：`σ = a√(τ/(1-τ))`
   - 漂移修正：`σ²δ/(2τ)`
   - 参考：lingbotvla的`sample_mean_var_val`

### 中优先级
3. **ExploreNoiseNet**：可学习噪声网络（π_RL Flow-Noise）
   - 参考：`rlinf/models/embodiment/modules/explore_noise_net.py`

4. **多种噪声方法**：支持flow_sde/flow_noise/flow_cps
   - 参考：lingbotvla的noise_method配置

### 低优先级
5. **LIBERO环境集成**：端到端测试
6. **文档**：RST文档（EN+ZH）
7. **CI**：Docker、e2e测试

## 参考实现

最重要的参考是 **lingbotvla**：
```
rlinf/models/embodiment/lingbotvla/lingbotvla_action_model.py
```

它实现了完整的flow-matching RL训练：
- `sample_actions`：逐步去噪+chain录制
- `sample_mean_var_val`：三种噪声方法
- `get_logprob_norm`：log概率计算
- `get_value_from_vlm`：VLM做critic
- `default_forward`：chain重放

## 权重下载

```bash
# 用hf-mirror下载LIBERO checkpoint
HF_ENDPOINT=https://hf-mirror.com python download_xr0.py

# 权重位置
/home/sw/models/Xiaomi-Robotics-0-LIBERO/
```

## 环境配置

```bash
conda create -n rlinf python=3.11.14
pip install -e .
pip install transformers==4.57.1
pip install torchvision
```

注意：transformers版本必须是4.57.1，更高版本不兼容。

## 测试

```bash
conda run -n rlinf python -m pytest tests/unit_tests/test_xr0_model_registration.py -v
```

## 已知问题

1. GPU显存不足（11.6GB）：真实模型需≥24GB
2. stub模型的logprobs/entropy是零（正常，stub没有真实去噪）
3. transformers版本锁定4.57.1（XR0自定义代码依赖）

## Git分支

```
feat/xr0-vla  →  https://github.com/sunwen-me/RLinf/tree/feat/xr0-vla
```

7个commit，全部已push。
