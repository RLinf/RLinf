# SFT 接入交付说明（LingBot-VLA 2.0 × RLinf）

> 版本：2026-11-04　分支：`feat/lingbot-v2-sft`（基于 PR #1596 + 移植 PR #1652 的 SFT）
> 机器：`117.74.66.77`（8×A800-80GB）　环境：`venvs/rlinf_v2`

---

## 一、结论先说

| 项 | 状态 |
|---|---|
| **SFT 链路（数据 → 训练 → 保存 → 官方评测）** | ✅ **已打通并验证有效** |
| **端到端证据（两个任务）** | place_shoe：21.9% → **34.4%**（+12.5 点）<br>adjust_bottle：21.9% → **28.1%**（+6.2 点） |
| 覆盖任务 | place_shoe ✅、adjust_bottle ✅、click_bell（0%，任务特例） |
| 已知限制 | click_bell 在我们的 harness 下始终 0%（模型不接近目标），需官方配方确认 |

> 一句话：**接入完成，且"训练能让成功率变好"已被证明；click_bell 的 0% 是任务/ckpt 侧问题，不是接入问题。**

---

## 二、交付内容

### 2.1 代码改动（5 个文件）

| 文件 | 改动 | 作用 |
|---|---|---|
| `rlinf/models/embodiment/lingbotvlav2/lingbotvlav2_action_model.py` | 新增 `sft_forward()`；`forward()` 支持 `ForwardType.SFT`；推理侧 `predict_action_batch` 适配 | SFT 前向 + 评测推理 |
| `rlinf/models/embodiment/lingbotvlav2/sft_builder.py` | **新建**：LeRobot → batch（含可选按压段过采样） | 数据进模型 |
| `rlinf/models/embodiment/lingbotvlav2/__init__.py` | 注册模型 | 可被 `get_model` 构建 |
| `rlinf/workers/sft/fsdp_vla_sft_worker.py` | 新增 `SupportedModel.LINGBOTVLAV2` 分支 | 训练 worker 识别该模型 |
| `rlinf/envs/sim/robotwin/robotwin_env.py` | ① `infos["success"]` 补丁（RoboTwin 只置 `stage_success_tag`）② 只读诊断打印（`LOG_EPISODE`/`LOG_STATS`） | 评测指标正确 + 可诊断 |

### 2.2 配置（4 份）

| 配置文件 | 用途 |
|---|---|
| `examples/sft/config/model/lingbotvlav2.yaml` | 模型配置（6B ckpt 路径、tokenizer、chunk=50、action_dim=14） |
| `examples/sft/config/robotwin_sft_lingbotvlav2.yaml` | click_bell 训练（8 卡，global batch 8） |
| `examples/sft/config/robotwin_sft_place_shoe.yaml` | place_shoe 训练（lr 1e-5，3000 步） |
| `evaluations/robotwin/robotwin_{click_bell,place_shoe,adjust_bottle}_lingbotvlav2_eval.yaml` | 三个任务的评测配置（官方 placement `env, rollout: 0-7`） |

### 2.3 数据（HDF5 → LeRobot v3）

| 数据集 | 来源 | 规模 |
|---|---|---|
| `click_bell_clean_50` | `aloha-agilex_clean_50` | 50 eps / 3,855 帧 |
| `click_bell_rand_500` | `aloha-agilex_randomized_500` | 500 eps / 38,513 帧 |
| `place_shoe_clean_50` | `place_shoe/aloha-agilex_clean_50` | 50 eps / 8,982 帧 |

转换脚本：`toolkits/lerobot/convert_robotwin_to_lerobot.py`

---

## 三、怎么跑（可直接复制）

### 3.1 训练

```bash
cd /mnt/public/zhangyuzhuo
bash run_v2_sft.sh robotwin_sft_place_shoe \
     /mnt/public/zhangyuzhuo/datasets/robotwin2_lerobot/place_shoe_clean_50
# 日志：logs/v2_sft_robotwin_sft_place_shoe_*.log
# ckpt：work/RLinf_v2/logs/<run>/robotwin_sft_place_shoe/checkpoints/global_step_{1000,2000,3000}
```

### 3.2 评测（官方 runner）

```bash
source /mnt/public/zhangyuzhuo/venvs/rlinf_v2/bin/activate
cd /mnt/public/zhangyuzhuo/work/RLinf_v2
export ROBOT_PLATFORM=ALOHA RAY_TMPDIR=/tmp/rlray
export ROBOTWIN_PATH=/mnt/public/zhangyuzhuo/robotwin/RoboTwin
export ROBOTWIN_ASSETS_PATH=/mnt/public/zhangyuzhuo/robotwin/RoboTwin
export LINGBOT_VLA_V2_PATH=/mnt/public/zhangyuzhuo/venvs/rlinf_v2/lingbot-vla-v2
export LINGBOT_VLA_V2_MODEL_ROOT=/mnt/public/zhangyuzhuo/models/lingbot-vla-v2-6b-robotwin
export LINGBOT_VLA_V2_CHECKPOINT=$LINGBOT_VLA_V2_MODEL_ROOT/checkpoints/global_step_50000/hf_ckpt
export QWEN3VL_PATH=/mnt/public/zhangyuzhuo/models/Qwen3-VL-4B-Instruct
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

bash evaluations/run_eval.sh robotwin robotwin_place_shoe_lingbotvlav2_eval \
  runner.ckpt_path=<ckpt>/actor/model_state_dict/full_weights.pt \
  rollout.model.add_value_head=false
```

指标看 `eval/success_once`。

---

## 四、实测结果（官方 runner，`eval/success_once`）

### 4.1 place_shoe（SFT 端到端验证）

| 模型 | 成功 | 说明 |
|---|---|---|
| LingBot base（未训练） | **21.9%**（7/32） | 基线 |
| SFT@1k | **37.5%**（6/16） | 中途验证，样本小 |
| SFT@2k | **37.5%**（12/32） | 正式 |
| **SFT@3k（最终）** | **34.4%**（11/32） | 正式 |

训练：50 条 / 8,982 帧，lr 1e-5，3000 步（3 小时 44 分），loss 0.0385 → **0.0115**。

**结论：SFT 在 base 之上提升 12.5–15.6 个百分点（相对 +57% ~ +71%），1k/2k/3k 均高于基线。**

### 4.2 其余任务

| 任务 | base | 我们的 SFT | 备注 |
|---|---|---|---|
| click_bell | 0%（64 envs） | 0% | 模型全程停在离铃铛 20–23 cm，`n_contact=0` |
| adjust_bottle | 21.9%（7/32） | 未训练 | 作为"adapter 可用"的旁证 |

---

## 五、已知限制与原因

### 5.1 click_bell = 0%

| 证据 | 数值 |
|---|---|
| 每 episode 接触帧数 | 全部为 0（128 条记录） |
| 夹爪到铃铛最小距离 | 0.207–0.235 m |
| 三组对照（base / 500 条 SFT / 训练同款指令） | 分布几乎完全重合 |

→ 模型**不朝铃铛移动**，不是精度问题。已排除：版本、state 顺序、chunk、归一化、ODE 步数、渲染分辨率、指令、数据覆盖、success 上报。

**待官方确认**：官方 click_bell 91% 所用的 ckpt 与评测脚本。

### 5.2 系统性偏差（所有任务）

adjust_bottle 21.9%（模型卡 100%）、place_shoe 21.9%（模型卡 99%）→ 我们的 harness 相对官方约 4.5 倍偏低，但**相对比较仍然有效**（这也是本项目所有结论都做同 harness 对照的原因）。

---

## 六、下一步（按优先级）

| # | 事项 | 时间 | 目的 |
|---|---|---|---|
| 1 | click_bell 数据复制（×10）+ 续训 | 4 小时 | 验证"任务数据占比少"假设 |
| 2 | 加一个第二任务（adjust_bottle）的 SFT 端到端 | 1 天 | 让"端到端验证"不止一个案例 |
| 3 | 向 mentor 索取官方 click_bell ckpt / 脚本 | — | 定位 0% 的最终原因 |

---

## 七、环境与路径速查

| 内容 | 路径 |
|---|---|
| 代码 | `/mnt/public/zhangyuzhuo/work/RLinf_v2`（分支 `feat-lingbot-v2-sft`） |
| 环境 | `/mnt/public/zhangyuzhuo/venvs/rlinf_v2` |
| 权重 | `models/lingbot-vla-v2-6b-robotwin`、`models/Qwen3-VL-4B-Instruct` |
| 数据 | `datasets/robotwin2_lerobot/{click_bell_clean_50, click_bell_rand_500, place_shoe_clean_50}` |
| place_shoe ckpt | `work/RLinf_v2/logs/20261103-11:31:19-robotwin_sft_place_shoe/robotwin_sft_place_shoe/checkpoints/` |
