# Dream API 评测

通过 RLinf 的具身评测入口调用 Dream，可使用 RoboTwin 仿真环境或现有的双臂 YAM 真机环境。Rollout worker 把三路图像、实测关节状态和任务指令传给 Dream SDK，再把关节目标交给原生环境执行。

## 安装

在同一个 Python 环境中安装 RoboTwin 和 Dream SDK：

```bash
bash requirements/install.sh embodied --env robotwin
source .venv/bin/activate
pip install -e '.[dream]'
```

按 [RLinf RoboTwin 安装文档](../../docs/source-zh/rst_source/evaluations/guides/robotwin.rst)准备资产，并在运行命令中指定资产目录。在 rollout worker 的环境中设置 `DREAM_API_KEY`。将 `DREAM_IK_ARM_FILE` 指向 Dream SDK 的机械臂描述文件，其 URDF、关节顺序、base 坐标系和 TCP 必须与所选 RoboTwin 机器人一致。

## RoboTwin

使用原生评测脚本：

```bash
export DREAM_IK_ARM_FILE=/path/to/aloha-arm.json
bash evaluations/run_eval.sh robotwin robotwin_dream_aloha_eval \
  dream_task=click_bell \
  env.eval.assets_path=/path/to/robotwin_assets
```

该配置运行十个 episode，seed 为 100100001–100100010。RLinf 按固定随机顺序排列 seed，每个 episode 结束后切换到下一个。
| 任务参数 | Episode 步长上限 |
| --- | --- |
| `dream_task=click_bell` | 400 |
| `dream_task=move_playingcard_away` | 400 |
| `dream_task=open_microwave` | 1500 |

Piper 使用对应的机械臂描述文件和配置：

```bash
export DREAM_IK_ARM_FILE=/path/to/piper-arm.json
bash evaluations/run_eval.sh robotwin robotwin_dream_piper_eval \
  dream_task=open_microwave \
  env.eval.assets_path=/path/to/robotwin_assets
```

ALOHA 和 Piper 的策略频率分别为 20 Hz、30 Hz。动作规划、物理仿真和成功判定均由 RoboTwin 执行。指标、TensorBoard 和视频保存在 `runner.logger.log_path` 下。

## 真机

真机适配使用 RLinf 的 `DualYamJointEnv-v1`。SDK 接收控制器实际接受的动作；episode 结束后机械臂保持当前位置，退出时关闭原生设备资源。

先用配置中的 dummy 设备检查调用链：

```bash
bash evaluations/run_eval.sh realworld realworld_dual_yam_dream_eval
```

该命令使用 dummy YAM 的观测与动作通道，并调用 Dream API。连接真机前，按 [YAM 安装文档](../../docs/source-zh/rst_source/examples/embodied/yam.rst)配置设备后端和相机连接，将 `env.eval.override_cfg.is_dummy` 设为 `false`，并填写任务指令。关节限位和最大单步关节变化应与设备匹配。当前适配使用 SDK 内置的 `yam` 机械臂和 20 Hz joint head；

## 历史结果

下表列出部分测试结果。

| 机械臂 | 任务 | 成功次数 |
| --- | --- | --- |
| aloha-agilex | click_bell | 5/10 |
| aloha-agilex | move_playingcard_away | 3/10 |
| piper | click_bell | 6/10 |
| piper | open_microwave | 2/10 |


## 接口检查

```bash
PYTHONPATH=. pytest tests/unit_tests/test_dream.py
```

检查覆盖配置组合、图像编码、SDK 动作传输、seed 轮转及 dummy YAM 的终止处理，仅在服务边界使用 fake SDK。API 和硬件异常直接返回调用方，由调用方决定是否重试。
