# Dream API 评测

使用 RLinf 原生 RoboTwin 入口评测 Dream。适配层将三路 RGB 图像、实测关节状态和任务指令传给 Dream SDK，并通过 RoboTwin 执行返回的关节目标。

## 安装

```bash
bash requirements/install.sh embodied --env robotwin
source .venv/bin/activate
pip install -e '.[dream]'
```

按 [RoboTwin 安装文档](../../docs/source-zh/rst_source/evaluations/guides/robotwin.rst)准备资产。在 rollout worker 的环境中设置 `DREAM_API_KEY`。`DREAM_IK_ARM_FILE` 必须与所选机器人的 URDF、关节顺序、base 坐标系和 TCP 匹配。

## 评测

```bash
export DREAM_IK_ARM_FILE=/path/to/aloha-arm.json
bash evaluations/run_eval.sh robotwin robotwin_dream_aloha_click_bell_eval \
  env.eval.assets_path=/path/to/robotwin_assets
bash evaluations/run_eval.sh robotwin robotwin_dream_aloha_click_bell_eval \
  env.eval.task_config.task_name=move_playingcard_away \
  env.eval.assets_path=/path/to/robotwin_assets

export DREAM_IK_ARM_FILE=/path/to/piper-arm.json
bash evaluations/run_eval.sh robotwin robotwin_dream_piper_click_bell_eval \
  env.eval.assets_path=/path/to/robotwin_assets
```

每份配置运行十个 episode，seed 为 100100001–100100010，每回合上限为 400 步。ALOHA 频率为 20 Hz，Piper 为 30 Hz。指标和视频保存在 `runner.logger.log_path` 下。

## 历史结果

| 机械臂 | 任务 | 成功次数 |
| --- | --- | --- |
| aloha-agilex | click_bell | 5/10 |
| aloha-agilex | move_playingcard_away | 3/10 |
| piper | click_bell | 6/10 |

## 接口检查

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_dream.py --confcutdir=tests/unit_tests
```
