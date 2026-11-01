# Dream API evaluation

Run Dream through RLinf's embodied evaluation entry point with RoboTwin or the existing dual-arm YAM environment. The rollout worker sends three camera streams, measured joint states and the task instruction to the Dream SDK, then passes its joint targets to the native environment.

## Setup

Install the RoboTwin environment and the Dream SDK in the same Python environment:

```bash
bash requirements/install.sh embodied --env robotwin
source .venv/bin/activate
pip install -e '.[dream]'
```

Follow the [RLinf RoboTwin installation guide](../../docs/source-en/rst_source/evaluations/guides/robotwin.rst) to prepare its assets. Set the asset path in the command below and export `DREAM_API_KEY` in the rollout worker's environment. Set `DREAM_IK_ARM_FILE` to a Dream SDK arm specification that matches the selected RoboTwin robot's URDF, joint order, base frame and TCP.

## RoboTwin

Use the standard evaluation script:

```bash
export DREAM_IK_ARM_FILE=/path/to/aloha-arm.json
bash evaluations/run_eval.sh robotwin robotwin_dream_aloha_eval \
  dream_task=click_bell \
  env.eval.assets_path=/path/to/robotwin_assets
```

This runs ten episodes with seeds 100100001–100100010. RLinf shuffles their order deterministically and advances to a different seed after each episode.

| Task override | Episode step limit |
| --- | --- |
| `dream_task=click_bell` | 400 |
| `dream_task=move_playingcard_away` | 400 |
| `dream_task=open_microwave` | 1500 |

For Piper, use its matching arm specification and config:

```bash
export DREAM_IK_ARM_FILE=/path/to/piper-arm.json
bash evaluations/run_eval.sh robotwin robotwin_dream_piper_eval \
  dream_task=open_microwave \
  env.eval.assets_path=/path/to/robotwin_assets
```

The ALOHA and Piper policy frequencies are 20 Hz and 30 Hz. RoboTwin retains its native motion planning, physics and success checker. Inspect `metrics.log`, TensorBoard and videos under `runner.logger.log_path`.

## Real hardware

The real adapter uses RLinf's `DualYamJointEnv-v1`. It forwards the action accepted by the robot controller to the SDK, holds the robot after termination and closes the native device resources when evaluation exits.

Start with the config's built-in dummy devices:

```bash
bash evaluations/run_eval.sh realworld realworld_dual_yam_dream_eval
```

This exercises the YAM observation/action path and calls the Dream API. For hardware, follow the [YAM setup guide](../../docs/source-en/rst_source/examples/embodied/yam.rst), configure the existing device backend and camera connections, set `env.eval.override_cfg.is_dummy=false`, and supply the task instruction. Keep the controller's joint limits and maximum joint delta configured for that robot. The included adapter uses the SDK's built-in `yam` arm and a 20 Hz joint head.

## Historical results

These are historical results.

| Robot | Task | Success |
| --- | --- | --- |
| aloha-agilex | click_bell | 5/10 |
| aloha-agilex | move_playingcard_away | 3/10 |
| piper | click_bell | 6/10 |
| piper | open_microwave | 2/10 |

## Interface checks

```bash
PYTHONPATH=. pytest tests/unit_tests/test_dream.py
```

These checks cover configuration composition, camera encoding, SDK action transport, seed rotation and terminal handling with dummy YAM devices. They use a fake SDK at the service boundary. API and hardware failures propagate to the caller so that a retry is an explicit decision.
