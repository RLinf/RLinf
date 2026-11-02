# Dream API evaluation

Evaluate Dream with RLinf's native RoboTwin runner. The adapter sends three RGB views, measured joint states and task instructions to the Dream SDK and executes the returned joint targets through RoboTwin.

## Setup

```bash
bash requirements/install.sh embodied --env robotwin
source .venv/bin/activate
pip install -e '.[dream]'
```

Prepare the assets with the [RoboTwin installation guide](../../docs/source-en/rst_source/evaluations/guides/robotwin.rst). Export `DREAM_API_KEY` in the rollout worker's environment. `DREAM_IK_ARM_FILE` must match the selected robot's URDF, joint order, base frame and TCP.

## Evaluation

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

Each configuration runs ten episodes with seeds 100100001–100100010 and a 400-step episode limit. ALOHA runs at 20 Hz and Piper at 30 Hz. Metrics and videos are saved under `runner.logger.log_path`.

## Historical results

| Robot | Task | Success |
| --- | --- | --- |
| aloha-agilex | click_bell | 5/10 |
| aloha-agilex | move_playingcard_away | 3/10 |
| piper | click_bell | 6/10 |

## Interface checks

```bash
PYTHONPATH=. pytest -q tests/unit_tests/test_dream.py --confcutdir=tests/unit_tests
```
