# Evaluating Android World with M3A from Android World

This document describes how to use `m3_worker.py` (M3A Agent Worker) and `android_reward_worker.py` (Android Reward Worker) to evaluate Android World tasks, so that other users can set up the environment and reproduce the experiments.

---

## 1. Overview

- **`m3a_worker.py`** (`rlinf/workers/env/m3a_worker.py`): Uses the **M3A** agent built into [Android World](https://github.com/google-research/android_world) to execute tasks on a real device/emulator, and communicates with the rollout LLM and the reward worker via Channels.
- **`android_reward_worker.py`** (`rlinf/workers/env/android_reward_worker.py`): Reconnects to the env in a separate process, computes rewards (e.g., task success) based on the agent's behavior and task spec, and sends the reward back to the agent worker.

Together they allow you to **evaluate the M3A agent only** (no training), in a way similar to the existing `agent_worker` + `reward_worker` flow, but with a much simpler agent-side logic that is easier to reproduce and debug.

---

## 2. Environment Requirements

### 2.1 Basic environment
- **Python**: 3.10+ recommended
- **OS**: Linux (recommended, easier for talking to Android devices/emulators)
- **Android device**: At least one ADB-connected device or emulator (e.g. `emulator-5554` or `localhost:5557`)

### 2.2 Dependencies

(1) RLinf dependencies

```bash
pip install -r RLinf/docs/requirements.txt
pip install qwen-vl-utils
```

(2) Android World dependencies

Both `m3a_worker` and `android_reward_worker` depend on **android_world**, which must be placed as a sibling of `RLinf` or in some path that can be found by `sys.path` (the code currently uses absolute paths; see “Path configuration” below).

```bash
# Clone android_world (if you haven't already)
git clone https://github.com/google-research/android_world.git /path/to/android_world

# Install dependencies used by android_world
sudo apt update && sudo apt install ffmpeg
pip install -r /path/to/android_world/requirements.txt
pip install uiautomator2
```

### 2.3 Path configuration

Currently the **android_world** path is configured via:

- `data.android_world_parent` in `qwen3vl-4b-eval.yaml`, which is used by `AndroidWorldDataset`.
- The `PYTHONPATH` environment variable in the `eval.sh` script, which should include:
  - the project root (e.g. `/path/to/root_project`),
  - `RLinf` (e.g. `/path/to/root_project/RLinf`),
  - and the `android_world` source root (e.g. `/path/to/root_project/android_world`),
  so that imports like `import android_world` and `from android_world.agents import m3a` work in all Ray workers.

We recommend **not** modifying `sys.path` inside library code. Instead:

1. Add `data.android_world_parent` in `qwen3vl-4b-eval.yaml`:

   ```yaml
   data:
     type: android
     task_family: android_world
     # ...
     android_world_parent: /absolute/path/to/android_world
   ```

2. In `eval.sh`, set `PYTHONPATH` before launching the evaluation script, for example:

   ```bash
   PROJECT_ROOT="/path/to/your/root_project"
   RLINF_ROOT="$PROJECT_ROOT/RLinf"
   ANDROID_WORLD_ROOT="$PROJECT_ROOT/android_world"

   export PYTHONPATH="$PROJECT_ROOT:$RLINF_ROOT:$ANDROID_WORLD_ROOT:${PYTHONPATH:-}"
   ```

### 2.4 ADB and devices

- Install [Android SDK Platform Tools]:

```bash
sudo apt update
sudo apt install android-tools-adb android-tools-fastboot
```

- After connecting a device or starting an emulator, run:

```bash
adb devices
```

- In your config, fill in the correct `device_id` (e.g. `localhost:5557`) and `adb_path` (e.g. `adb`).

## 3. Changes Needed in Android World

To make Android World reproducible on a server, you need to patch Android World as follows:

1. Modify `uiautomator_dump` in `android_world/android_world/env/adb_utils.py`:

```python
def _stop_uiautomator2_agent(env) -> None:
  try:
    issue_generic_request(
        'shell am force-stop com.github.uiautomator',
        env, timeout_sec=5,
    )
  except Exception:
    pass
  try:
    issue_generic_request(
        'shell am force-stop com.github.uiautomator.test',
        env, timeout_sec=5,
    )
  except Exception:
    pass


def uiautomator_dump(env, timeout_sec: Optional[float] = 30) -> str:
  device_id = None
  if hasattr(env, 'controller') and env.controller is not None:
    device_id = getattr(env.controller, 'device_id', None) or ''
  if not device_id and hasattr(env, 'device_id'):
    device_id = env.device_id or ''

  if device_id:
    try:
      u2_device_id = device_id
      if device_id.startswith("localhost:"):
          port = device_id.split(":", 1)[1]
          port = int(port) - 1
          u2_device_id = f"emulator-{port}"
      device = u2.connect(u2_device_id)
      xml_content = device.dump_hierarchy()
      _stop_uiautomator2_agent(env)
      return xml_content
    except Exception as e:
      print(
          'Managed uiautomator2 dump failed: %s, stopping agent and '
          'falling back to ADB.', e,
      )
      _stop_uiautomator2_agent(env)
      dump_args = 'shell uiautomator dump /sdcard/window_dump.xml'
      issue_generic_request(dump_args, env, timeout_sec=timeout_sec)

      read_args = 'shell cat /sdcard/window_dump.xml'
      response = issue_generic_request(read_args, env, timeout_sec=timeout_sec)

      return response.generic.output.decode('utf-8')
```

2. In `android_world/android_world/env/android_world_controller.py`, set `a11y_method` to `A11yMethod.UIAUTOMATOR`.

3. Replace all occurrences of `representation_utils.forest_to_ui_elements` with `env.controller.get_ui_elements()`.

4. Add a `device_id` argument to `env_launcher._get_env` and `load_and_setup_env`, add a `self.device_id` field to `android_world_controller.AndroidWorldController`, and add a `device_id` parameter to its `__init__`.

5. In `android_world_controller.py`, add the following code:

```python
  android_env_instance = loader.load(config)

  # Attach device_id directly to the underlying AndroidEnv instance
  # so that env.device_id can be accessed anywhere later.
  if device_id:
    try:
      setattr(android_env_instance, 'device_id', device_id)
    except Exception:
      logging.warning('Failed to attach device_id to AndroidEnv instance.')
```

## 4. Reproduction Steps and Commands

### 4.1 Configuration

1. **Cluster**: Define placement for `agent_worker` and `reward_worker`, and configure the `android_world` node group and ADB hardware info.
2. **Rollout**: Provide an LLM inference service (e.g. SGLang) for M3A to call.
3. **Data**: Use `AndroidWorldDataset` with `data.type: android`, `task_family: android_world`, etc.
4. **Reward**: Configure `reward.reward_type: android`, etc.

You can refer to `rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml`.

### 4.2 How to run

Currently we only support running tests with an emulator started on your local machine.

1. **Install and start an emulator locally**

Install Android Studio, create an emulator (hardware: Pixel 6), choose system image Tiramisu, API level 33, and name the AVD `AndroidWorldAvd`:

```bash
EMULATOR_NAME=AndroidWorldAvd  # From previous step
~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
```

2. **Set up reverse SSH port forwarding from local to server**

You need to reverse-forward both the emulator's ADB control port and the gRPC port used by Android World to the server.  
Example for `emulator-5554` and `grpc_port 8554`:

```bash
# Check ADB port
adb devices  # shows emulator-5554 device

# Reverse forwarding from local machine
ssh -fNR 5555:localhost:5555 <user>@<server-host-or-ip>
ssh -fNR 8554:localhost:8554 <user>@<server-host-or-ip>
```

3. **On the server**

On the server, connect to the reverse-forwarded ADB port.  
Continuing the `emulator-5554` / gRPC 8554 example:

```bash
adb connect localhost:5555
```

4. **Required pre-steps**

(1) Android World needs an initial setup run to install required apps. Use `--perform_emulator_setup`:

```bash
cd path/to/android_world
python run.py \
  --suite_family=android_world \
  --agent_name=t3a_gpt4 \
  --perform_emulator_setup \
  --tasks=ContactsAddContact
```

(2) In the emulator, manually start the Clipper app once.

5. **Edit config and run eval**

In `rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml`, adjust the settings according to your server and device config, then run `eval.sh`:

```bash
cd rlinf/example/mobile-agent/
chmod +x eval.sh
./eval.sh
```

## 5. Tests and Verification

To comply with RLinf's Prime Directive, user-facing changes to the Android World
integration are covered by unit tests. The most relevant tests are:

- `RLinf/tests/test_android_world_integration.py`

This test module verifies:

1. **AndroidReward behavior**

   - `test_android_reward_returns_zero_when_not_done`  
     Ensures that `AndroidReward.get_reward_new` returns `0.0` when `result.done` is `False`.

   - `test_android_reward_scales_score_when_done`  
     Uses a fake task whose `is_successful` returns a score and checks that the
     reward is correctly scaled by `reward_scale`.

   - `test_android_reward_swallows_task_exception`  
     Uses a fake task whose `is_successful` raises an exception and checks that
     the method does **not** crash and instead returns `0.0`. This guards
     against reward computation killing the whole rollout.

2. **AndroidWorldDataset configuration and registry integration**

   - `test_android_world_dataset_uses_android_world_parent_from_config`  
     Injects a fake `android_world.registry.TaskRegistry` into `sys.modules` so
     that no real `android_world` installation is required. It then constructs
     a minimal `DictConfig` containing:

     ```yaml
     data:
       max_prompt_length: 128
       task_family: android_world
       n_instances_per_task: 1
       apply_chat_template: False
       android_world_parent: /does/not/matter/for/test
     ```

     and a minimal tokenizer stub. The test asserts that
     `len(AndroidWorldDataset(...)) == 1`, which confirms that:

     - `data.android_world_parent` from the config is honored, and  
     - `AndroidWorldDataset._load_data()` correctly integrates with
       `TaskRegistry.get_registry(...)`.

To run these tests in your environment:

```bash
cd /path/to/your/root_project
export PYTHONPATH="$PWD/RLinf:${PYTHONPATH:-}"
pytest RLinf/tests/test_android_world_integration.py
```