# 使用Android world中的m3a进行Android world评估

本文档说明如何使用 `m3a_worker.py`（M3A Agent Worker）与 `android_reward_worker.py`（Android Reward Worker）进行 Android World 任务评估，便于其他用户安装环境并复现运行。

---

## 1. 概述

- **`m3a_worker.py`**（`rlinf/workers/env/m3a_worker.py`）：基于 [Android World](https://github.com/google-research/android_world) 内置的 **M3A** agent，在真机/模拟器上执行任务，并通过 Channel 与 Rollout（LLM）和 Reward Worker 通信。
- **`android_reward_worker.py`**（`rlinf/workers/env/android_reward_worker.py`）：在独立进程中重连 env，根据 agent 的执行结果与任务定义计算 reward（如任务是否成功），并回传给 Agent Worker。

二者配合可用于 **仅评估 M3A agent**（不训练），与现有 `agent_worker` + `reward_worker` 的流程类似，但 agent 侧逻辑更简洁，便于复现与调试。

---

## 2. 环境要求

### 2.1 基础环境
- **Python**：建议 3.10+
- **操作系统**：Linux（推荐，用于与 Android 设备/模拟器通信）
- **Android 设备**：至少一台已通过 ADB 连接的设备或模拟器（如 `emulator-5554` 或 `localhost:5557`）

### 2.2 依赖安装

(1) RLinf 依赖安装

```bash
pip install -r RLinf/docs/requirements.txt
pip install qwen-vl-utils
```
(2) AndroidWorld 依赖安装

`m3a_worker` 与 `android_reward_worker` 均依赖 **android_world**，需放在与 `RLinf` 同级或可被 `sys.path` 找到的路径（代码中默认使用绝对路径，见下方「路径配置」）。

```bash
# 克隆 android_world（若尚未存在）
git clone https://github.com/google-research/android_world.git /path/to/android_world

# 安装 android_world 使用到的依赖
sudo apt update && sudo apt install ffmpeg
pip install -r /path/to/android_world/requirements.txt
pip install uiautomator2

```

### 2.3 路径配置
当前代码中 **android_world** 的路径为硬编码，需与你的实际路径一致：
- `rlinf/workers/env/m3a_worker.py` 中`android_world_parent = "path/to/your/android_world"`需要修改成你的路径
-`rlinf/workers/env/android_reward_worker.py` 和 `rlinf/algorithms/rewards/android/__init__.py`  同理

### 2.4 ADB 与设备
- 安装 [Android SDK Platform Tools]
```bash
sudo apt update
sudo apt install android-tools-adb android-tools-fastboot
```
- 连接设备或启动模拟器后执行：
```bash
adb devices
```
- 在配置中填写正确的 `device_id`（如 `localhost:5557`）和 `adb_path`（如 `adb`）。

## 3. Android world项目修改
为了适配服务器复现Androidworld项目，需要对Android world进行如下修改：
1. 修改`android_world/android_world/env/adb_utils.py`中的`uiautomator_dump`函数
```bash
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
2. 修改`android_world/android_world/env/android_world_controller.py`中`a11y_method` 为 ` A11yMethod.UIAUTOMATOR`
3. 将文件中所有`representation_utils.forest_to_ui_elements` 修改成`env.controller.get_ui_elements()`
4.为 `env_launcher._get_env` 和 `load_and_setup_env` 函数增加 `device_id` 参数，并在`android_world_controller.AndroidWorldController` 中新增 `self.device_id` 字段,在其`__init__`函数中新增`device_id`参数
4. 在`android_world_controller.py`中添加如下代码：
```bash
  android_env_instance = loader.load(config)

  # 将 device_id 直接挂在底层 AndroidEnv 实例上，方便后续任何地方通过 env.device_id 访问。
  if device_id:
    try:
      setattr(android_env_instance, 'device_id', device_id)
    except Exception:  
      logging.warning('Failed to attach device_id to AndroidEnv instance.')
```

## 4. 复现步骤与命令

### 4.1 配置说明
1. **Cluster 配置**：定义 `agent_worker`、`reward_worker` 的 placement，以及 `android_world` 节点组和 ADB 硬件信息。
2. **Rollout**：提供 LLM 推理服务（如 SGLang），供 M3A 调用。
3. **Data**：使用 `AndroidWorldDataset`，配置 `data.type: android`、`task_family: android_world` 等。
4. **Reward**：配置 `reward.reward_type: android` 等。
可以参考`rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml`

### 4.2 操作说明
目前只支持在本地机器上启动模拟器进行测试

1. 安装模拟器
本地安装安卓模拟器（android studio），创建一个模拟器，hardware 选择 Pixel 6, 在系统镜像中选择 Tiramisu、API 级别 33，并将 AVD 名称设为 AndroidWorldAvd。
```bash
    EMULATOR_NAME=AndroidWorldAvd # From previous step
    ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554
```
2. 在本地进行反向代理
需要将模拟器的adb控制端口以及启动所采用的grpc端口反代理到服务器，以下按`emulator-5554`,`grpc_port 8554`作为示例
```bash
#查看adb端口
adb devices #显示 emulator-5554 device 

#本地反代理
ssh -fNR 5555:localhost:5555 用户名@目标服务器IP/域名
ssh -fNR 8554:localhost:8554 用户名@目标服务器IP/域名
```

3. 服务器操作
在服务器中使用adb命令连接上一步反代理过来的端口,以下按`emulator-5554`,`grpc_port 8554`作为示例
```bash
adb connect localhost:5555
```

4. 必备前置操作
（1）Androidworld需要先进行一次初始化来安装所需要的应用文件，使用`--perform_emulator_setup` 参数来安装
```bash
cd path/to/android_world
python run.py \
  --suite_family=android_world \
  --agent_name=t3a_gpt4 \
  --perform_emulator_setup \ 
  --tasks=ContactsAddContact\
```
（2）在模拟器中，手动启动一次clipper软件

5. 修改配置
在`rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml`中按照自己的服务器配置修改相应设置，进而运行`eval.sh`便可
```bash
cd rlinf/example/mobile-agent/
chmod +x eval.sh
./ eval.sh
```