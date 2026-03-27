=====================================================
使用 Android World 的 M3A 进行 Android 评估
=====================================================

本文档说明如何在 RLinf 中使用 m3a_worker.py（M3A Agent Worker）和
android_reward_worker.py（Android Reward Worker）进行 Android World 任务评估，
方便其他用户安装环境并复现整个流程。

本示例基于 Android World 官方仓库中的 M3A agent，并将其与 RLinf 的调度、
rollout 和 reward 体系集成在一起。
 
----------------
1. 总体概览
----------------

* ``m3a_worker.py``（``rlinf/workers/env/m3a_worker.py``）：

  使用 Android World 内置的 **M3A** agent，在真机或模拟器上执行任务，并通过
  ``Channel`` 与 rollout 侧的 LLM（例如基于 SGLang 的 Qwen3-VL）以及 reward worker
  通信。

* ``android_reward_worker.py``（``rlinf/workers/env/android_reward_worker.py``）：

  在独立进程中重连 Android 环境，根据 agent 的执行结果与任务定义计算 reward
  （例如任务是否成功），并将 reward 回传给 agent worker。

两者配合，可以只评估 M3A agent（不进行训练），整体流程与已有的
``agent_worker`` + ``reward_worker`` 相似，但 agent 侧逻辑更简洁，更便于复现与调试。

--------------------
2. 环境与依赖准备
--------------------

2.1 基础环境
=============

* **Python**：推荐 3.10+
* **操作系统**：Linux（推荐，更方便与 Android 设备/模拟器通信）
* **Android 设备**：至少一台已通过 ADB 连接的真机或模拟器
  （例如 ``emulator-5554`` 或 ``localhost:5557``）

2.2 安装 RLinf 依赖
====================

在 RLinf 根目录下，可以采用 requirements/install.sh 进行安装依赖和 qwen-vl-utils。
在 RLinf 中采用 uv 进行 Python 包管理，具体可参考 RLinf 文档：https://rlinf.readthedocs.io/en/latest/rst_source/examples/agentic/wideseek_r1/index.html

.. code-block:: bash

   cd path/to/RLinf
   bash requirements/install.sh agentic --venv reason
   source reason/bin/activate
   uv pip install qwen-vl-utils

2.3 安装 Android World 及其依赖
================================

``m3a_worker`` 和 ``android_reward_worker`` 都依赖 **android_world**。
建议将 android_world 仓库放在 ``RLinf`` 同级目录下，并安装其依赖：

.. code-block:: bash

   # 克隆 android_world（如尚未存在）
   git clone https://github.com/google-research/android_world.git /path/to/android_world

   # 安装 android_world 使用到的依赖
   sudo apt update && sudo apt install ffmpeg 
   uv pip install -r /path/to/android_world/requirements.txt
   uv pip install uiautomator2

2.4 路径配置
============

当前 **android_world** 的路径主要通过以下两部分进行配置：

* 在 ``qwen3vl-4b-eval.yaml`` 中使用 ``data.android_world_parent``，供
  ``AndroidWorldDataset`` 使用；
* 在启动脚本 ``eval.sh`` 中设置 ``PYTHONPATH``，应包含：

  - 项目根目录（例如 ``/path/to/your/root_project``）；
  - ``RLinf`` 目录（例如 ``/path/to/your/root_project/RLinf``）；
  - ``android_world`` 源码根目录（例如
    ``/path/to/your/root_project/android_world``），
    以便在所有 Ray worker 中 ``import android_world`` 以及
    ``from android_world.agents import m3a`` 能够正常工作。

推荐 **不要** 在库代码内部修改 ``sys.path``，而是在配置和启动脚本中完成路径设置：

1. 在 ``qwen3vl-4b-eval.yaml`` 中增加 ``data.android_world_parent`` 字段，例如：

   .. code-block:: yaml

      data:
        type: android
        task_family: android_world
        # ...
        android_world_parent: /absolute/path/to/android_world

2. 在 ``eval.sh`` 中，在运行 Python 之前设置好 ``PYTHONPATH``，例如：

   .. code-block:: bash

      PROJECT_ROOT="/path/to/your/root_project"
      RLINF_ROOT="$PROJECT_ROOT/RLinf"
      ANDROID_WORLD_ROOT="$PROJECT_ROOT/android_world"

      export PYTHONPATH="$PROJECT_ROOT:$RLINF_ROOT:$ANDROID_WORLD_ROOT:${PYTHONPATH:-}"

2.5 安装 ADB 与连接设备
========================

* 安装 Android SDK Platform Tools：

  .. code-block:: bash

     sudo apt update
     sudo apt install android-tools-adb android-tools-fastboot

* 连接设备或启动模拟器后，在服务器上确认 ADB 连通性：

  .. code-block:: bash

     adb devices

* 在配置文件中填写正确的 ``device_id``（例如 ``localhost:5557``）和
  ``adb_path``（例如 ``adb``）。

-------------------------------
3. 对 Android World 的必要修改
-------------------------------

为了在服务器环境中稳定复现 Android World，需要对其进行如下修改。

3.1 基于 uiautomator2 的 UI 层级获取
=====================================

在 ``android_world/android_world/env/adb_utils.py`` 中修改
``uiautomator_dump`` 函数，增加基于 uiautomator2 的 dump 逻辑，并在失败时
回退到 ADB：

.. code-block:: python

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
         logging.warning(
             'Managed uiautomator2 dump failed: %s, stopping agent and '
             'falling back to ADB.', e,
         )
         _stop_uiautomator2_agent(env)
         dump_args = 'shell uiautomator dump /sdcard/window_dump.xml'
         issue_generic_request(dump_args, env, timeout_sec=timeout_sec)

         read_args = 'shell cat /sdcard/window_dump.xml'
         response = issue_generic_request(read_args, env, timeout_sec=timeout_sec)

         return response.generic.output.decode('utf-8')

3.2 Controller 与 device_id 相关修改
====================================

* 在 ``android_world/android_world/env/android_world_controller.py`` 中，将
  ``a11y_method`` 设置为 ``A11yMethod.UIAUTOMATOR``。
* 将所有 ``representation_utils.forest_to_ui_elements`` 调用替换为
  ``env.controller.get_ui_elements()``。
* 为 ``env_launcher._get_env`` 和 ``load_and_setup_env`` 增加 ``device_id`` 参数，
  并在 ``android_world_controller.AndroidWorldController`` 中：

  - 新增 ``self.device_id`` 字段；
  - 在 ``__init__`` 中保存传入的 ``device_id``。

* 在 ``android_world_controller.py`` 中，在创建底层 ``AndroidEnv`` 实例后，
  将 ``device_id`` 挂载到 env 上，便于后续访问：

  .. code-block:: python

     android_env_instance = loader.load(config)

     # 将 device_id 直接挂在底层 AndroidEnv 实例上，方便后续在任意位置通过
     # env.device_id 访问。
     if device_id:
       try:
         setattr(android_env_instance, 'device_id', device_id)
       except Exception:
         logging.warning('Failed to attach device_id to AndroidEnv instance.')

--------------------------------
4. 运行评估的整体流程与命令
--------------------------------

4.1 配置简要说明
=================

1. **Cluster 配置**：

   在集群配置中定义 ``agent_worker`` 与 ``reward_worker`` 的 placement，
   并配置 ``android_world`` 节点组和 ADB 硬件信息。

2. **Rollout 配置**：

   提供一个 LLM 推理服务（例如基于 SGLang 的 Qwen3-VL 模型）供 M3A 调用。

3. **Data 配置**：

   使用 ``AndroidWorldDataset``，配置 ``data.type: android``、
   ``task_family: android_world`` 等字段，并设置 ``data.android_world_parent``。

4. **Reward 配置**：

   设置 ``reward.reward_type: android``，以及 Android World 相关参数
   （如 ``device_id``、``grpc_port``、``reward_scale`` 等）。

可以参考 ``rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml`` 中的示例配置。

4.2 运行步骤
=============

目前仅支持在本地机器启动模拟器，然后通过反向 SSH 端口转发到服务器进行评估。

步骤 1：本地安装并启动模拟器
-----------------------------

在本地机器上安装 Android Studio，创建一个模拟器：

* 硬件：Pixel 6
* 系统镜像：Tiramisu，API level 33
* AVD 名称：``AndroidWorldAvd``

启动模拟器并开启 gRPC 端口：

.. code-block:: bash

   EMULATOR_NAME=AndroidWorldAvd  # 上一步创建的名称
   ~/Library/Android/sdk/emulator/emulator -avd $EMULATOR_NAME -no-snapshot -grpc 8554

步骤 2：在本地进行反向 SSH 端口转发
------------------------------------

需要将模拟器使用的 ADB 控制端口和 Android World 使用的 gRPC 端口反向代理到服务器。
以 ``emulator-5554`` 和 ``grpc_port 8554`` 为例：

.. code-block:: bash

   # 确认 ADB 端口
   adb devices  # 应该看到 emulator-5554 设备

   # 在本地机器执行反向端口转发
   ssh -fNR 5555:localhost:5555 <user>@<server-host-or-ip>
   ssh -fNR 8554:localhost:8554 <user>@<server-host-or-ip>

步骤 3：在服务器上连接反向代理的端口
------------------------------------

在服务器上使用 ADB 连接映射过来的端口（继续使用上面的例子）：

.. code-block:: bash

   adb connect localhost:5555

步骤 4：评估前的必备准备
--------------------------

1. Android World 首次运行需要初始化环境，安装必要应用，可使用：

   .. code-block:: bash

      cd path/to/android_world
      python run.py \
        --suite_family=android_world \
        --agent_name=t3a_gpt4 \
        --perform_emulator_setup \
        --tasks=ContactsAddContact

2. 在模拟器中手动启动一次 Clipper 应用，确保剪贴板相关任务不会因为权限问题直接失败。

步骤 5：修改配置并运行评估
--------------------------

在 ``rlinf/example/mobile-agent/config/qwen3vl-4b-eval.yaml`` 中，根据你的服务器/
设备环境修改：

* ``cluster`` 中的硬件与 placement；
* ``data.android_world_parent``、``data.task_family``、``data.task_name`` 等；
* ``reward.device_id``、``reward.grpc_port``、``reward.adb_path`` 等。

然后运行：

.. code-block:: bash

   cd rlinf/example/mobile-agent/
   chmod +x eval.sh
   ./eval.sh

执行成功后，评估结果会写入 ``results/eval_results.json``，其中包含每个任务的
reward 及整体统计信息。

----------------
5. 测试与验证
----------------

为遵守 RLinf 的 Prime Directive，本次 Android World 集成新增了对应的单元测试：

* ``RLinf/tests/test_android_world_integration.py``

测试覆盖了两部分核心逻辑。

5.1 AndroidReward 行为测试
===========================

* ``test_android_reward_returns_zero_when_not_done``：

  当 ``result.done=False`` 时，验证 ``AndroidReward.get_reward_new`` 必须返回 ``0.0``。

* ``test_android_reward_scales_score_when_done``：

  使用一个返回固定分数的假任务，检查 reward 是否正确按 ``reward_scale`` 进行缩放。

* ``test_android_reward_swallows_task_exception``：

  使用一个在 ``is_successful`` 中抛出异常的假任务，验证不会向外抛错，而是安全返回
  ``0.0`` 并记录日志。

5.2 AndroidWorldDataset 与配置集成测试
======================================

* ``test_android_world_dataset_uses_android_world_parent_from_config``：

  通过向 ``sys.modules`` 注入一个假的 ``android_world.registry.TaskRegistry``，
  在不依赖真实 android_world 安装的前提下验证：

  * ``AndroidWorldDataset`` 会读取 ``data.android_world_parent``；
  * ``_load_data()`` 能够正确调用 ``TaskRegistry.get_registry(...)`` 并构建至少一个任务实例。

5.3 运行测试
============

在 RLinf 根目录下，可以通过以下命令运行这些测试：

.. code-block:: bash

   cd /path/to/your/root_project/RLinf
   PYTHONPATH=. pytest tests/test_android_world_integration.py


