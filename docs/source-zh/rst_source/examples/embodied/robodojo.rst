RoboDojo 强化学习
=================

RoboDojo 是基于 Isaac Sim 5.1 的双臂 ARX-X5 操作基准。RLinf 将它接入为普通
Gym 环境，策略输出的连续动作可以直接交给 RoboDojo 执行。

这条路径目前只支持 **Linux x86_64 + NVIDIA CUDA GPU**。第一次运行建议只用
一个环境，先确认模拟器、相机和资产都正常，再增加并行环境数。

只需要记住三件事
----------------

1. 用 RLinf 安装器安装依赖，不要手工逐个安装 Isaac Sim、IsaacLab、cuRobo。
2. ``ROBODOJO_ASSETS_ROOT`` 要指向 **包含 ``Assets/`` 的目录**。
3. RoboDojo 每个并行环境都有自己的 Isaac Sim 进程；先在 24 GB GPU 上跑一个环境。

安装前检查
----------

.. code-block:: bash

   nvidia-smi
   uname -m
   df -h .
   python --version


一键安装依赖
------------

在 RLinf 仓库根目录执行：

.. code-block:: bash

   bash requirements/install.sh embodied --env robodojo
   source .venv/bin/activate

国内网络可以加 ``--use-mirror``。如果还要安装 OpenPI 策略：

.. code-block:: bash

   bash requirements/install.sh embodied --model openpi --env robodojo

安装器会自动完成以下工作：

* 安装 PyTorch 2.7、Isaac Sim 5.1、固定版本的 IsaacLab 和 cuRobo；
* 使用 ``requirements/embodied/envs/robodojo-override.txt`` 处理
  ``starlette``、``uvicorn``、``filelock`` 等依赖冲突；
* 把 ``OMNI_KIT_ACCEPT_EULA`` 和 ``ROBODOJO_ASSETS_ROOT`` 写入虚拟环境；
* 检查 ``robodojo_runtime``、PyTorch CUDA 和内置 RoboDojo 源码；
* 不启动训练，也不默认下载大体积场景资产。

.. note::

   安装器会设置 ``OMNI_KIT_ACCEPT_EULA=YES``。继续执行安装即表示接受
   NVIDIA Omniverse 的相关许可条款。

不要再执行普通的 ``pip install rlinf-robodojo-runtime``。RoboDojo 的
IsaacLab 元数据与 Isaac Sim 的 FastAPI 依赖存在已知冲突，必须使用安装器提供的
override 文件。

安装场景资产
------------

完整 ``Assets/`` 约 39 GB。可以手动放到任意持久目录：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   mkdir -p "$ROBODOJO_ASSETS_ROOT"

也可以让安装器顺便下载：

.. code-block:: bash

   export ROBODOJO_ASSETS_ROOT=/path/to/robodojo
   ROBODOJO_DOWNLOAD_ASSETS=1 \
     bash requirements/install.sh embodied --env robodojo

正确目录结构如下：

.. code-block:: text

   /path/to/robodojo/
   └── Assets/
       ├── Robots/
       ├── Object/
       ├── Material/
       └── Eval_Layout/

``ROBODOJO_ASSETS_ROOT`` 不能直接写成 ``/path/to/robodojo/Assets``。
如果使用 Hugging Face 镜像，可以在安装前设置 ``HF_ENDPOINT``。

安装后自检
----------

以下检查不会启动仿真器：

.. code-block:: bash

   python - <<'PY'
   import torch
   from robodojo_runtime import assets_root, source_root
   from robodojo_runtime.bridge import VectorEnv

   print("torch:", torch.__version__)
   print("cuda:", torch.cuda.is_available())
   print("source:", source_root())
   print("assets:", assets_root())
   PY

第一次真实启动
--------------

先在空 GPU 上做单环境、单动作冒烟测试：

.. code-block:: bash

   python - <<'PY'
   import numpy as np
   from robodojo_runtime.bridge import VectorEnv

   env = VectorEnv(
       {
           "task_name": "put_bottles_into_dustbin",
           "env_cfg_type": "arx_x5",
           "headless": True,
           "cuda_device": 0,
           "max_episode_steps": 700,
       },
       n_envs=1,
       env_seeds=[0],
   )
   try:
       env.reset()
       obs = env.get_obs()[0]
       print(obs["full_image"].shape, obs["state"].shape)
       env.step(np.zeros((1, 1, 14), dtype=np.float32))
   finally:
       env.close()
   PY

第一次测试不要同时开启策略服务、多个环境和视频录制。确认这个命令能正常退出后，
再使用 ``examples/embodiment/config/env/robodojo_put_bottles_into_dustbin.yaml``。

常见问题
--------

``No module named 'robodojo_runtime'``
   确认已激活安装器创建的虚拟环境；如果使用了 ``--venv``，进入对应的环境。

``No solution found when resolving dependencies``
   不要绕过安装器手工安装 runtime。删除错误的安装结果后，使用本文的一键命令重新安装。

``Could not detect CUDA`` 或 ``torch.cuda.is_available() == False``
   先运行 ``nvidia-smi``。RoboDojo 当前不支持纯 CPU、macOS 或 AMD ROCm。

``Assets`` 相关报错或找不到 ``cam_head``
   检查 ``ROBODOJO_ASSETS_ROOT`` 是否指向 ``Assets/`` 的父目录，并确认四个必需子目录都存在。

启动后显存不足
   先设置一个环境，并关闭其他训练或推理进程。RoboDojo 的多环境模式会为每个环境启动
   Isaac Sim 进程，不能直接照搬 RoboTwin 的大并行数。

环境退出后仍有 GPU 显存
   确认通过 ``try/finally`` 调用了 ``env.close()``；RoboDojo adapter 不支持挂起和恢复，
   只支持关闭后重新创建。
