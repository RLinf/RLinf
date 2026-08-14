摩尔线程 MUSA
=============

在摩尔线程（MTT）GPU 上运行 RLinf 具身训练。本页只介绍与 NVIDIA 流程不同的部分：
环境如何构建、渲染与物理仿真如何配置，以及目前尚不支持的能力。任务说明、算法、
模型下载与指标都与硬件平台无关，参见
:doc:`LIBERO 基准强化学习 <../examples/embodied/libero>` 和
:doc:`ManiSkill 强化学习 <../examples/embodied/maniskill>`\ 。

RLinf 会自动识别 MTT GPU：调度器将其报告为 ``MUSA_GPU`` 加速器，并为每个 worker
设置 ``MUSA_VISIBLE_DEVICES``\ ，因此 placement 与多卡配置无需改动即可使用。

安装
----

MUSA 与其他平台有一个重要区别：**RLinf 不会在 MUSA 上安装 PyTorch。**
``torch-musa`` 及其配套的 MUSA 版 torch 并未发布到 PyPI 或 PyTorch 的 wheel 索引，
它们随摩尔线程 training-suite 镜像一起提供。因此
``install.sh --platform musa`` 会：

- 复用镜像自带的解释器，而不是下载 uv 托管的 Python；
- 使用 ``--system-site-packages`` 创建虚拟环境，使镜像中的
  ``torch``/``torch_musa`` 仍然可以导入；
- 将整个 torch 系列（以及仅适用于 CUDA 的 ``nvidia-*`` 运行时 wheel）
  排除在依赖解析之外；
- 跳过 flash-attention 与 apex（镜像已自带二者的 MUSA 版本），
  以及 vLLM/SGLang 的 CUDA kernel。

因此，安装必须在摩尔线程容器**内部**执行。

选项 1：Docker 镜像
~~~~~~~~~~~~~~~~~~~

使用 RLinf 的 Dockerfile 并指定 ``PLATFORM=musa`` 构建，基础镜像为
``registry.mthreads.com/mcctest/ai/training-suite:${MUSA_VER}``\ ；
构建目标本身无需针对 MUSA 做任何特殊处理：

.. code-block:: bash

   DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile \
      --build-arg PLATFORM=musa \
      --build-arg MUSA_VER=v2.1.5-musa4.3.7 \
      --build-arg BUILD_TARGET=embodied-maniskill_libero \
      -t rlinf:embodied-maniskill_libero .

必须启用 BuildKit：传统构建器会解析 Dockerfile 中的每个 ``FROM``\ ，
包括 Docker Hub 上的 CUDA 与 ROCm 基础镜像，而 MUSA 主机通常访问不到它们。

.. important::

   mthreads 容器运行时会在运行时注入 MUSA 驱动库，这与 NVIDIA 运行时注入
   ``libcuda`` 的方式完全相同。这些库在镜像构建期间并不存在，因此构建时
   ``import torch`` 会以 ``ImportError: libmusa.so.1`` 失败。MUSA 镜像构建过程中
   任何步骤都不能导入 torch。ManiSkill 资产是通过 ``mani_skill``\ （进而通过 torch）
   下载的，因此在构建时会自动跳过。请改为在运行中的容器内下载一次：

   .. code-block:: bash

      source switch_env openpi && download_assets --assets maniskill

使用 ``mthreads`` 容器运行时启动，只有它才能把 MTT GPU 暴露给容器：

.. code-block:: bash

   docker run -it --rm \
      --runtime=mthreads \
      --ipc=host \
      --shm-size=100g \
      -e MTHREADS_VISIBLE_DEVICES=all \
      -v .:/workspace/RLinf \
      rlinf:embodied-maniskill_libero bash

训练前先确认驱动可见：

.. code-block:: bash

   mthreads-gmi
   python -c "import torch, torch_musa; print(torch.musa.device_count())"

选项 2：在已有 MUSA 容器中安装
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

按上面的方式从 training-suite 镜像启动容器，然后执行：

.. code-block:: bash

   bash requirements/install.sh --platform musa embodied --model openpi --env maniskill_libero
   source .venv/bin/activate

中国大陆用户可加 ``--use-mirror`` 加速下载。

如果 ``torch_musa`` 无法导入，``install.sh`` 会立即报错退出；如果它报告没有可用
设备，则只会给出警告——在运行时这通常意味着容器启动时缺少 ``--runtime=mthreads``\ 。

渲染：使用 OSMesa 而非 EGL
--------------------------

摩尔线程驱动栈不提供 EGL 设备，因此仿真器必须通过 OSMesa 进行软件渲染。
``examples/embodiment/run_embodiment.sh`` 与 e2e 运行脚本都会遵循这两个变量，
e2e 脚本还支持以第二个参数指定渲染后端；若直接调用 Python，请显式设置：

.. code-block:: bash

   export MUJOCO_GL=osmesa
   export PYOPENGL_PLATFORM=osmesa

软件渲染显著慢于 EGL，相同 batch size 下 rollout 耗时会明显高于 NVIDIA 主机。

.. note::

   ``training-suite:v2.1.5-musa4.3.7`` 自带 ``libGL`` 但没有 ``libOSMesa``\ 。
   ``requirements/sys_deps.sh`` 会安装它（``libosmesa6``\ ），与 Ascend、ROCm 上的
   做法完全一致，因此 ``PLATFORM=musa`` 构建出的镜像自带可用的软件渲染器。
   如果你手工组装镜像，请在运行仿真器前用
   ``ldconfig -p | grep -i osmesa`` 检查。

ManiSkill：CPU 物理后端
-----------------------

ManiSkill 的 GPU 物理后端（SAPIEN PhysX）依赖 CUDA，而 MUSA 不提供 CUDA。
因此 ManiSkill 配置必须选择 CPU 后端：

.. code-block:: yaml

   env:
     train:
       init_params:
         sim_backend: "cpu"
     eval:
       init_params:
         sim_backend: "cpu"

CPU 后端无法在单个进程内做环境向量化，因此每个 env rank 只能驱动一个环境，
需要把 ``total_num_envs`` 设为 env 的 world size。LIBERO 没有这一限制，
它本身就在 CPU 上做向量化，无需改动。

.. warning::

   ManiSkill 还需要针对 MUSA 适配的 SAPIEN，而该版本并未发布到 PyPI。官方
   ``sapien`` 无法在 ``cpu`` 设备上创建渲染系统，会报
   ``Failed to find a supported physical device "cpu"``\ ；而厂商版 SAPIEN 又需要
   打过补丁的 ManiSkill（公开的 ``v3.0.0b22`` tag 会导入
   ``sapien.sensor.StereoDepthSensor``\ ，厂商版 SAPIEN 并不提供该符号）。两者都
   随摩尔线程提供的 RLinf 镜像（``registry.mthreads.com/lgpublic/rlinf:*``\ ）一起
   发布，因此目前 MUSA 上的 ManiSkill 只能在这些镜像中运行。无论哪种方式，
   ``install.sh --platform musa`` 都能得到可用的 LIBERO 环境。

启动训练
--------

.. code-block:: bash

   MUJOCO_GL=osmesa \
   PYOPENGL_PLATFORM=osmesa \
   ROBOT_PLATFORM=LIBERO \
   bash examples/embodiment/run_embodiment.sh libero_10_ppo_openpi_pi05

``*_musa`` 端到端配置是两个基准的最小可运行配置，适合作为起点。
与 ROCm、Ascend 的 CI 作业一样，渲染后端通过第二个参数传入：

.. code-block:: bash

   export REPO_PATH=$(pwd)
   ROBOT_PLATFORM=LIBERO bash tests/e2e_tests/embodied/run.sh libero_10_ppo_openpi_pi05_musa osmesa
   ROBOT_PLATFORM=BRIDGE bash tests/e2e_tests/embodied/run.sh maniskill_async_ppo_openpi_pi05_musa osmesa

暂不支持
--------

- **transformers 的 ``flash_attention_2`` 路径** —— 镜像自带可用的 MUSA
  flash-attn（``flash_attn`` 2.6.3，kernel 可在 ``musa`` 上运行），但它没有在顶层
  重新导出 ``flash_attn_func``\ ，因此
  ``transformers.utils.is_flash_attn_2_available()`` 返回 ``False``\ 。默认使用
  ``attn_implementation: flash_attention_2`` 的模型需改为 ``sdpa``\ ；
  如需直接调用 kernel，请通过 ``flash_attn.flash_attn_interface``\ 。
- **vLLM / SGLang rollout** —— RLinf 的 agentic 目标依赖这两个引擎的 CUDA 构建。
  具身训练使用 ``huggingface`` rollout 后端，不受影响。
- **ManiSkill GPU 向量化仿真** —— 见上文。

保持一致的部分
--------------

- 配置、模型与 ``model_path`` 流程与 NVIDIA 上完全相同。
- PPO/GRPO 算法设置与 placement 概念完全相同。
- 训练、rollout 与环境指标完全相同。
