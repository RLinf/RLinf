常见问题
========

下面按从新手首次运行到多机集群与调试的顺序整理 RLinf 常见问题。

按现象快速定位
--------------

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 现象 / 问题
     - 对应章节
   * - 安装第二个模型或环境后原环境失效
     - :ref:`多套环境覆盖 .venv <faq-venv>`
   * - 在单张 GPU 上运行 8 卡配置
     - :ref:`单卡运行 8 卡配置 <faq-single-gpu>`
   * - ``model_path``、``ckpt_path`` 与 ``resume_dir`` 的区别
     - :ref:`model_path 与 ckpt_path 的区别 <faq-ckpt>`
   * - ``total_num_envs`` 或 ``num_action_chunks`` 整除报错
     - :ref:`环境数与动作块整除校验 <faq-divisibility>`
   * - 下载慢、超时或无法连接 Ray GCS
     - :ref:`下载与网络问题 <faq-network>`
   * - CUDA 显存不足、NCCL 报错或集合通信超时
     - :ref:`CUDA 与显存问题 <faq-cuda>`
   * - ``MUJOCO_EGL_DEVICE_ID`` 或 Vulkan 驱动报错
     - :ref:`渲染问题 <faq-rendering>`
   * - 对 Ray worker 做断点调试
     - :ref:`断点调试 <faq-debugging>`

.. _faq-setup:

环境安装与配置
--------------

.. _faq-venv:

安装第二个模型或环境后原环境失效
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 再次执行 ``bash requirements/install.sh ...`` 安装另一套模型或仿真器后，之前安装的依赖被覆盖或导入报错。

**原因：** ``install.sh`` 默认写入 ``.venv``，重跑会用新依赖覆盖原有虚拟环境。

**修复：** 为不同模型与环境组合指定独立的 ``--venv <dir>``，运行前激活对应目录：

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla --env maniskill_libero --venv .venv-maniskill
   bash requirements/install.sh embodied --model openpi --env libero --venv .venv-libero

.. _faq-single-gpu:

在单张 GPU 上运行 8 卡配置
~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 基准配置默认面向 8 卡（例如 ``actor,rollout,env: 0-7``），在单卡机器上直接启动会报 placement 错误或显存不足。

**修复：** 将所有组件固定到 0 号 GPU，同步调小并行环境数和 batch size，并开启 offload：

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh <config_name> \
     cluster.component_placement.actor,rollout,env=0 \
     env.train.total_num_envs=8 \
     actor.micro_batch_size=1 \
     actor.global_batch_size=8 \
     actor.enable_offload=True \
     rollout.enable_offload=True

保持 ``actor.global_batch_size`` 能被 ``actor.micro_batch_size`` 整除，且 ``env.train.total_num_envs`` 能被 ``rollout.pipeline_stage_num`` 整除。

.. _faq-ckpt:

model_path、ckpt_path 与 resume_dir 的区别
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 不确定 Hugging Face 模型、训练保存的 ``.pt`` 权重和断点续训目录分别应填到哪个字段。

**修复：** 三个字段各司其职：

- ``actor.model.model_path`` / ``rollout.model.model_path``：基座模型目录或 Hugging Face ``owner/repo``\ （包含模型配置、tokenizer、归一化统计和基础权重）。训练时 rollout 自动继承 ``actor.model.model_path``；独立评测时设置 ``rollout.model.model_path``。
- ``runner.ckpt_path``：训练产出的 ``.pt`` 权重文件（例如 RL 或 SFT 保存的 ``full_weights.pt``），在 ``model_path`` 构建的模型结构之上加载，用于评测或热启动。
- ``runner.resume_dir``：完整 checkpoint 目录（例如 ``logs/<run>/checkpoints/global_step_50``），恢复模型权重、优化器状态和步数计数，用于继续中断的训练。

.. _faq-divisibility:

total_num_envs 或 num_action_chunks 整除报错
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 启动配置校验报错，提示 ``total_num_envs``、``env_world_size`` 或 ``max_steps_per_rollout_epoch`` 无法整除。

**修复：** 同时满足两条整除规则：

1. ``total_num_envs`` 必须能被 ``env worker 数量 × rollout.pipeline_stage_num`` 整除。例如 ``10`` 在 4 卡和 8 卡上会直接报错，推荐设为 ``8``、``16`` 或 ``128``\ （同时兼容 1、2、4、8 卡）。
2. ``max_steps_per_rollout_epoch`` 必须能被 ``rollout.model.num_action_chunks`` 整除（例如 OpenPI LIBERO 取 ``5`` 的倍数，RoboLab 取 ``15`` 的倍数，SIMPLE 取 ``24`` 的倍数）。

.. _faq-network:

下载与网络问题
--------------

Hugging Face / GitHub 下载慢或超时
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 运行 ``hf download`` 或 ``requirements/install.sh`` 拉取模型、数据集或依赖包时卡住或超时。

**修复：** 在中国大陆或受限网络环境下启用镜像源：

.. code-block:: bash

   export HF_ENDPOINT=https://hf-mirror.com
   bash requirements/install.sh embodied --model openvla --env maniskill_libero --use-mirror

也可以从 `Docker Hub <https://hub.docker.com/r/rlinf/rlinf>`__ 拉取预构建镜像（详见 :doc:`../start/installation`）。

无法连接 GCS（ip:port）
~~~~~~~~~~~~~~~~~~~~~~~

**现象：** Worker 节点无法连接到给定地址上的 Ray head（GCS）。

**原因：** 在 0 号节点上用以下命令获取 head 节点 IP：

.. code-block:: bash

   hostname -I | awk '{print $1}'

若该命令选择了其他节点不可达的网卡（例如网卡顺序不一致，可达的是 ``eth0``，却选中了别的接口），Worker 就会连接失败。

**修复：**

- 确认所选 IP 能被其他节点访问（例如使用 ``ping`` 测试）。
- 显式指定可达网卡的 IP 启动 Ray head，并将该 IP 告知各 Worker。

.. _faq-cuda:

CUDA 与显存问题
---------------

.. _faq-oom:

显存不足（OOM）
~~~~~~~~~~~~~~~

**现象：** rollout 或 actor 更新阶段报错 ``torch.OutOfMemoryError: CUDA out of memory``。

**修复：** 在配置 YAML 中降低单卡显存占用并开启 offload：

.. code-block:: yaml

   env:
     train:
       total_num_envs: 8
       enable_offload: True
   rollout:
     enable_offload: True
   actor:
     micro_batch_size: 1
     enable_offload: True

保持 ``actor.global_batch_size`` 能被 ``actor.micro_batch_size`` 乘以 actor worker 数量整除，且 ``env.train.total_num_envs`` 能被 env worker 数量乘以 ``rollout.pipeline_stage_num`` 整除。

任务迁移时出现 NCCL “cuda invalid argument”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** P2P 任务传输失败，报错 ``NCCL cuda invalid argument``。

**修复：** 若此机器上之前运行过任务，先停止 Ray 再重新启动（共享机器上先用 ``ray status`` 确认无他人任务）：

.. code-block:: bash

   ray stop

SGLang 加载参数时出现 NCCL “cuda invalid argument”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** SGLang 在加载权重时报 ``NCCL cuda invalid argument``。

**原因：** Placement 不匹配。例如配置使用 *共置（collocated）*，但训练（trainer）与生成（generation）实际跑在不同 GPU 上。

**修复：** 检查 placement 配置，确保训练组与生成组按照 ``cluster.component_placement`` 指定的 GPU 放置。

FSDP 集合通信被后端看门狗超时杀掉
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 训练步仍在正常推进，FSDP actor 却被集合通信看门狗杀掉：

.. code-block:: text

   WorkNCCL(SeqNum=1878, OpType=_ALLGATHER_BASE, ..., Timeout(ms)=1800000) ran for
   1800000 milliseconds before timing out.

报错里的超时值来自后端自带的默认值：NCCL 和 Gloo 是 1800000 ms，昇腾 HCCL 是 3636000 ms。

**可能原因：** 某个 rank 在两次 FSDP 集合通信之间耗时超过了这个时间，例如梯度累积步过大、checkpoint 写入过慢，或者该 rank 挂在调试器里，其余 rank 都在 all-gather 处等待。

**修复：** FSDP 的集合通信与 RLinf 其余 worker 间通信共用同一个超时，默认 180 分钟。用 ``RLINF_TIMEOUT`` 调大它（单位为分钟）：

.. code-block:: bash

   export RLINF_TIMEOUT=360

Ray 会在启动时捕获环境变量，因此必须在每个节点上 ``ray start`` **之前** 导出。

数值精度 / 推理后端
~~~~~~~~~~~~~~~~~~~

**提示：** RLinf 默认将 ``rollout.sglang.attention_backend`` 设为 ``triton``，以兼顾不同 GPU 的稳定性与兼容性（硬件支持时也可切换为 ``flashinfer``；使用 vLLM 时配置 ``rollout.vllm.attention_backend``，默认为 ``FLASH_ATTN``）：

.. code-block:: yaml

   rollout:
     sglang:
       attention_backend: triton  # 或 flashinfer

.. dropdown:: 进阶排查：torch_memory_saver.cpp 中 CUDA CUresult Error（result=2）

   **现象：** ``CUresult error result=2 file=csrc/torch_memory_saver.cpp func=cu_mem_create line=103``

   **原因：** SGLang 恢复缓存缓冲区时可用显存不足，常见于更新前未卸载推理权重。

   **修复：**

   - 降低 SGLang 的静态显存占用（例如调低 ``gpu_memory_utilization``）。
   - 确保重新加载前已释放推理权重。

.. dropdown:: 进阶排查：Gloo 超时 / “Global rank x is not part of group”

   **现象：**

   - ``RuntimeError: [../third_party/gloo/.../unbound_buffer.cc:81] Timed out waiting ... for recv``
   - ``ValueError: Global rank xxx is not part of group``

   **可能原因：** 上一阶段的 SGLang 故障（见上方 ``CUresult`` 错误）导致生成未完成，Megatron 持续等待直至 Gloo 超时。

   **修复：**

   1. 在日志中定位上一阶段的 SGLang 错误。
   2. 先解决 SGLang 的恢复或显存问题。
   3. 重新启动作业（必要时先执行 ``ray stop`` 重启 Ray）。

.. dropdown:: 进阶排查：创建通信组时 worker 中止（pybind11_object_dealloc）

   **现象：** worker 在连接新的对端时退出，日志中的第一条原生错误为：

   .. code-block:: text

      pybind11_object_dealloc(): Tried to deallocate unregistered instance!

   随后进程收到 ``SIGABRT``，driver 端只报告 actor 已退出。

   **可能原因：** PyTorch 2.7 之前，``torch.distributed`` 中有若干构造函数（包括 Gloo 后端和 ``TCPStore``）在未持有 GIL 的情况下登记新建的 Python 对象。RLinf 在不同线程上为不同对端创建通信组，两次登记可能重叠，从而破坏 pybind11 记录存活对象的表。BEHAVIOR 环境（PyTorch 2.5.1）和 Ascend 默认环境（PyTorch 2.6.0）都受影响。

   **修复：** 在这些版本上，RLinf 改用持有 GIL 完成登记的构造函数来创建 Gloo 后端和通信组选项，Gloo 通信组因此不再触发该问题。``TCPStore`` 以及 ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` 启用的调试包装没有这样的构造函数；``TCPStore`` 每个通信组只创建一次，因此仍可能偶发中止。条件允许时，请使用 PyTorch 2.7 及以上版本。报告问题时，请附上 PyTorch 版本和第一份原生调用栈，Ray 的 ``actor died`` 信息不足以判断原因。

.. _faq-rendering:

渲染问题
--------

EGL 渲染使用哪块 GPU？
~~~~~~~~~~~~~~~~~~~~~~

RLinf 会为每个 worker 指定渲染设备：向驱动查询该 worker 所分配 GPU 对应的 EGL 索引，并将其导出为 ``MUJOCO_EGL_DEVICE_ID``\ （MuJoCo、robosuite）和 ``EGL_DEVICE_ID``\ （其他 EGL 渲染器，如 pyrender）。这一步是必要的，因为 EGL 设备索引与 CUDA 设备号属于两套不同的命名空间：EGL 会列出驱动可见的所有设备，因此容器只分配到节点上的部分 GPU 时，CUDA 设备 0 通常\ **并不是** EGL 设备 0。

仅在需要覆盖该选择时才手动设置 ``MUJOCO_EGL_DEVICE_ID``，显式设置的值始终优先。若将其设为 CUDA 序号，在两套命名空间不一致时会渲染到错误的 GPU 上。

MUJOCO_EGL_DEVICE_ID 超出范围
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 运行将 ``MUJOCO_GL`` 设为 ``"egl"`` 的模拟器时报如下错误：

.. code-block:: text

   RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and 0 (inclusive), got 1.

**原因：** GPU 环境未正确配置图形渲染支持，尤其是在 NVIDIA GPU 上。

**修复：** 检查是否存在 ``/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0``。

1. 如果存在该文件，检查是否同时存在 ``/usr/share/glvnd/egl_vendor.d/10_nvidia.json``。如果没有，创建此文件并写入以下内容：

   .. code-block:: json

      {
         "file_format_version" : "1.0.0",
         "ICD" : {
            "library_path" : "libEGL_nvidia.so.0"
         }
      }

   然后在运行脚本中添加以下环境变量：

   .. code-block:: shell

      export NVIDIA_DRIVER_CAPABILITIES="all"

2. 如果不存在该文件，表示 NVIDIA 驱动未安装图形功能。可以采用以下任一方案：

   * 重新安装带图形能力的 NVIDIA 驱动。在 Ubuntu 上运行 ``apt install libnvidia-gl-<driver-version>``，详见 NVIDIA 的 `Ubuntu 驱动安装文档 <https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html#compute-only-headless-and-desktop-only-no-compute-installation>`__。

   * 改用 **osmesa** 渲染：在运行脚本中将 ``MUJOCO_GL`` 和 ``PYOPENGL_PLATFORM`` 设为 ``"osmesa"``。使用 CPU 渲染的 osmesa 速度通常比 EGL 慢约 10 倍。

Vulkan 驱动不兼容
~~~~~~~~~~~~~~~~~

**现象：** 运行基于 Vulkan 的模拟器（如 ManiSkill）时报驱动不兼容。

**原因：** GPU 驱动未安装图形渲染能力，或版本不兼容。

**修复：**

1. 按上述步骤确认 GPU 驱动已安装图形渲染支持。
2. 使用 ``nvidia-smi`` 查看驱动版本：

   * Ampere（A100、A800、RTX 30 系列）推荐使用 535 版本（常测：535.161.08）。570、580 等高版本已知可能触发该问题。
   * Hopper（H100）与 Ada Lovelace（L40S、RTX 40 系列）推荐使用 570 版本。

.. _faq-debugging:

断点调试
--------

使用 Ray 分布式调试器（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 VSCode 或 Cursor 中用 Ray Distributed Debugger 调试 Ray worker：

1. 安装 ``debugpy``：

   .. code-block:: bash

      uv pip install debugpy

2. 在 VSCode 或 Cursor 中安装 “Ray Distributed Debugger” 扩展（也可从 `Open VSX <https://open-vsx.org/extension/anyscalecompute/ray-distributed-debugger>`__ 下载 VSIX）。随后打开 Ray 侧边栏，点击 “+” 添加 ``127.0.0.1:8265``\ （或 Ray head 地址），并将本地目录设为 RLinf 仓库路径。

3. 在需要暂停的位置插入 ``breakpoint()``，再运行训练脚本：

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

4. 在 Ray 扩展的 “Paused Tasks” 列表中连接到暂停的 worker。调试完断开连接即可切换其他任务；首次连接报错时，执行 ``Developer: Reload Window`` 重载窗口。

参考：`Ray Distributed Debugger 文档 <https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html>`__。

使用 Ray Legacy Debugger（备用）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

无法安装编辑器扩展时，插入 ``breakpoint()`` 并设置 ``RAY_DEBUG=legacy`` 启动训练：

.. code-block:: bash

   RAY_DEBUG=legacy bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

日志提示 ``use 'ray debug' to connect`` 后，另开终端执行 ``ray debug`` 即可连接调试。参考：`Ray 调试文档 <https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html>`__。
