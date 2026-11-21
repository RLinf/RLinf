常见问题
========

下面整理了 RLinf 的常见问题。该部分会持续更新，欢迎大家不断提问，帮助我们改进！

.. _faq-rendering:

渲染问题
--------

EGL 渲染使用哪块 GPU？
~~~~~~~~~~~~~~~~~~~~~~~~

RLinf 会为每个 worker 指定渲染设备：向驱动查询该 worker 所分配 GPU 对应的 EGL 索引，并将其导出为 ``MUJOCO_EGL_DEVICE_ID``\ （MuJoCo、robosuite）和 ``EGL_DEVICE_ID``\ （其他 EGL 渲染器，如 pyrender）。这一步是必要的，因为 EGL 设备索引与 CUDA 设备号属于两套不同的命名空间：EGL 会列出驱动可见的所有设备，因此容器只分配到节点上的部分 GPU 时，CUDA 设备 0 通常\ **并不是** EGL 设备 0。

仅在需要覆盖该选择时才手动设置 ``MUJOCO_EGL_DEVICE_ID``，显式设置的值始终优先。若将其设为 CUDA 序号，在两套命名空间不一致时会渲染到错误的 GPU 上。

RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and 0 (inclusive), got 1.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 运行设置了 MUJOCO_GL 环境变量为 "egl" 的模拟器时出现上述错误信息。

**原因：** 该错误是因为你的 GPU 环境未正确设置图形渲染，尤其是在 NVIDIA GPU 上。

**修复：** 检查是否包含文件 `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`。

1. 如果存在该文件，请检查是否还包含 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`。如果没有，请创建此文件并添加以下内容：

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

2. 如果不存在该文件，则表示 NVIDIA 驱动程序未正确安装图形功能。可以尝试以下解决方案：

   * 重新安装 NVIDIA 驱动程序，并使用正确的选项启用图形功能。安装 NVIDIA 驱动程序时，有几个选项会禁用图形驱动程序。因此，需要尝试安装 NVIDIA 的图形驱动。在 Ubuntu 上可以通过命令 ``apt install libnvidia-gl-<driver-version>`` 完成，具体参见 NVIDIA 的文档 https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html#compute-only-headless-and-desktop-only-no-compute-installation 。

   * 使用 **osmesa** 渲染，将运行脚本中的 `MUJOCO_GL` 和 `PYOPENGL_PLATFORM` 环境变量更改为 "osmesa"。但是，这可能会导致采样过程比 EGL 慢 10 倍，因为它使用 CPU 渲染。

Vulkan Incompatible GPU Driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 运行基于 Vulkan 的模拟器（如 ManiSkill）时报驱动不兼容。

**原因：** GPU 驱动未正确安装图形渲染能力，或版本不兼容。

**修复：**

1. 按上述步骤确认 GPU 驱动已安装图形渲染支持。
2. 使用 ``nvidia-smi`` 查看驱动版本：
   
   * Ampere（A100/A800/RTX 30 系列）推荐使用 535 版本（常测：535.161.08）。570/580 等高版本已知可能触发问题。
   * Hopper（H100/L40S/RTX 40 系列）推荐使用 570 版本。

网络问题
--------

无法连接 GCS（ip:port）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** Worker 节点无法连接到给定地址上的 Ray head（GCS）。

**原因：** 在 0 号节点上通过以下命令获取 head 节点 IP：

.. code-block:: bash

   hostname -I | awk '{print $1}'

若该命令选择了其他节点不可达的网卡（如网卡顺序不一致；可达的是 ``eth0``，却选中了别的接口），Worker 将连接失败。

**修复：**

- 确认所选 IP 能被其他节点访问（例如使用 ping 测试）。  
- 如有需要，请显式选择正确网卡对应的 IP 作为 Ray head，并将该 IP 告知各 Worker。

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** P2P 任务传输失败，报错 ``NCCL cuda invalid argument``。

**修复：** 若此机器上之前运行过任务，请先停止 Ray 并重新启动。

.. code-block:: bash

   ray stop

SGLang 加载参数时出现 NCCL “cuda invalid argument”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** SGLang 在加载权重时报 ``NCCL cuda invalid argument``。

**原因：** Placement 不匹配。例如配置使用 *共享式（collocated）*，但训练（trainer）与生成（generation）实际跑在不同 GPU 上。

**修复：** 检查 Placement 策略。确保训练组与生成组按照 ``cluster.component_placement`` 指定的 GPU 放置。

torch_memory_saver.cpp 中 CUDA CUresult Error（result=2）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：**
``CUresult error result=2 file=csrc/torch_memory_saver.cpp func=cu_mem_create line=103``

**原因：** SGLang 恢复缓存缓冲区时可用显存不足；常见于在更新前没有卸载推理权重的情况。

**修复：**

- 降低 SGLang 的静态显存占用（例如调低 ``gpu_memory_utilization``）。
- 确保在重新加载前，已正确释放推理权重。

Gloo 超时 / “Global rank x is not part of group”
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：**

- ``RuntimeError: [../third_party/gloo/.../unbound_buffer.cc:81] Timed out waiting ... for recv``
- ``ValueError: Global rank xxx is not part of group``

**可能原因：** 之前的 SGLang 故障（见上面的 CUresult 错误）导致生成阶段未完成，Megatron 随后一直等待，直到 Gloo 超时。

**修复：**

1. 在日志中定位上一阶段的 SGLang 错误。  
2. 先解决 SGLang 的恢复/显存问题。  
3. 重新启动作业（必要时也重启 Ray）。

FSDP 集合通信被后端看门狗超时杀掉
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 训练步仍在正常推进，FSDP actor 却被集合通信看门狗杀掉：

.. code-block:: text

   WorkNCCL(SeqNum=1878, OpType=_ALLGATHER_BASE, ..., Timeout(ms)=1800000) ran for
   1800000 milliseconds before timing out.

报错里的超时值来自后端自带的默认值——NCCL 和 Gloo 是 1800000 ms，昇腾 HCCL 是 3636000 ms。

**可能原因：** 某个 rank 在两次 FSDP 集合通信之间耗时超过了这个时间——例如梯度累积步过大、checkpoint 写入过慢，或者该 rank 挂在调试器里，而其余 rank 都在 all-gather 处等待。

**修复：** FSDP 的集合通信与 RLinf 其余 worker 间通信共用同一个超时，默认 180 分钟。用 ``RLINF_TIMEOUT`` 调大它（单位为分钟）：

.. code-block:: bash

   export RLINF_TIMEOUT=360

Ray 会在启动时捕获环境变量，因此必须在每个节点上 ``ray start`` **之前** 导出。

数值精度 / 推理后端
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**提示：** SGLang 默认使用 **flashinfer** 作为注意力实现。若需更高稳定性或兼容性，可尝试 **triton**：

.. code-block:: yaml

   rollout:
     attention_backend: triton

创建通信组时 worker 中止
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** worker 在连接新的对端时退出，日志中的第一条原生错误为：

.. code-block:: text

   pybind11_object_dealloc(): Tried to deallocate unregistered instance!

随后进程收到 ``SIGABRT``，driver 端只报告 actor 已退出。

**可能原因：** PyTorch 2.7 之前，``torch.distributed`` 中有若干构造函数（包括 Gloo 后端和 ``TCPStore``）在未持有 GIL 的情况下登记新建的 Python 对象。RLinf 在不同线程上为不同对端创建通信组，两次登记可能重叠，从而破坏 pybind11 记录存活对象的表。BEHAVIOR 环境（PyTorch 2.5.1）和 Ascend 默认环境（PyTorch 2.6.0）都受影响。

**修复：** 在这些版本上，RLinf 改用持有 GIL 完成登记的构造函数来创建 Gloo 后端和通信组选项，Gloo 通信组因此不再触发该问题。``TCPStore`` 以及 ``TORCH_DISTRIBUTED_DEBUG=DETAIL`` 启用的调试包装没有这样的构造函数；``TCPStore`` 每个通信组只创建一次，因此仍可能偶发中止。条件允许时，请使用 PyTorch 2.7 及以上版本。报告问题时，请附上 PyTorch 版本和第一份原生调用栈，Ray 的 ``actor died`` 信息不足以判断原因。

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
