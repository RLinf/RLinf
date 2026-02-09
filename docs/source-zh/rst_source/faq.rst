常见问题
========

下面整理了 RLinf 的常见问题。该部分会持续更新，欢迎大家不断提问，帮助我们改进！

断点调试方法
------------

使用 Ray 分布式调试器（推荐）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**说明：** 在 VSCode/Cursor 中使用 Ray Distributed Debugger 远程调试 Ray 任务（断点或 post-mortem）。

1. 准备环境（一次性）：

   .. code-block:: bash

      uv pip install debugpy

2. 在 VSCode/Cursor 中注册集群：

   - 在扩展市场搜索 “Ray Distributed Debugger” 并安装。
   - 如果找不到扩展，可从 `这里 <https://open-vsx.org/extension/anyscalecompute/ray-distributed-debugger>`__ 下载 VSIX，使用“Extensions: Install from VSIX”或拖拽到 VSCode/Cursor 窗口安装。
   - 在侧边栏打开 Ray 扩展面板，点击 “+” 注册集群地址（本地默认 ``127.0.0.1:8265``，或你通过 ``ray start --head`` 启动的地址）。
   - 点击已添加集群的 “⚙”（Configure Debugger Local Folder），填写 RLinf 仓库路径。

3. 在代码中任意位置添加 `breakpoint()`：

   .. code-block:: python

      chains = data["chains"]
      denoise_inds = data["denoise_inds"]
      
      # 添加断点
      breakpoint()
      
      # input transform
      observation = self.input_transform(data, transpose=False)
      observation = _model.Observation.from_dict(observation)

4. 运行任意 RLinf 脚本（示例：PPO Pi05）：

   .. code-block:: bash

      bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

5. 连接调试器：

   - 执行暂停时，任务会出现在 Ray 扩展的 “Paused Tasks” 列表。
   - 点击播放/连接图标开始交互式调试。
   - 调试完成后在 VSCode/Cursor 调试工具栏断开连接，便于切换到其他暂停任务。

常用提示：

- 若首次调试遇到 Runtime Error，可 `Ctrl+Shift+P` 选择 “Developer: Reload Window” 重新加载窗口。

**参考：** Ray Distributed Debugger 文档 https://docs.ray.io/en/latest/ray-observability/ray-distributed-debugger.html

使用 Ray Legacy Debugger（备用）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**说明：** 无法使用 Ray 分布式调试器扩展时的后备方案。

1. 在需要调试的位置设置断点：

.. code-block:: python

   chains = data["chains"]
   denoise_inds = data["denoise_inds"]
   
   # 添加断点
   breakpoint()
   
   # input transform
   observation = self.input_transform(data, transpose=False)
   observation = _model.Observation.from_dict(observation)

2. 运行程序（示例：PPO Pi05）：

.. code-block:: bash

   RAY_DEBUG=legacy bash examples/embodiment/run_embodiment.sh maniskill_ppo_openpi_pi05

3. 当程序提示 ``use 'ray debug' to connect ...`` 时，另开终端执行：

.. code-block:: bash

   ray debug

**参考：** Ray 调试文档 https://docs.ray.io/en/latest/ray-observability/user-guides/debug-apps/ray-debugging.html

渲染问题
--------

RuntimeError: The MUJOCO_EGL_DEVICE_ID environment variable must be an integer between 0 and 0 (inclusive), got 1.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**现象：** 运行设置了 MUJOCO_GL 环境变量为 "egl" 的模拟器时出现上述错误信息。

**原因：** 该错误是因为您的 GPU 环境未正确设置图形渲染，尤其是在 NVIDIA GPU 上。

**修复：** 检查您是否有此文件 `/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`。

1. 如果您有此文件，请检查您是否还拥有 `/usr/share/glvnd/egl_vendor.d/10_nvidia.json`。如果没有，请创建此文件并添加以下内容：

   .. code-block:: json

      {
         "file_format_version" : "1.0.0",
         "ICD" : {
            "library_path" : "libEGL_nvidia.so.0"
         }
      }

   然后在您的运行脚本中添加以下环境变量：

   .. code-block:: shell

      export NVIDIA_DRIVER_CAPABILITIES="all"

2. 如果您没有此文件，则表示您的 NVIDIA 驱动程序未正确安装图形功能。您可以尝试以下解决方案：

   * 重新安装 NVIDIA 驱动程序，并使用正确的选项启用图形功能。安装 NVIDIA 驱动程序时，有几个选项会禁用图形驱动程序。因此，您需要尝试安装NVIDIA的图形驱动。在Ubuntu上可以通过命令 ``apt install libnvidia-gl-<driver-version>`` 完成，具体参见NVIDIA的文档 https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/ubuntu.html#compute-only-headless-and-desktop-only-no-compute-installation 。

   * 使用 **osmesa** 进行渲染，将运行脚本中的 `MUJOCO_GL` 和 `PYOPENGL_PLATFORM` 环境变量更改为 "osmesa"。但是，这可能会导致滚动过程比 EGL 慢 10 倍，因为它使用 CPU 进行渲染。

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

CUDA 问题
----------

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

数值精度 / 推理后端
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**提示：** SGLang 默认使用 **flashinfer** 作为注意力实现。若需更高稳定性或兼容性，可尝试 **triton**：

.. code-block:: yaml

   rollout:
     attention_backend: triton
