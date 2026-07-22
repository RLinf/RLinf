RTC 评测
========

实时控制（Real-Time Control，RTC）是一种在部署期间隐藏推理延迟的技术。它将一个 action chunk 的执行与下一个 action chunk 的推理重叠：策略输出一个动作序列（chunk），环境开始逐帧执行；在执行过程中，RTC 提前请求下一 chunk 的推理结果，使推理延迟被动作执行时间覆盖。

RTC 同时支持仿真（LIBERO）和真机（Franka）评测，当前与 OpenPI π₀.₅ 策略集成。

RTC 的工作原理
--------------

RTC 的核心思想是将策略推理与动作执行流水线化。标准 rollout 流程是同步的（推理 → 执行 chunk → 推理 → 执行 chunk），推理期间 GPU 计算，执行期间 GPU 空闲。RTC 改为异步流水线：

流程：

1. **Bootstrap**：策略执行初始推理，产生第一个 action chunk。
2. **执行与重叠**：环境开始逐帧执行当前 chunk。执行 ``min_exec_horizon`` 步后，Env Worker 异步发送下一帧 obs 到 Rollout Worker 请求推理。
3. **延迟自适应**：RTC 维护一个滑动窗口（``delay_buffer_size``）记录实际观测到的推理延迟，用于预测下一轮请求时机。
4. **停止**：episode 结束（或达到最大步数）后，Env Worker 发送 ``stop`` 信号终止 Rollout Worker 循环。

RTC 使用 ``软重叠引导``（soft overlap guidance）来平滑 chunk 边界。当新 chunk 到达时，前一个 chunk 可能还有若干步未执行。RTC 引导机制使新 chunk 的前 ``delay_steps`` 步与旧 chunk 的剩余部分保持一致，避免动作突变：

- ``硬掩码`` （hard mask）：前 ``delay_steps`` 步，强制使用旧 chunk 的剩余动作。
- ``软掩码`` （soft mask）：后续若干步，通过指数衰减引导新 chunk 逐渐过渡到其自身预测。
- 引导强度由 ``rtc_guidance_clip`` 限制，避免过度修正。

仿真实验
--------

环境安装与模型下载

.. code:: bash

   # 安装依赖
   bash requirements/install.sh embodied --model openvla --env maniskill_libero

   # 下载模型
   hf download RLinf/RLinf-Pi05-LIBERO-SFT --local-dir ./RLinf-Pi05-LIBERO-SFT

运行 RTC 评测

.. code:: bash

   bash evaluations/run_eval.sh libero_spatial_eval_pi05_RTC

启用 / 禁用 RTC 通过 YAML 中 ``runner.rtc.enabled`` 控制（默认 ``True``）：

.. code:: bash

   bash evaluations/run_eval.sh libero_spatial_eval_pi05_RTC \
       'runner.rtc.enabled=True'

.. note::

   仿真中使用 ``runner.rtc.chunk_pause_seconds`` 模拟动作执行时间。

实验结果
~~~~~~~~

``action_chunk=8``，LIBERO Spatial 评测：

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - 指标
     - RTC 关闭
     - RTC 开启
   * - ``episode_time`` （秒）
     - 56.93
     - 54.02
   * - ``step_total_time``
     - 45.73
     - 45.96
   * - ``wait_inference_time``
     - 5.61
     - 0.002
   * - ``env_reset_time``
     - 5.42
     - 5.15
   * - ``rtc_bootstrap_time``
     -
     - 2.02
   * - ``residual``
     - 0.17
     - 0.89
   * - ``speed_up`` （秒）
     -
     - 2.91

``action_chunk=5``，LIBERO Spatial 评测：

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - 指标
     - RTC 关闭
     - RTC 开启
   * - ``episode_time``
     - 60.19
     - 52.76
   * - ``step_total_time``
     - 45.57
     - 45.55
   * - ``wait_inference_time``
     - 8.46
     - 0.002
   * - ``env_reset_time``
     - 5.91
     - 5.12
   * - ``rtc_bootstrap_time``
     -
     - 1.24
   * - ``residual``
     - 0.09
     - 0.85
   * - ``speed_up``
     -
     - 7.43

RTC 将 ``wait_inference_time`` 从数秒降至约 2ms（接近零），几乎完全隐藏了推理延迟。``action_chunk`` 越小，推理请求越频繁，加速效果越明显（7.4s vs 2.9s）。

真机实验
--------

真机部署请参考 :doc:`Franka 真机强化学习 <franka>`。此处仅说明 RTC 评测的差异。

控制节点安装

.. code:: bash

   bash requirements/install.sh embodied --env franka

训练节点安装

.. code:: bash

   bash requirements/install.sh embodied --model openvla --env maniskill_libero

下载模型

.. code:: bash

   hf download RLinf/RLinf-Pi05-Pick_Red --local-dir ./RLinf-Pi05-Pick_Red

启动 Ray 集群

训练节点（head）：

.. code:: bash

   source ray_utils/realworld/setup_before_ray.sh
   ray start --head --port=6379 --node-ip-address=<head_node_ip_address>

控制节点（worker）：

.. code:: bash

   source .venv/franka_catkin_ws/devel/setup.bash
   source ray_utils/realworld/setup_before_ray.sh
   ray start --address='<head_node_ip_address>:6379'

在训练节点上启动评测

.. code:: bash

   bash evaluations/run_eval.sh realworld_pnp_eval_pi05_sft_RTC

实验结果
~~~~~~~~

``action_chunk=8``，Franka PnP 真机评测：

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - 指标
     - RTC 关闭
     - RTC 开启
   * - ``episode_time``
     - 48.18
     - 45.56
   * - ``step_total_time``
     - 35.58
     - 36.04
   * - ``wait_inference_time``
     - 5.64
     - 0.003
   * - ``env_reset_time``
     - 6.60
     - 6.51
   * - ``rtc_bootstrap_time``
     -
     - 0.85
   * - ``residual``
     - 0.37
     - 2.16
   * - ``speed_up``
     -
     - 2.63

``action_chunk=5``，Franka PnP 真机评测：

.. list-table::
   :header-rows: 1
   :widths: 24 16 16

   * - 指标
     - RTC 关闭
     - RTC 开启
   * - ``episode_time``
     - 51.66
     - 44.05
   * - ``step_total_time``
     - 35.80
     - 34.34
   * - ``wait_inference_time``
     - 8.56
     - 0.004
   * - ``env_reset_time``
     - 6.67
     - 6.55
   * - ``rtc_bootstrap_time``
     -
     - 1.00
   * - ``residual``
     - 0.63
     - 2.15
   * - ``speed_up``
     -
     - 7.61

.. note::

   真机实验中 ``residual`` 增加约 1.5s，来自 RTC 异步通信的额外开销（发送请求、接收响应的网络传输）。即便如此，RTC 仍然在推理延迟上节省了 5-8 秒，总 episode 时间加速约 3-8 秒。

指标说明
--------

.. list-table::
   :header-rows: 1
   :widths: 20 40

   * - 指标
     - 含义
   * - ``episode_time``
     - 整个 rollout 的总时间（从 reset 到 episode 结束），单位秒。
   * - ``step_total_time``
     - 整个 rollout 中执行所有动作的累加时间，单位秒。仿真中为 ``chunk_step`` 耗时；真机中为机械臂实际执行时间。
   * - ``wait_inference_time``
     - 整个 rollout 中等待推理结果的总时间，单位秒。RTC 开启时该值接近零（推理被动作执行隐藏）。
   * - ``env_reset_time``
     - 环境重置耗时，单位秒。
   * - ``rtc_bootstrap_time``
     - RTC 首次发送 action request 到收到 response 的时间，单位秒。仅 RTC 开启时存在。
   * - ``residual``
     - 其他开销：``episode_time - step_total_time - wait_inference_time - env_reset_time - rtc_bootstrap_time``，单位秒。
   * - ``speed_up``
     - ``wait_inference_time (disable) - wait_inference_time (enable)``，即 RTC 节省的推理等待时间，单位秒。

配置参考
--------

RTC 参数分布在两个位置：

``runner.rtc`` （控制 RTC 循环行为）

.. list-table::
   :header-rows: 1
   :widths: 20 10 50

   * - 参数
     - 默认值
     - 说明
   * - ``enabled``
     - ``False``
     - 是否启用 RTC。设置为 ``True`` 时，Env Worker 使用异步重叠推理模式；Rollout Worker 使用无限循环等待请求，直到收到 ``stop`` 信号。
   * - ``fixed_delay_steps``
     - ``0``
     - 仿真专用：模拟推理延迟的固定步数。仅在仿真中有效，真机中强制为 0。
   * - ``chunk_pause_seconds``
     - ``0.0``
     - 仿真专用：每一步执行后暂停的秒数，用于模拟真机执行时间。真机中强制为 0。
   * - ``inject_delay_ms``
     - ``0``
     - 真机专用：模拟推理延迟的毫秒数。仿真中强制为 0。
   * - ``min_exec_horizon``
     - ``2``
     - 开始请求下一 chunk 前至少执行的步数。过小可能导致请求过于频繁，过大则减少重叠覆盖。
   * - ``initial_delay_steps``
     - ``1``
     - 首次请求时预测的延迟步数。用于 bootstrap 后第一次 replan 请求的初始估计。
   * - ``delay_buffer_size``
     - ``8``
     - 滑动窗口大小，用于估计推理延迟。RTC 使用窗口内的最大延迟步数作为下一轮预测值。

``model.openpi.rtc_*`` （控制 RTC 引导行为，建议与 ``runner.rtc.enabled`` 保持同步）

.. list-table::
   :header-rows: 1
   :widths: 20 10 50

   * - 参数
     - 默认值
     - 说明
   * - ``rtc_enabled``
     - ``False``
     - 建议使用 Hydra 引用 ``${runner.rtc.enabled}`` 保持同步。启用后模型输出 action chunk 时会参考前一 chunk 的剩余动作。
   * - ``rtc_guidance_mode``
     - ``"approx"``
     - 引导模式，当前仅支持 ``"approx"`` （近似引导）。
   * - ``rtc_guidance_clip``
     - ``5.0``
     - 引导强度裁剪阈值。值越大引导越强，chunk 边界越平滑；过大可能导致动作偏离。推荐范围 3.0-10.0。
