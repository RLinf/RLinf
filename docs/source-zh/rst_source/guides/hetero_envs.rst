异构环境
========

本页介绍如何在一次运行中，让同一个 policy 在多种不同的环境配置上训练或评估，例如 ManiSkill 与 LIBERO、两个 LIBERO 任务集，或仿真环境与真机。RLinf 把这种用法称为异构环境。

它与 ``total_num_envs`` 不同， ``total_num_envs`` 是把同一份环境配置并行运行多份；它也不同于 :doc:`hetero` ，后者决定各个 worker 运行在哪些机器上。下面先说明异构环境如何运行，再依次介绍如何配置、如何决定哪些 env worker 运行哪个环境、如何让观测适配 policy，以及如何查看每个环境的结果。

运行方式
--------

通常所有 env worker 都运行同一份环境配置。使用异构环境时，每个 env worker 仍然只运行一份环境配置，但不同的 worker 可以运行不同的配置。

训练流程的其余部分保持不变。rollout 把所有 env worker 的观测收集成一个 batch，运行一次 policy，再把每个 worker 对应的动作发回去。每个 worker 把动作转换成自己的仿真器或机器人能接受的格式，然后推进环境。最后，actor 用全部轨迹一起训练。

由于所有数据最终汇入同一个 batch，需要满足两条规则：

- 每个 env worker 运行的并行环境数相同。
- 每份环境配置发给 policy 的观测种类和图像尺寸相同。

下文会说明 RLinf 如何自动满足第一条，以及你需要如何处理第二条。

配置异构环境
------------

把各份环境配置列在 ``env.train.envs`` 下，每份配置用一个你自己起的名字。下面的配置在 LIBERO-Goal 和一个 ManiSkill 场景上训练，各有 8 个并行环境：

.. code-block:: yaml

   defaults:
     - env/libero_goal@env.train.envs.libero_goal
     - env/maniskill_put_on_plate_in_scene_25_main@env.train.envs.maniskill
     # model、FSDP 和 weight syncer 预设与平常相同

   env:
     train:
       rollout_epoch: 1
       envs:
         libero_goal:
           total_num_envs: 8
           max_episode_steps: 8
           max_steps_per_rollout_epoch: 8
         maniskill:
           total_num_envs: 8
           max_episode_steps: 8
           max_steps_per_rollout_epoch: 8

每一行 ``defaults`` 把一个环境预设加载到一个名字下，这里是 ``libero_goal`` 和 ``maniskill`` 。每份环境配置可以使用普通 ``env.train`` 的全部设置，其中的 ``total_num_envs`` 是它自己的并行环境数。 ``env.eval`` 的写法相同，并且与 ``env.train`` 互相独立，例如可以在两份环境配置上训练、只在一份上评估。

有几项设置属于共享的 rollout 循环，所以每份环境配置都必须取相同的值： ``max_steps_per_rollout_epoch`` 、 ``auto_reset`` 和 ``ignore_terminations`` 。不同环境预设的默认值并不相同，请像示例那样在每份环境配置上显式设置。 ``max_episode_steps`` 可以各不相同；训练时如果取值不同，还需要设置 ``env.train.max_episode_steps`` ，actor 用它对 loss 做归一化。

其他设置（例如视频或奖励）都写在各自所属的环境配置里。 ``env.train.total_num_envs`` 这类汇总值由 RLinf 自动填写；设置互相冲突时，RLinf 会在启动阶段停止，并给出明确的报错信息。

分配 env worker
---------------

RLinf 按你列出的顺序，按各份环境配置 ``total_num_envs`` 的比例划分 env worker。假设 ``env`` 有 4 个 worker，配置了 48 个 LIBERO-Goal 环境和 16 个 ManiSkill 环境，那么每个 worker 运行 16 个环境：

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - 环境配置
     - ``total_num_envs``
     - env worker
   * - ``libero_goal``
     - 48
     - 0、1、2
   * - ``maniskill``
     - 16
     - 3

每份环境配置都必须占满整数个 worker。如果只有 2 个 worker，LIBERO-Goal 需要一个半 worker，RLinf 会拒绝这份配置；这时请调整环境数或 worker 数。

在分给自己的 worker 上，一份环境配置的行为与它单独运行时相同：随机种子和重置状态只在它自己的 worker 之间划分， ``override_cfgs`` 列表也是每个 worker 对应一项。

这些 worker 运行在哪里，由 ``env`` 的 placement 决定。多个 env worker 可以共用一块 GPU，例如 ``env: 0:0-1`` 。如果要驱动真机，请把这份环境配置的 worker 放到机器人所在的节点上，具体做法见 :doc:`hetero` 。

.. warning::

   worker 按列出的顺序分配。调整 ``envs`` 的顺序或修改环境数，都会改变每个 worker 运行的环境，修改后请重新检查 placement，使用真机时尤其如此。

对齐观测
--------

rollout 会把所有 env worker 的观测叠成一个 batch，因此这些观测必须一致。来自不同仿真器的环境通常并不一致：LIBERO 发送 256x256 的主相机和腕部相机图像，而这个 ManiSkill 场景只发送一个 480x640 的相机图像。不做处理的话，rollout 会直接报错。

设置 ``policy_obs`` ，指定 policy 接收哪些观测：

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` 列出 policy 读取的观测，其余观测不会发送； ``image_size`` 把所有相机图像缩放到同一个高度和宽度。这样设置后，两个环境都只发送 256x256 的主相机图像和任务描述，这正是 OpenVLA-OFT 需要的输入。

被改动的只是发给 policy 的那一份观测，奖励模型、视频录制和数据采集看到的仍是环境的原始观测。缩放在每一步都会消耗时间，所以图像尺寸本来一致时，不要设置 ``image_size`` 。 ``policy_obs`` 不会缩放机器人状态，如果 policy 读取 ``states`` ，各份环境配置的状态维度必须相同。

动作方向不需要额外设置：每个 env worker 会把属于自己的那部分动作转换成自己环境的格式。不过 policy 只有一种动作格式和一套动作归一化（对 OpenVLA-OFT 来说是一个 ``unnorm_key`` ），所以所有环境配置都必须接受同一类动作。

查看各环境的结果
----------------

每个指标都会记录两份：一份汇总所有环境，一份按环境配置区分。 ``env/success_once`` 统计全部 episode， ``env/libero_goal/success_once`` 只统计 LIBERO-Goal。评估指标以同样的方式记录在 ``eval/`` 下。

视频保存在 ``<video_base_dir>/seed_<n>/<name>/`` 下，因此共用同一个视频目录的环境配置不会互相覆盖文件。

运行示例
--------

端到端测试 ``maniskill_libero_grpo_openvlaoft`` 用 GRPO 在 LIBERO-Goal 和 ManiSkill 上训练 OpenVLA-OFT，并在两者上评估。运行它需要一个同时安装了两个仿真器的虚拟环境、配置中指定的 checkpoint 和 LoRA adapter，以及 2 块 GPU。

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

这组命令依次完成：

1. 安装 OpenVLA-OFT 以及 ManiSkill 和 LIBERO。
2. 在 env worker 0 上运行 LIBERO-Goal，在 env worker 1 上运行 ManiSkill。
3. 运行两个 GRPO step，并把 ``env/<name>/*`` 和 ``eval/<name>/*`` 记录到 ``runner.logger.log_path`` 下的 TensorBoard 中。

这个测试用 8 步的 episode 检查各部分能否正确连通。用于实际训练时，请加长 episode 并增加环境数。ManiSkill 场景还需要准备资源文件，具体做法见 :doc:`../examples/embodied/maniskill` 。

限制
----

- policy 只有一种动作格式和一套动作归一化，也无法得知某个样本来自哪个环境。
- 机器人状态不会被缩放，因此各份环境配置的状态维度必须相同。
- 平滑干预（ ``smooth_intervene`` ）和实时控制评估（ ``runner.rtc`` ）只支持单份环境配置；RLT 仿真 replay 要求训练中的每份环境配置都是 ``maniskill_rlt`` 。配置不满足这些要求时，RLinf 会在启动阶段停止。
