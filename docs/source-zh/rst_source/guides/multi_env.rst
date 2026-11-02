在一次运行中使用多个环境
========================

本页介绍如何让同一个 policy 同时在多个环境上训练和评估，例如 ManiSkill 与 LIBERO、两个 LIBERO 任务集，或仿真环境与真机并行。常规的具身配置里，``env.train`` 和 ``env.eval`` 各对应一个环境。下面我们先在一个 mode 下列出多个环境，再确定每个环境分到多少并行实例，然后让它们的观测能够拼进同一个 policy batch，最后分别查看每个环境的结果。页面末尾给出一个可直接运行的双仿真器示例，并列出目前仍只支持单环境的功能。

在一个 mode 下列出多个环境
--------------------------

起点仍是你熟悉的环境预设。原来把一个预设挂载到 ``env.train``，现在改为把每个预设挂到 ``env.train.envs`` 下的一个名字上。名字由你决定，RLinf 会用它组成指标的 key 和视频路径。``env.eval`` 的写法相同，而且两个 mode 互不影响：可以在两个环境上训练、只在一个环境上评估，反过来也可以。

下面的配置在 LIBERO-Goal 和一个 ManiSkill 场景上训练，各使用 8 个并行环境：

.. code-block:: yaml

   defaults:
     - env/libero_goal@env.train.envs.libero_goal
     - env/maniskill_put_on_plate_in_scene_25_main@env.train.envs.maniskill
     - env/libero_goal@env.eval.envs.libero_goal
     - env/maniskill_ood_template@env.eval.envs.maniskill
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

``envs`` 下的每一项都是完整的环境配置，字段与单环境时的 ``env.train`` 一致，包括 ``env_type``、任务设置、``init_params``、``video_cfg`` 和奖励选项。其中的 ``total_num_envs`` 表示这个环境分到的并行环境数。mode 本身只保留描述整个 rollout 的字段，例如这里的 ``rollout_epoch``。

校验配置时，RLinf 会把 rollout、actor 和 runner 需要读取的汇总值写回 mode：``env.train.total_num_envs`` 成为各环境份额之和 16，所有环境必须一致的字段也一并写回。哪些字段必须一致，见下一节。

保持 rollout 循环一致
---------------------

同一个 mode 下的所有环境由同一个 rollout 循环推进，actor 也用同一条规则屏蔽已结束的 episode。因此有几个字段描述的是循环本身而不是某个环境，每个环境都必须取相同的值：

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - 字段
     - 必须一致的原因
   * - ``max_steps_per_rollout_epoch``
     - rollout 对每个 env worker 执行相同数量的 chunk step。
   * - ``auto_reset``
     - actor 根据这一个值判断 episode 是否会在一个 rollout epoch 内重新开始。
   * - ``ignore_terminations``
     - actor 为整个 batch 构造同一个 loss mask。

不同环境预设的默认值并不相同（LIBERO-Goal 每个 epoch 走 320 步，这个 ManiSkill 场景走 80 步），所以要像示例那样在每个环境上显式设置这些字段。取值不一致时，启动阶段就会报错，并列出每个环境的取值。

``max_episode_steps`` 的情况不同。它是每个环境自己的截断步数，各环境可以不一样；但 actor 还会用 ``env.train.max_episode_steps`` 这一个值对 loss 做归一化。所有训练环境取值一致时，RLinf 会把它写回 mode；不一致时，请自行设置 ``env.train.max_episode_steps``，作为 loss 归一化所用的步数。

mode 这一层只接受 ``envs``、``rollout_epoch``、``total_num_envs``、``max_episode_steps``、``policy_obs`` 以及上表中的共享字段。其他字段（例如 ``env.train.video_cfg``）写在这里不会作用到任何环境，RLinf 会拒绝这样的配置，并提示你把它写到各个环境里。

为每个环境分配 worker
---------------------

汇总值确定之后，RLinf 把各个环境分配到 env worker 上。每个 env worker 只运行一个环境，而且每个 worker 承载的并行环境数相同，因为 env 与 rollout 之间的 channel 会把每个 batch 平均切分。各环境按列出的顺序依次占用连续的一段 worker，段的长度由该环境的份额决定。

假设 ``env`` 放置在 4 个 worker 上，mode 共有 64 个环境，即每个 worker 承载 16 个：

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - 环境
     - ``total_num_envs``
     - env worker
   * - ``libero_goal``
     - 48
     - 0、1、2
   * - ``maniskill``
     - 16
     - 3

如果某个环境的份额不是整数个 worker，启动阶段会报错。同样这两个环境若只放在 2 个 worker 上（每个 32），48 个 LIBERO 环境就要占一个半 worker，这时需要增加 worker 数，或调整各环境的份额。

在自己的那段 worker 里，环境的行为与它单独运行在这些 worker 上时完全相同：随机种子和重置状态只在承载它的 worker 之间划分；环境里的 ``override_cfgs`` 列表也按这段 worker 逐个对应，而不是按整个 worker group。

这些 worker 运行在哪里，由 placement 决定。多个 env worker 可以共用一块 GPU，因此 ``env: 0:0-1``\ （资源 0，进程 0 和 1）就能把两个环境放在同一块卡上。如果某个环境驱动的是真机，请调整 ``env`` 的 placement，让这个环境对应的那段 worker 落在机器人硬件上；跨 node group 的写法见 :doc:`hetero`。

.. warning::

   worker 按列出顺序分配。调整 ``envs`` 的顺序或修改份额，都会改变每个 worker rank 承载的环境，真机环境可能因此被分到 GPU 节点上。修改其中任何一项后，请重新检查 placement。

让观测拼进同一个 policy batch
-----------------------------

环境各自在自己的 worker 上运行，但观测会在 rollout 处汇合：每个 rollout worker 都把收到的观测拼接成一个 batch 交给 policy。来自不同仿真器的环境，很少会给出相同的视角和分辨率。LIBERO 给出 256x256 的主视角和腕部视角，上面的 ManiSkill 场景只给出一个 640x480 的视角，没有腕部相机。rollout 无法把它们叠在一起；某个视角在一个环境中存在、在另一个环境中缺失时，会直接报错。

在 mode 上设置 ``policy_obs``，描述 policy 实际读取的观测。``keys`` 列出需要发送的观测 key（``main_images``、``wrist_images``、``extra_view_images``、``states``、``task_descriptions``），其余 key 以 ``None`` 发送；``image_size`` 把所有图像视角缩放到同一个 ``[height, width]``：

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

这个设置适用于只输入一张图像、不使用本体感知的 OpenVLA-OFT，它只读取主视角和任务描述。如果 policy 需要读取 ``states``，各环境的 state 维度必须相同，因为 ``policy_obs`` 只缩放图像，不处理 state。

被处理的只是发给 policy 的那一份观测。奖励模型、视频录制和数据采集看到的仍是环境的原始观测。缩放在 env worker 的 CPU 上逐步执行，所以各环境图像尺寸本来一致时，不要设置 ``image_size``。

动作的流向正好相反。rollout 返回一个动作 batch，每个 env worker 按自己承载的环境转换属于它的那部分，因此 LIBERO 和 ManiSkill 各自使用自己的动作转换。不过 policy 仍然只输出一种动作空间，也只用一套动作归一化（对 OpenVLA-OFT 来说就是一个 ``unnorm_key``），所以各环境必须接受同一类动作，例如带夹爪指令的 7 维末端位姿增量。

分别查看每个环境的结果
----------------------

训练时，每个环境在汇总指标之外，还会以自己的名字上报一份指标。``env/success_once`` 是所有环境全部 episode 的平均值，``env/libero_goal/success_once`` 和 ``env/maniskill/success_once`` 则分别只统计一个环境。评估指标以同样的方式记录在 ``eval/`` 下；``episode_len``、交互延迟等其他环境指标也都有按环境划分的一份。

视频保存在 ``<video_base_dir>/seed_<n>/<name>/`` 下。每个环境的种子都从零开始编号，环境名可以避免共用同一个 ``video_base_dir`` 的两个环境互相覆盖文件。``data_collection`` 写出的 episode 数据本来就以 env worker 的全局 rank 命名。

运行示例
--------

端到端测试 ``maniskill_libero_grpo_openvlaoft`` 把本页的内容串在了一起：它用 GRPO 在 LIBERO-Goal 和 ManiSkill 上训练 OpenVLA-OFT，并在每一步之后对两个环境做评估。运行它需要同时安装两个仿真器的环境、配置中指定的 OpenVLA-OFT checkpoint 和 LoRA adapter，以及 2 块 GPU。

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

这组命令依次完成：

1. 把 OpenVLA-OFT 以及 ManiSkill、LIBERO 安装到同一个虚拟环境中。
2. 在 GPU 0 和 1 上启动两个 env worker：训练和评估时，worker 0 都承载 ``libero_goal``，worker 1 都承载 ``maniskill``。
3. 运行两个 GRPO step，并在 ``runner.logger.log_path`` 下的 TensorBoard 中为每个环境记录 ``env/<name>/*`` 和 ``eval/<name>/*``。

这个测试用 8 步的 episode 检查整条链路。用于真正的联合训练时，请加长步数、增加环境数，并保持共享字段一致。ManiSkill 场景还需要把资源文件放在 ``rlinf/envs/sim/maniskill/assets`` 下，具体做法见 :doc:`../examples/embodied/maniskill`。

假定单一环境的功能
------------------

下面这些功能假定一个 mode 中只有一个环境，或只有一类环境；mode 不满足这一前提时，启动阶段会报错：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 功能
     - 限制原因
   * - 平滑干预（``smooth_intervene``）
     - 它的动作绕过 policy，无法与其他环境的 policy 动作放进同一个 rollout batch。
   * - 实时控制评估（``runner.rtc``）
     - RTC rollout 根据唯一评估环境的类型选择延迟模型。
   * - RLT 仿真 replay
     - actor 只维护一种 replay 布局，因此训练环境要么全部是 ``maniskill_rlt``，要么都不是。
