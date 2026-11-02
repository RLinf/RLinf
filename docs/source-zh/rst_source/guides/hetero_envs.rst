异构环境
========

本页介绍如何运行异构环境：在一次运行中，让不同的 env worker 运行不同的环境配置，同一个 policy 在一次 rollout 里同时作用于所有这些环境。典型用法包括在 ManiSkill 和 LIBERO 上联合训练、混合两个 LIBERO 任务集，或让仿真环境与真机一起运行。

默认情况下， ``env.train`` 和 ``env.eval`` 各自只包含一份环境配置，所有 env worker 都运行这份配置， ``total_num_envs`` 只决定它并行运行多少份；本页把这种情况称为同构。当 ``env.train`` 或 ``env.eval`` 在 ``envs`` 下列出多份环境配置时，它就成为异构的。异构硬件（见 :doc:`hetero`）是另一项功能，它决定 worker 运行在哪些节点上，而不决定每个 worker 运行哪个环境。当某个环境需要特定硬件（例如机器人）时，两者可以组合使用。

后面各步都建立在运行机制之上，所以本页先说明异构环境如何运行，再依次介绍如何配置、RLinf 如何把环境配置分配给 env worker、如何对齐 policy 的观测，以及如何查看各环境的结果。页面最后给出可运行的示例和目前的限制。

异构环境如何运行
----------------

先约定三个术语，后文都按这个含义使用：

- **环境配置** （env configuration）：一套完整的环境设置，包括 ``env_type`` 、任务、 ``init_params`` 、奖励选项等。
- **环境实例** （env instance）：一份环境配置的一个并行副本。 ``total_num_envs`` 统计的就是环境实例的数量。
- **env worker** ：一个 env worker 进程。下文用 ``W`` 表示 env worker 的数量。

``env.train`` 同构时，每个 env worker 都构建同一份环境配置，并推进其中 ``total_num_envs / W`` 个环境实例。异构环境保留这个工作单元，即每个 env worker 仍然只运行一份环境配置，区别只在于不同的 env worker 可以构建不同的环境配置。一次 rollout step 的流程与同构时相同：

1. 每个 env worker 推进自己的环境实例，并通过 env 与 rollout 之间的 channel 把观测发给 rollout。
2. 每个 rollout worker 把收到的观测拼接成一个 batch，运行一次 policy，再把每个 env worker 对应那几行的动作发回去。
3. 每个 env worker 按自己的 ``env_type`` 转换动作，然后推进环境实例。
4. rollout 结束后，actor 用所有 env worker 的轨迹一起训练。

rollout、actor 和 runner 中没有按环境类型区分的代码路径，每个仿真器或机器人 SDK 都运行在独立的进程里。第 1、2 步对一起运行的环境配置提出了两项要求：

- **batch 大小相同。** env 与 rollout 之间的 channel 要求每个 env worker 发送的 batch 大小相同，所以每个 env worker 承载的环境实例数必须相同。RLinf 如何据此划分 worker，见 `为环境配置分配 env worker`_ 。
- **观测可以拼接。** rollout 按 key 逐个拼接观测，所以每份环境配置发送的 key 和 shape 都必须一致。如何对齐观测，见 `对齐 policy 的观测`_ 。

配置异构环境
------------

要让 ``env.train`` 成为异构的，就把它的环境配置放到 ``envs`` 下，并为每份环境配置起一个名字。下面的配置在 LIBERO-Goal 和一个 ManiSkill 场景上训练，各有 8 个环境实例：

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

``env/<preset>@env.train.envs.<name>`` 形式的 defaults 条目会把一个环境预设加载为名为 ``<name>`` 的环境配置。名字由你决定，RLinf 会用它组成指标的 key 和视频路径。 ``envs`` 下的每份环境配置，字段都与同构的 ``env.train`` 相同，其中的 ``total_num_envs`` 是它的环境实例数。 ``env.eval`` 的写法相同。两者互相独立，其中任何一个都可以保持同构。

设置 ``envs`` 之后， ``env.train`` 自身只保留描述整个 rollout 的字段。其中一部分由你设置，其余由 RLinf 根据各份环境配置推导：

.. list-table::
   :header-rows: 1
   :widths: 40 15 45

   * - ``env.train`` 的字段
     - 设置方
     - 取值
   * - ``envs``
     - 你
     - 以名字为 key 的各份环境配置。
   * - ``rollout_epoch``
     - 你
     - 每个训练 step 的 rollout 轮数，与同构的 ``env.train`` 相同。
   * - ``policy_obs``
     - 你（可选）
     - policy 收到的观测格式，见 `对齐 policy 的观测`_ 。
   * - ``total_num_envs``
     - RLinf
     - 各份环境配置 ``total_num_envs`` 之和，示例中为 16。
   * - ``max_steps_per_rollout_epoch`` 、 ``auto_reset`` 、 ``ignore_terminations``
     - RLinf
     - 从各份环境配置复制，它们必须取相同的值。
   * - ``max_episode_steps``
     - RLinf 或你
     - 所有环境配置取值一致时复制；在 ``env.train`` 中，取值不一致时需要你自行设置。

这些复制上来的字段描述的是共享的 rollout 循环，而不是某一个环境。rollout 对每个 env worker 同步推进，每个 rollout 轮推进 ``max_steps_per_rollout_epoch`` 个环境步；actor 屏蔽已结束的 episode 时，对整个 batch 使用同一条 ``auto_reset`` 和 ``ignore_terminations`` 规则，并用同一个 ``max_episode_steps`` 对 loss 做归一化。每份环境配置仍可以用各自的 ``max_episode_steps`` 截断自己的 episode。不同预设的默认值并不相同（LIBERO-Goal 每个 rollout 轮走 320 步，这个 ManiSkill 场景走 80 步），所以要像示例那样，在每份环境配置上显式设置这些共享字段。

RLinf 在启动时校验配置，这些字段就在这一步推导一次并写回 ``env.train`` 。之后 rollout、actor 和 runner 读取它们的方式与同构的 ``env.train`` 完全相同。如果你自己设置了其中某个字段，它必须等于推导出的值。直接写在 ``env.train`` 下的其他字段（例如 ``env.train.video_cfg`` ）不会被任何环境读取，因此会被拒绝，请把它们写到各份环境配置里。

校验还会逐份检查环境配置，一旦出错就停止运行，并在报错信息中给出出错的环境配置和冲突的取值：

- 每份环境配置都设置了 ``env_type`` ，以及一个正整数 ``total_num_envs`` 。
- 需要复制的字段在所有环境配置中取值相同。
- 按环境类型的规则只作用于对应的环境配置：ManiSkill 环境配置根据模型的 ``policy_setup`` 得到 ``init_params.control_mode`` ；Behavior 环境配置必须使用 ``r1pro_behavior`` ；开启在线 LeRobot 采集时，LIBERO、RoboTwin 和 Behavior 环境配置需要设置 ``skip_intermediate_renders: False`` 。
- 每份环境配置的 ``group_size`` 能整除每个 env worker、每个 pipeline stage 上的环境实例数，因此一个 GRPO group 不会跨越两份环境配置。

为环境配置分配 env worker
-------------------------

校验推导出汇总值之后，RLinf 按 ``envs`` 中列出的顺序，为每份环境配置分配一段连续的 env worker rank。设共有 ``W`` 个 env worker，各份环境配置的环境实例数为 ``n_1, ..., n_k`` ：

1. 每个 env worker 承载 ``e = (n_1 + ... + n_k) / W`` 个环境实例。总数必须能被 ``W`` 整除。
2. 第 ``i`` 份环境配置分到 ``n_i / e`` 个 env worker。每个 ``n_i`` 都必须是 ``e`` 的整数倍。
3. 第一份环境配置占用 rank ``0`` 到 ``n_1 / e - 1`` ，第二份占用接下来的 ``n_2 / e`` 个 rank，依此类推。

例如有 4 个 env worker、48 个 LIBERO-Goal 实例和 16 个 ManiSkill 实例时， ``e = 64 / 4 = 16`` ：

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - 环境配置
     - ``total_num_envs``
     - env worker rank
   * - ``libero_goal``
     - 48
     - 0、1、2
   * - ``maniskill``
     - 16
     - 3

如果只有 2 个 env worker，则 ``e = 32`` ， ``libero_goal`` 需要 1.5 个 worker，校验会失败。请调整各份环境配置的实例数或 env worker 数量，直到每份环境配置都分到整数个 worker。

在自己的那段 worker 中，一份环境配置的行为与它单独运行在 ``n_i / e`` 个 env worker 上时完全相同。每个 env worker 会得到一个局部 rank，即它的 rank 减去这段 worker 的起始 rank；凡是同构环境使用全局 rank 的地方，这里都改用局部 rank：

- 随机种子和重置状态只在这段 worker 之间划分。
- 环境配置中 ``override_cfgs`` 的第 ``k`` 项作用于这段 worker 的局部 rank ``k`` ，因此列表的长度等于这段 worker 的数量。

因此，一份环境配置的随机种子和重置状态只取决于它自身和这段 worker 的大小，与这段 worker 从哪个 rank 开始无关。

rank 的顺序遵循 ``env`` 的 placement，因此 placement 决定了每段 worker 运行在哪里。多个 env worker 可以共用一块 GPU，例如 ``env: 0:0-1`` 会把 rank 0 和 rank 1 都放在 GPU 0 上。要让某份环境配置运行在机器人硬件上，就把它那段 rank 放到机器人节点上；node group 的写法见 :doc:`hetero` 。

.. warning::

   rank 按 ``envs`` 中列出的顺序分配。调整环境配置的顺序或修改某份环境配置的实例数，都会改变每个 rank 运行的环境配置，真机环境配置可能因此被分到 GPU 节点上。修改其中任何一项后，请重新检查 placement。

对齐 policy 的观测
------------------

每个 env worker 运行的是自己的环境配置，但它们的观测最终仍要进入同一个 policy batch。每个 env worker 发给 rollout 的观测包含以下 key：

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - key
     - shape
   * - ``main_images``
     - ``[B, H, W, C]``
   * - ``wrist_images``
     - ``[B, H, W, C]`` ，有 ``V`` 个视角时为 ``[B, V, H, W, C]``
   * - ``extra_view_images``
     - ``[B, V, H, W, C]``
   * - ``states``
     - ``[B, D]``
   * - ``task_descriptions``
     - ``B`` 个字符串

rollout 拼接多个 env worker 的 batch 时，每个 key 必须在所有 env worker 上都存在或都不存在，而且张量在 ``B`` 之后的每一维都必须一致。来自不同仿真器的环境配置很少能满足这一点。在上面的示例中，LIBERO 发送 256x256 的主视角和一个腕部视角，ManiSkill 场景只发送一个 480x640（高 x 宽）的主视角、没有腕部视角，因此 rollout 会报 ``Cannot merge present and absent batch values`` 并停止。

在 ``env.train`` 或 ``env.eval`` 上设置 ``policy_obs`` ，让所有 env worker 使用同一种输出格式：

.. code-block:: yaml

   env:
     train:
       policy_obs:
         keys: [main_images, task_descriptions]
         image_size: [256, 256]

``keys`` 列出 policy 需要接收的 key，其余 key 都以 ``None`` 发送，因此在所有 env worker 的 batch 中都不存在。 ``image_size`` 的格式是 ``[height, width]`` ，所有图像 key 都会在 ``H`` 和 ``W`` 两个维度上用带抗混叠的双线性插值缩放到这个尺寸，其他维度保持不变。整数类型的图像会先取整再截断到取值范围内，所以 ``uint8`` 仍然是 ``uint8`` 。已经是目标尺寸的图像原样通过。按示例设置之后，两份环境配置都会发送 ``[B, 256, 256, 3]`` 的 ``uint8`` 图像 ``main_images`` ，以及 ``task_descriptions`` ，这正是 OpenVLA-OFT 读取的输入。

每个 env worker 只在发给 rollout 的那份观测上应用 ``policy_obs`` ，训练和评估都是如此。奖励模型、视频录制和数据采集拿到的仍是环境的原始观测。缩放在每个 chunk step 中对每个图像 key 执行一次，运行在环境产生图像的设备上：LIBERO 在 CPU 上，使用 GPU 后端的 ManiSkill 在仿真所用的 GPU 上。如果各环境配置的分辨率本来就一致，就不要设置 ``image_size`` 。同构的 ``env.train`` 或 ``env.eval`` 同样可以使用 ``policy_obs`` 。

动作方向不需要对应的设置。rollout 返回一个动作张量，每个 env worker 先按自己的 ``env_type`` 转换属于它的那几行，再推进环境。不过 policy 仍然只输出一种动作空间，也只使用一套动作归一化，对 OpenVLA-OFT 来说就是一个 ``unnorm_key`` 。因此每份环境配置都必须接受同一类动作，例如带夹爪指令的 7 维末端位姿增量。

查看各环境的结果
----------------

一个训练 step 中混合了来自多份环境配置的 episode，所以 RLinf 既给出汇总指标，也给出每份环境配置的指标。使用异构环境时，每个 env worker 会把每个指标上报两次：一次用原来的 key，一次用 ``<name>/<key>`` 。runner 把所有上报同一个 key 的 env worker 的逐 episode 数值拼接起来，再取平均：

- ``env/success_once`` 是所有环境配置全部 episode 的平均值。
- ``env/libero_goal/success_once`` 只统计 ``libero_goal`` 的 episode。

评估指标以同样的方式记录在 ``eval/`` 下。每个环境指标都有按环境配置划分的一份，包括 ``episode_len`` 和交互延迟统计。

视频保存在 ``<video_base_dir>/seed_<n>/<name>/`` 下。每份环境配置的种子都从 0 开始编号，有了名字，共用同一个 ``video_base_dir`` 的环境配置就不会互相覆盖文件。数据采集写出的文件以 env worker 的全局 rank 命名，本身就能区分不同的环境配置。

运行示例
--------

端到端测试 ``maniskill_libero_grpo_openvlaoft`` 用到了本页的全部内容：它用 GRPO 在 LIBERO-Goal 和 ManiSkill 上训练 OpenVLA-OFT，并在每一步之后对两者做评估。运行它需要一个同时安装了两个仿真器的虚拟环境、配置中指定的 OpenVLA-OFT checkpoint 和 LoRA adapter，以及 2 块 GPU。

.. code-block:: bash

   bash requirements/install.sh embodied --model openvla-oft --env maniskill_libero
   source .venv/bin/activate
   export REPO_PATH=$(pwd)
   bash tests/e2e_tests/embodied/run.sh maniskill_libero_grpo_openvlaoft

这组命令依次完成：

1. 把 OpenVLA-OFT、ManiSkill 和 LIBERO 安装到同一个虚拟环境中。
2. 在 GPU 0 和 GPU 1 上启动两个 env worker。训练和评估时，rank 0 都运行 ``libero_goal`` ，rank 1 都运行 ``maniskill`` 。
3. 运行两个 GRPO step，并在 ``runner.logger.log_path`` 下的 TensorBoard 中为每份环境配置记录 ``env/<name>/*`` 和 ``eval/<name>/*`` 。

这个测试用 8 步的 episode 检查数据通路。用于真正的联合训练时，请加长步数、增加环境实例数，并保持共享字段一致。ManiSkill 场景还需要把资源文件放在 ``rlinf/envs/sim/maniskill/assets`` 下，具体做法见 :doc:`../examples/embodied/maniskill` 。

限制
----

同一个 ``envs`` 下的所有环境配置共用一个 policy 和一个 rollout 循环，由此带来以下限制：

- policy 只有一种动作空间和一套动作归一化，也无法得知某个样本来自哪份环境配置。
- ``policy_obs`` 只缩放图像，因此每份环境配置的 ``states`` 维度 ``D`` 必须相同。
- 每份环境配置都必须分到整数个 env worker，计算方法见 `为环境配置分配 env worker`_ 。

下列功能要求 ``env.train`` 或 ``env.eval`` 是同构的，或其中只有一类环境，否则会在启动时报错：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 功能
     - 原因
   * - 平滑干预（ ``smooth_intervene`` ）
     - 它的动作绕过 policy，无法与其他环境配置的 policy 动作放进同一个 rollout batch。
   * - 实时控制评估（ ``runner.rtc`` ）
     - RTC rollout 根据同构 ``env.eval`` 的 ``env_type`` 选择延迟模型。
   * - RLT 仿真 replay
     - actor 只维护一种 replay 布局，因此 ``env.train`` 中的环境配置要么全部是 ``maniskill_rlt`` ，要么都不是。
