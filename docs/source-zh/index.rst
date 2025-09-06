.. image:: _static/svg/logo_white.svg
   :width: 500px
   :align: center
   :class: logo-svg

.. raw:: html

   <h1 style="text-align: center;">欢迎来到 <b>RLinf</b>！</h1>

RLinf 是一个灵活且可扩展的开源基础架构，专为通过强化学习对基础模型进行后训练而设计。名称中的 "inf" 代表 Infrastructure（基础架构），强调其作为新一代训练强大支撑系统的角色；同时也代表 Infinite（无限），象征该系统支持开放式学习、持续泛化和智能发展的无限可能性。

----------------

.. image:: _static/svg/overview.svg
   :width: 1000px
   :align: center
   :class: overview-svg

----------------

**RLinf 的独特之处在于：**

- 宏观到微观流程（Macro-to-Micro Flow）：一种新范式 M2Flow，通过微观级的执行流程完成宏观级的逻辑流程，**解耦逻辑工作流构建（可编程）与物理通信调度（高效执行）**。

- 灵活的执行模式

  - **共享式**：所有任务共享全部 GPU。
  - **分离式**：支持细粒度流水线。
  - **混合式**：可定制的混合部署，结合了共享式和分离式两种模式。

- 自动调度策略：根据训练任务自动选择最合适的执行模式，无需手动资源分配。

- 具身智能支持

  - 快速适配主流 VLA 模型：`OpenVLA`_, `OpenVLA-OFT`_, `π₀`_
  - 通过标准化 RL 接口支持主流基于 CPU 和 GPU 的模拟器：`ManiSkill3`_、`LIBERO`_
  - 支持 π₀ 模型族首次基于 flow-matching 动作专家进行的强化学习微调。

**RLinf 拥有出色的训练速度：**

- 结合细粒度流水线的混合式：相比其他框架，**吞吐率提升超过 120%**。
- 自动在线扩缩策略：训练资源动态扩展，GPU 切换只需数秒，**进一步提高效率 20–40%**，同时保持 RL 算法的 on-policy 特性。

**RLinf 同时兼具灵活性与易用性：**

- 多种后端集成支持

  - 统一接口可驱动两种互补的后端，无需修改代码即可无缝切换。
  - **FSDP + Hugging Face**：快速适配新模型与算法，适合初学者与快速原型开发。
  - **Megatron + SGLang**：优化大规模训练效率，适用于对性能要求极高的专家用户。

- 通过异步通信通道实现自适应通信

- 内建对多种主流强化学习方法的支持，包括 `PPO`_、`GRPO`_、`DAPO`_、`Reinforce++`_ 等。

.. _PPO: https://arxiv.org/abs/1707.06347
.. _GRPO: https://arxiv.org/abs/2402.03300
.. _DAPO: https://arxiv.org/abs/2503.14476
.. _Reinforce++: https://arxiv.org/abs/2501.03262



.. _OpenVLA: https://github.com/openvla/openvla
.. _OpenVLA-OFT: https://github.com/moojink/openvla-oft
.. _IsaacLab: https://github.com/isaac-sim/IsaacLab
.. _ManiSkill3: https://github.com/haosulab/ManiSkill
.. _LIBERO: https://github.com/Lifelong-Robot-Learning/LIBERO
.. _π₀: https://github.com/Physical-Intelligence/openpi
.. _Megatron-LM: https://github.com/NVIDIA/Megatron-LM
.. _SGLang: https://github.com/sgl-project/sglang
.. _vLLM: https://github.com/vllm-project/vllm



--------------------------------------------

.. toctree::
  :maxdepth: 2
  :includehidden:
  :titlesonly:

  rst_source/start/index

--------------------------------------------

.. toctree::
  :maxdepth: 3
  :includehidden:
  :titlesonly:

  rst_source/tutorials/index

--------------------------------------------

.. toctree::
  :maxdepth: 2
  :includehidden:
  :titlesonly:

  rst_source/examples/index

--------------------------------------------

.. toctree::
  :maxdepth: 2
  :includehidden:
  :titlesonly:

  rst_source/blog/index

--------------------------------------------

.. toctree::
  :maxdepth: 2
  :includehidden:
  :titlesonly:

  rst_source/apis/index

--------------------------------------------

.. toctree::
  :maxdepth: 1
  :includehidden:
  :titlesonly:

  rst_source/faq

--------------------------------------------

.. _contribution-guidelines:

贡献指南
~~~~~~~~~~~~~~~~~~~~~~~~~

太棒了！我们一直欢迎新的贡献者加入我们的代码库。

首先，即使你不确定或者担心出错，也请大胆提 issue 或提交 pull request。我们不会因为你的努力尝试而批评你。最坏的情况也不过是我们礼貌地建议你修改一些内容。我们欢迎各种形式的贡献，不希望一堆规则阻碍你的参与。

当然，如果你希望了解更明确的贡献方式，可以继续阅读本指南。它将介绍我们对贡献的基本要求，帮助你更快通过审核或得到反馈。

以下是你在提交代码前需要遵循的一些简单指南。

**代码格式检查与规范**

我们使用 **pre-commit** 和 **ruff** 来统一代码风格并进行质量检查。安装方式如下：

.. code:: bash

   pip install pre-commit
   pip install ruff
   pre-commit install

修改代码后，在提交 PR 前建议运行以下命令：

.. code:: bash

   ruff check . --fix --preview

这将检查代码是否符合格式规范。  
如果存在如缩进错误等问题，它会自动进行修复。

**创建 Pull Request**

当你的代码通过了格式检查后，就可以提交 PR 了。请确保你的 commit 信息和分支命名符合规范，否则 CI 测试会拒绝合并。

- **提交信息规范**：参见 `规范化 Commits 约定 <https://www.conventionalcommits.org/en/v1.0.0/>`_。
- **分支命名规范**：参见 `规范化分支约定 <https://conventional-branch.github.io/#summary>`_。