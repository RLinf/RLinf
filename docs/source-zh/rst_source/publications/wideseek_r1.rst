WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning
===========================================================================================================

.. raw:: html

   <div style="text-align:center">
     <div class="authors">
        <p style="font-size: 1.25em; line-height: 1.4; margin: 0;">
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/ZelaiXu" target="_blank">徐泽来</a><sup>1*</sup>,
        徐哲轩<sup>1*</sup>,
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/RuizeZhang" target="_blank">张瑞泽</a><sup>2*</sup>,
        朱春阳<sup>3</sup>,
        <a href="https://yumiao20071126.github.io" target="_blank">余势</a><sup>4</sup>,
        <br>
        刘巍林<sup>3</sup>,
        张权路<sup>3</sup>,
        <a href="https://ssr-group.net" target="_blank">丁文伯</a><sup>2</sup>,
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/ChaoYu" target="_blank">于超</a><sup>2&dagger;</sup>,
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/YuWang" target="_blank">汪玉</a><sup>1&dagger;</sup>
        </p>
        <p class="affiliations" style="margin-top: 16px; margin-bottom: 0;">
            <sup>1</sup>清华大学电子工程系 &nbsp;&nbsp;
            <sup>2</sup>清华大学深圳国际研究生院 &nbsp;&nbsp;
            <sup>3</sup>无问芯穹 &nbsp;&nbsp;
            <sup>4</sup>清华大学交叉信息研究院
        </p>
       <p class="affiliations" style="color: #333;">
         <sup>*</sup>共同第一作者。 &nbsp;&nbsp;<sup>&dagger;</sup>通讯作者。
       </p>
     </div>
   </div>

**论文：** `arXiv:2602.04634 <https://arxiv.org/abs/2602.04634>`__  

**代码：** `WideSeek-R1目录 <https://github.com/RLinf/RLinf/tree/main/examples/wideseek_r1>`__ 

**数据集：** `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__  

**模型：** `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__ 

**快速开始：** :doc:`快速入门指南 <../examples/agentic/wideseek_r1/index>`


摘要
----

近年来，大语言模型（LLM）的进展主要聚焦于深度扩展，即由单个智能体通过多轮推理与工具使用来解决长时程问题。
然而，随着任务的广度不断增加，关键瓶颈正从个体能力转向组织能力。
在这项工作中，我们探索了一个互补维度，即利用多智能体系统进行**宽度扩展**，以解决广域信息检索问题。
现有多智能体系统通常依赖手工设计的工作流和轮流交互机制，无法有效实现并行化。
为弥合这一差距，我们提出了 **WideSeek-R1**，这是一个通过 **多智能体强化学习（MARL）** 训练得到的主智能体-子智能体框架，用于协同实现可扩展的编排与并行执行。
WideSeek-R1 采用共享的 LLM、隔离的上下文以及专用工具，在一个精心构建的 2 万条广域信息检索任务数据集上，对主智能体和并行子智能体进行联合优化。
大量实验表明，WideSeek-R1-4B 在 WideSearch 基准上取得了 40.0% 的 item F1 分数，可与单智能体 DeepSeek-R1-671B 的表现相当。
此外，随着并行子智能体数量的增加，WideSeek-R1-4B 持续获得性能提升，凸显了宽度扩展的有效性。

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/scaling.png
   :alt: 深度扩展与宽度扩展对比
   :align: center

*图 1：深度扩展与宽度扩展的对比。深度扩展通过串行的多轮交互提升性能，而宽度扩展则通过编排多智能体系统实现并行执行。WideSeek-R1 通过 MARL 推动了宽度扩展的前沿，使编排与执行能够协同优化。*

贡献
----

- 我们提出了 **WideSeek-R1**，这是一个通过 MARL 训练的多智能体系统，能够为广域信息检索协同实现可扩展编排与并行执行。
- 我们开源了一个包含 20,000 个广域信息检索任务的 **大规模数据集**，为现有多跳数据集提供了互补的训练资源。
- 我们验证了 **宽度扩展** 在 WideSeek-R1-4B 上的有效性：它不仅取得了与 DeepSeek-R1-671B 相当的性能，还随着并行智能体数量的增加持续获得收益。


动机
----

随着任务的广度不断增加，基于多智能体系统的宽度扩展变得至关重要；然而，单智能体方法与现有多智能体系统都在不同方面存在不足。广域信息检索需要收集并整合多个实体的属性，并将其组织成结构化表格，因此非常适合作为这一挑战的测试平台。

单智能体方法的局限性
~~~~~~~~~~~~~~~~~~~~

当任务在广度上扩展时，单智能体方法面临两个根本性限制。

- **上下文污染。** 随着智能体上下文不断累积先前子任务的信息，无关内容会越来越多地干扰推理，从而降低后续子任务的性能。
- **串行执行。** 单个智能体必须逐个处理彼此独立的子任务，导致本可并行的工作被串行化，整体过程低效。

这些局限性凸显了多智能体系统的必要性，因为多智能体天然能够实现上下文隔离和并行执行，从而支持有效的宽度扩展。

现有多智能体系统的局限性
~~~~~~~~~~~~~~~~~~~~~~~~

尽管前景可观，现有多智能体系统仍未能充分释放宽度扩展的潜力，主要原因在于几乎没有系统能够通过端到端训练学习可扩展的编排与并行执行能力。

- **手工设计的编排。** 既有工作大多依赖人工设计的工作流，而非通过学习得到的智能体，因此随着智能体数量增长，难以实现灵活且可扩展的协同。
- **轮流式执行。** 当前系统通常通过轮流交互一次只处理一个子任务，使整体进度被串行化，无法对独立工作进行并行化。

因此，现有多智能体系统的性能受限于较差的可扩展性和不足的并行化能力。WideSeek-R1 正是为通过端到端多智能体强化学习同时解决这两个层面的问题而设计的。

方法
----

WideSeek-R1 是一个层级式的主智能体-子智能体系统，通过 **端到端 MARL 训练，以协同实现可扩展编排与并行执行**，从而支持宽度扩展。主智能体和子智能体共享同一个 LLM，但拥有彼此隔离的上下文以及专用工具：主智能体负责任务分解与编排，而每个子智能体则使用外部工具并行执行分配到的子任务，收集信息并返回结果。

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/overview.png
   :alt: WideSeek-R1 的 rollout 与训练流程
   :align: center

*图 2. WideSeek-R1 的 rollout 与训练流程概览。Rollout 阶段：主智能体负责任务分解，子智能体并行执行子任务。训练阶段：共享模型通过 GRPO 进行训练，并采用多智能体优势分配与双层优势重加权。*

用于可扩展编排的主智能体
~~~~~~~~~~~~~~~~~~~~~~~~

主智能体将一个广域任务分解为可并行执行的子任务，并将其委派给子智能体。不同于使用固定手工工作流的系统，主智能体被训练为能够在子智能体数量增加时实现 **可扩展且可学习的编排**。为减少上下文污染，主智能体被限制只能使用类似 ``call_subagent`` 的工具。

用于并行执行的子智能体
~~~~~~~~~~~~~~~~~~~~~~

子智能体负责并行的信息检索，通过同时执行多个子任务来实现宽度扩展。这一设计同时缓解了单智能体流程中的上下文污染与串行执行瓶颈。子智能体使用两种工具：

- ``search``：为查询检索相关片段和 URL。
- ``access``：对指定 URL 中的信息进行摘要。

多智能体强化学习
~~~~~~~~~~~~~~~~

WideSeek-R1 通过共享模型对主智能体和子智能体进行联合优化，从而同时学习编排行为和信息检索行为。该方法将 GRPO 扩展到多智能体系统，并包含两个关键设计：

- **多智能体优势分配：** 对每次多智能体 rollout 使用可验证的结果奖励，并将相同的优势分配给所有智能体和所有 token，以获得稳定训练。
- **双层优势重加权：** 在策略梯度目标中同时采用智能体级和 token 级重加权，以支持多智能体多轮优化。

.. math::

   \mathbb{E}\left[
   \frac{1}{G}\sum_{i=1}^{G}
   \frac{1}{N_i}\sum_{a=1}^{N_i}
   \frac{1}{\sum_{t=1}^{T_{i,a}} |o^t_{i,a}|}
   \sum_{t=1}^{T_{i,a}}\sum_{j=1}^{|o^t_{i,a}|}
   \min\left(
   r^{t,j}_{i,a}\hat{A}_i,
   \operatorname{clip}\left(r^{t,j}_{i,a},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}}\right)\hat{A}_i
   \right)
   \right]

训练数据构建
------------

为了释放宽度扩展的潜力，WideSeek-R1 需要大量广域信息检索任务。我们构建了一条全自动数据生成流程，通过带有模式约束的查询和标准化表格输出，合成高质量训练样本，最终得到 **20,000** 条样本。

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/data_pipeline.png
   :alt: 自动化数据构建流程
   :align: center

*图 3. 自动化数据构建流程包含三个阶段：查询生成、答案生成和问答对筛选。*

该流程包含三个阶段：

1. **查询生成：** 从 HybridQA 中提取用户意图，并将其改写为带有明确表格要求的复杂模式约束查询。
2. **答案生成：** 生成两个相互独立的回答，并识别其中的独有列，以进行自一致性验证。
3. **问答对筛选：** 去除一致性低或难度低的样本，保留稳健且具有挑战性的样本。

实验结果
--------

WideSearch 上的主要结果
~~~~~~~~~~~~~~~~~~~~~~~

在 4B 和 8B 基线模型中，WideSeek-R1-4B 在六项指标中的五项上取得了最佳结果。该多智能体系统稳定优于单智能体变体，在 item F1 分数上绝对提升 11.9 个点，并且在相同多智能体设定下，相比基础模型 Qwen3-4B 提升了 8.8 个点。值得注意的是，WideSeek-R1-4B 仅用近 170 倍更少的参数量，就达到了与单智能体 DeepSeek-R1-671B 可比的表现。

.. list-table:: 表 1a. WideSearch 基准上的单智能体结果
   :header-rows: 1
   :widths: 22 11 11 11 11 11 11
   :align: left

   * - 模型
     - Item F1 Avg@4
     - Item F1 Max@4
     - Row F1 Avg@4
     - Row F1 Max@4
     - Success Avg@4
     - Success Pass@4
   * - SingleSeek-R1-4B
     - 28.1
     - 39.2
     - 6.5
     - 12.5
     - 0.3
     - 1.0
   * - Qwen3-4B
     - 20.1
     - 30.2
     - 3.0
     - 4.8
     - 0.0
     - 0.0
   * - Search-R1-7B
     - 15.5
     - 24.4
     - 2.0
     - 4.4
     - 0.0
     - 0.0
   * - ASearcher-7B
     - 16.5
     - 26.0
     - 2.8
     - 5.8
     - 0.0
     - 0.0
   * - DeepSeek-R1-671B
     - 41.3
     - 55.1
     - 20.7
     - 31.7
     - 0.4
     - 1.5

.. list-table:: 表 1b. WideSearch 基准上的多智能体结果
   :header-rows: 1
   :widths: 22 11 11 11 11 11 11
   :align: left

   * - 模型
     - Item F1 Avg@4
     - Item F1 Max@4
     - Row F1 Avg@4
     - Row F1 Max@4
     - Success Avg@4
     - Success Pass@4
   * - **WideSeek-R1-4B**
     - **40.0**
     - **51.8**
     - **15.3**
     - **24.4**
     - **0.4**
     - **1.0**
   * - Qwen3-4B
     - 31.2
     - 42.3
     - 8.4
     - 15.5
     - 0.0
     - 0.0
   * - AgentFlow-7B
     - 28.7
     - 45.4
     - 9.0
     - 20.2
     - 0.4
     - 1.5
   * - OWL-8B
     - 20.2
     - 29.3
     - 3.1
     - 5.8
     - 0.0
     - 0.0
   * - MiroFlow-8B
     - 23.7
     - 37.7
     - 5.8
     - 12.7
     - 0.4
     - 1.0

探索宽度扩展
~~~~~~~~~~~~

为了在测试时计算量的约束下比较深度扩展与宽度扩展，我们绘制了性能随轮次数（深度）和智能体数量（宽度）变化的曲线。

- **深度扩展：** 性能在轮次数增加时最初会提升，但由于固定上下文长度限制，很快趋于平台期。
- **宽度扩展：** 性能在子智能体数量增加时最初会提升，但当未训练的主智能体接收到相互冲突的信号时，在更大宽度下可能下降。
- **宽度扩展 + MARL：** 随着子智能体数量增加，性能持续提升，在 10 个子智能体时达到 40.0% 的 item F1。

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/exp_scaling.png
   :alt: 深度扩展与宽度扩展
   :align: center

*图 4. 在测试时计算量维度上，深度扩展与宽度扩展的对比。蓝色曲线表示深度扩展随轮次数的变化；红色曲线表示宽度扩展随子智能体数量的变化。*

消融实验
~~~~~~~~

我们进行了消融实验，以回答两个问题：
(1) 是否有必要同时联合优化主智能体与子智能体；
以及 (2) 数据组成如何影响最终能力。

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/ablation_agent.png
   :alt: 主智能体与子智能体的消融实验
   :align: center

*图 5. 主智能体与子智能体的消融实验。*

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/ablation_data.png
   :alt: 训练数据组成的消融实验
   :align: center

*图 6. 训练数据组成的消融实验。*

- **主智能体与子智能体：** 当两个角色都使用 WideSeek-R1-4B 时表现最佳。单独升级任一角色都能带来相近收益，而两者结合还能进一步提升，验证了端到端 MARL 下的角色协同效应。
- **训练数据：** 在相同数据规模下，混合数据集（wide + deep）始终优于仅宽度或仅深度训练，说明聚焦编排的数据与聚焦执行的数据之间具有互补收益。

标准 QA 基准
~~~~~~~~~~~~

为了评估其在广域信息检索之外的泛化能力，我们在七个开放域 QA 基准上进行了测试（三个单跳、四个多跳）。WideSeek-R1-4B 的平均分达到 59.0%，比其骨干多智能体 Qwen3-4B 高出 7.7 个点，也超过了 OWL-8B 和 MiroFlow-8B 等更大的多智能体系统。

.. list-table:: 表 2a. 标准 QA 基准上的单智能体结果（3 个单跳 + 4 个多跳）
   :header-rows: 1
   :widths: 18 8 8 10 8 8 10 10 8
   :align: left

   * - 模型
     - 平均
     - NQ
     - TriviaQA
     - PopQA
     - 2Wiki
     - HotpotQA
     - Bamboogle
     - MuSiQue
   * - SingleSeek-R1-4B
     - 57.0
     - 58.8
     - 78.3
     - 48.0
     - 70.9
     - 62.1
     - 54.6
     - 26.5
   * - Qwen3-4B
     - 48.3
     - 48.5
     - 68.7
     - 43.0
     - 58.9
     - 51.4
     - 48.2
     - 19.2
   * - Search-R1-7B
     - 55.4
     - 49.9
     - 78.0
     - 55.7
     - 58.1
     - 60.8
     - 58.4
     - 27.1
   * - ASearcher-7B
     - 61.0
     - 54.5
     - 79.3
     - 55.9
     - 77.6
     - 67.6
     - 60.0
     - 32.6

.. list-table:: 表 2b. 标准 QA 基准上的多智能体结果（3 个单跳 + 4 个多跳）
   :header-rows: 1
   :widths: 18 8 8 10 8 8 10 10 8
   :align: left

   * - 模型
     - 平均
     - NQ
     - TriviaQA
     - PopQA
     - 2Wiki
     - HotpotQA
     - Bamboogle
     - MuSiQue
   * - **WideSeek-R1-4B**
     - **59.0**
     - **56.1**
     - **78.5**
     - **48.5**
     - **75.0**
     - **64.2**
     - **61.8**
     - **28.9**
   * - Qwen3-4B
     - 51.3
     - 49.6
     - 70.7
     - 44.9
     - 65.0
     - 54.3
     - 52.6
     - 21.7
   * - AgentFlow-7B
     - 61.0
     - 58.5
     - 87.0
     - 52.5
     - 77.2
     - 57.0
     - 69.6
     - 25.3
   * - OWL-8B
     - 57.2
     - 64.0
     - 74.2
     - 52.2
     - 62.6
     - 61.0
     - 55.8
     - 30.4
   * - MiroFlow-8B
     - 50.0
     - 50.9
     - 73.1
     - 42.8
     - 58.6
     - 52.4
     - 50.8
     - 21.3

引用
----

.. code-block:: bibtex

   @article{xu2026wideseek,
     title={WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning},
     author={Xu, Zelai and Xu, Zhexuan and Zhang, Ruize and Zhu, Chunyang and Yu, Shi and Liu, Weilin and Zhang, Quanlu and Ding, Wenbo and Yu, Chao and Wang, Yu},
     journal={arXiv preprint arXiv:2602.04634},
     year={2026},
   }
