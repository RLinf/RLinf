WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning
===========================================================================================================

.. .. centered::
..    Zelai Xu :sup:`1*`, Zhexuan Xu :sup:`1*`, Ruize Zhang :sup:`2*`, Chunyang Zhu :sup:`3`, Shi Yu :sup:`4`

.. .. centered::
..    Weilin Liu :sup:`3`, Quanlu Zhang :sup:`3`, Wenbo Ding :sup:`2`, Chao Yu :sup:`2†`, Yu Wang :sup:`1†`

.. .. centered::
..    :sup:`1` EE, Tsinghua University   :sup:`2` SIGS, Tsinghua University   :sup:`3` Infinigence AI   :sup:`4` IIIS, Tsinghua University

.. .. centered::
..    :sup:`*` Equal Contribution   :sup:`†` Corresponding Authors

.. raw:: html

   <div style="text-align:center">
     <div class="authors">
        <p style="font-size: 1.25em; line-height: 1.4; margin: 0;">
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/ZelaiXu" target="_blank">Zelai Xu</a><sup>1*</sup>,
        Zhexuan Xu<sup>1*</sup>,
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/RuizeZhang" target="_blank">Ruize Zhang</a><sup>2*</sup>,
        Chunyang Zhu<sup>3</sup>,
        <a href="https://yumiao20071126.github.io" target="_blank">Shi Yu</a><sup>4</sup>,
        <br>
        Weilin Liu<sup>3</sup>,
        Quanlu Zhang<sup>3</sup>,
        <a href="https://ssr-group.net" target="_blank">Wenbo Ding</a><sup>2</sup>,
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/ChaoYu" target="_blank">Chao Yu</a><sup>2&dagger;</sup>,
        <a href="https://nicsefc.ee.tsinghua.edu.cn/people/YuWang" target="_blank">Yu Wang</a><sup>1&dagger;</sup>
        </p>
        <p class="affiliations" style="margin-top: 16px; margin-bottom: 0;">
            <sup>1</sup>EE, Tsinghua University &nbsp;&nbsp;
            <sup>2</sup>SIGS, Tsinghua University &nbsp;&nbsp;
            <sup>3</sup>Infinigence AI &nbsp;&nbsp;
            <sup>4</sup>IIIS, Tsinghua University
        </p>
       <p class="affiliations" style="color: #333;">
         <sup>*</sup>Equal Contribution. &nbsp;&nbsp;<sup>&dagger;</sup>Corresponding Authors.
       </p>
     </div>
   </div>

**Paper:** `arXiv:2602.04634 <https://arxiv.org/abs/2602.04634>`__  

**Code:** `RLinf Example (WideSeek-R1) <https://github.com/RLinf/RLinf/tree/main/examples/wideseek_r1>`__ 

**Dataset:** `WideSeek-R1-train-data <https://huggingface.co/datasets/RLinf/WideSeek-R1-train-data>`__  

**Model:** `WideSeek-R1-4B <https://huggingface.co/RLinf/WideSeek-R1-4b>`__




Abstract
--------

Recent advancements in Large Language Models (LLMs) have largely focused on depth scaling,
where a single agent solves long-horizon problems with multi-turn reasoning and tool use. However, as tasks
grow broader, the key bottleneck shifts from individual competence to organizational capability. In this work,
we explore a complementary dimension of **width scaling** with multi-agent systems to address
broad information seeking. Existing multi-agent systems often rely on hand-crafted workflows and turn-taking
interactions that fail to parallelize work effectively. To bridge this gap, we propose
**WideSeek-R1**, a lead-agent&ndash;subagent framework trained via
**multi-agent reinforcement learning (MARL)** to synergize scalable orchestration and parallel
execution. By utilizing a shared LLM with isolated contexts and specialized tools, WideSeek-R1 jointly
optimizes the lead agent and parallel subagents on a curated dataset of
20k broad information-seeking tasks. Extensive experiments show that
WideSeek-R1-4B achieves an item F1 score of 40.0% on the WideSearch benchmark, which is
comparable to the performance of single-agent DeepSeek-R1-671B. Furthermore,
WideSeek-R1-4B exhibits consistent performance gains as the number of parallel subagents increases,
highlighting the effectiveness of width scaling.

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/scaling.png
   :alt: Depth vs width scaling comparison
   :align: center

*Figure 1: Comparison of depth and width scaling. While depth scaling enhances performance through sequential multi-turn interactions, width scaling orchestrates multi-agent systems for parallel execution. WideSeek-R1 pushes the frontier of width scaling via MARL for synergized orchestration and execution.*

Contributions
-------------

- We introduce **WideSeek-R1**, a multi-agent system trained via MARL to synergize scalable orchestration and parallel execution for broad information seeking.
- We open-source **a large-scale dataset** of 20,000 broad information-seeking tasks, offering a complementary training resource to existing multi-hop datasets.
- We demonstrate the effectiveness of **width scaling** with WideSeek-R1-4B, which achieves comparable performance to DeepSeek-R1-671B and exhibits consistent gains as the number of parallel agents increases.


Motivation
----------

As tasks grow broader, width scaling via multi-agent systems becomes essential, yet
both single-agent methods and existing multi-agent systems fall short in different ways.
Broad information seeking, which requires gathering and synthesizing attributes of multiple entities
into a structured table, serves as an ideal testbed for this challenge.

Limitations of Single-Agent Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Single-agent methods face two fundamental limitations when tasks grow in breadth.

- **Context pollution.** As the agent's context accumulates information from previous subtasks, irrelevant content increasingly interferes with reasoning, degrading performance on later subtasks.
- **Sequential execution.** A single agent must process independent subtasks one by one, leaving parallelizable work serialized and making the overall process inefficient.

These limitations underscore the necessity of multi-agent systems, which naturally enable context isolation
and parallel execution for effective width scaling.

Limitations of Existing Multi-Agent Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Despite their promise, existing multi-agent systems have not fully realized the potential of width
scaling, primarily because few systems are trained end-to-end to learn scalable orchestration and
parallel execution.

- **Hand-crafted orchestration.** Most prior work relies on manually designed workflows rather than learned agents, hindering flexible and scalable coordination as the number of agents grows.
- **Turn-taking execution.** Current systems typically process subtasks one at a time through turn-taking interactions, serializing progress and failing to parallelize independent work.

As a result, the performance of existing multi-agent systems is bottlenecked by limited scalability and
insufficient parallelization. WideSeek-R1 is designed to address both levels through
end-to-end multi-agent reinforcement learning.

Method
------

WideSeek-R1 is a hierarchical lead-agent-subagent system trained via **end-to-end MARL
to synergize scalable orchestration and parallel execution** for width scaling.
The lead agent and subagents share a single LLM but operate with isolated contexts and specialized tools:
the lead agent focuses on task decomposition and orchestration, while each subagent executes its assigned
subtask in parallel using external tools to gather information and return findings.

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/overview.png
   :alt: WideSeek-R1 rollout and training pipeline
   :align: center

*Figure 2. Overview of WideSeek-R1 rollout and training pipeline. Rollout: The lead agent coordinates task decomposition while subagents execute parallel subtasks. Training: A shared model is trained via GRPO with multi-agent advantage assignment and dual-level advantage reweighting.*

Lead Agent for Scalable Orchestration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The lead agent decomposes a broad task into parallelizable subtasks and delegates them to
subagents. Unlike systems with fixed hand-crafted workflows, the lead agent is trained for
**scalable and learnable orchestration** as the number of subagents increases. The lead agent is
restricted to a ``call_subagent``-style tool to reduce context pollution.

Subagents for Parallel Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subagents are responsible for parallel information seeking, enabling width scaling by executing
multiple subtasks simultaneously. This design addresses both context pollution and sequential
execution bottlenecks in single-agent pipelines. Subagents use two tools:

- ``search``: retrieves relevant snippets and URLs for a query.
- ``access``: summarizes information from a specific URL.

Multi-Agent Reinforcement Learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

WideSeek-R1 jointly optimizes the lead agent and subagents with a shared model, enabling
simultaneous learning of orchestration and information-seeking behavior. The method extends GRPO
for multi-agent systems with two key designs:

- **Multi-Agent Advantage Assignment:** use a verifiable outcome reward per multi-agent rollout and assign the same advantage to all agents and all tokens for stable training.
- **Dual-Level Advantage Reweighting:** use both agent-level and token-level reweighting in the policy gradient objective for multi-agent multi-turn optimization.

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

Training Data Construction
--------------------------

To unlock width scaling, WideSeek-R1 requires large quantities of broad information-seeking
tasks. We build a fully automated data construction pipeline that synthesizes high-quality training
instances with schema-constrained queries and standardized tabular outputs, yielding **20,000**
instances.

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/data_pipeline.png
   :alt: Automated data construction pipeline
   :align: center

*Figure 3. Automated data construction pipeline with three stages: Query Generation, Answer Generation, and QA Pair Filtering.*

The pipeline includes three stages:

1. **Query Generation:** extract user intents from HybridQA and refine them into complex schema-constrained queries with explicit table requirements.
2. **Answer Generation:** generate two independent responses and identify unique columns for self-consistency verification.
3. **QA Pair Filtering:** remove low-consistency or low-difficulty samples to keep robust and challenging instances.

Experiment Results
------------------

Main Results on WideSearch
~~~~~~~~~~~~~~~~~~~~~~~~~~

WideSeek-R1-4B achieves the best results on five out of six metrics among 4B and 8B baselines.
The multi-agent system consistently outperforms the single-agent variant, with an absolute
improvement of 11.9 points in item F1 score, and attains an 8.8-point gain over the base
Qwen3-4B in the same multi-agent setting. Notably, WideSeek-R1-4B is comparable to
single-agent DeepSeek-R1-671B with nearly 170x fewer parameters.

.. list-table:: Table 1. Results on the WideSearch benchmark (100 English + 100 Chinese tasks requiring tabular output)
   :header-rows: 1
   :widths: 12 22 11 11 11 11 11 11
   :align: left

   * - Setting
     - Model
     - Item F1 Avg@4
     - Item F1 Max@4
     - Row F1 Avg@4
     - Row F1 Max@4
     - Success Avg@4
     - Success Pass@4
   * - Single Agent
     - SingleSeek-R1-4B
     - 28.1
     - 39.2
     - 6.5
     - 12.5
     - 0.3
     - 1.0
   * - Single Agent
     - Qwen3-4B
     - 20.1
     - 30.2
     - 3.0
     - 4.8
     - 0.0
     - 0.0
   * - Single Agent
     - Search-R1-7B
     - 15.5
     - 24.4
     - 2.0
     - 4.4
     - 0.0
     - 0.0
   * - Single Agent
     - ASearcher-7B
     - 16.5
     - 26.0
     - 2.8
     - 5.8
     - 0.0
     - 0.0
   * - Single Agent
     - DeepSeek-R1-671B
     - 41.3
     - 55.1
     - 20.7
     - 31.7
     - 0.4
     - 1.5
   * - Multi-Agent System
     - **WideSeek-R1-4B**
     - **40.0**
     - **51.8**
     - **15.3**
     - **24.4**
     - **0.4**
     - **1.0**
   * - Multi-Agent System
     - Qwen3-4B
     - 31.2
     - 42.3
     - 8.4
     - 15.5
     - 0.0
     - 0.0
   * - Multi-Agent System
     - AgentFlow-7B
     - 28.7
     - 45.4
     - 9.0
     - 20.2
     - 0.4
     - 1.5
   * - Multi-Agent System
     - OWL-8B
     - 20.2
     - 29.3
     - 3.1
     - 5.8
     - 0.0
     - 0.0
   * - Multi-Agent System
     - MiroFlow-8B
     - 23.7
     - 37.7
     - 5.8
     - 12.7
     - 0.4
     - 1.0

Exploring Width Scaling
~~~~~~~~~~~~~~~~~~~~~~~

To compare depth scaling and width scaling under test-time compute, we plot performance as a
function of the number of turns (depth) and the number of agents (width).

- **Depth Scaling:** performance initially improves with more turns, then quickly plateaus due to fixed context limits.
- **Width Scaling:** performance initially improves with more subagents but can decline at larger width when an untrained lead agent receives conflicting signals.
- **Width Scaling + MARL:** performance improves consistently as subagents increase, reaching 40.0% item F1 with 10 subagents.

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/exp_scaling.png
   :alt: Depth scaling versus width scaling
   :align: center

*Figure 4. Comparison of depth and width scaling with respect to test-time compute. The blue curve is depth scaling vs turns; the red curves are width scaling vs number of subagents.*

Ablation Studies
~~~~~~~~~~~~~~~~

We conduct ablations to answer two questions:
(1) whether jointly optimizing both the lead agent and subagents is necessary,
and (2) how data composition affects final capability.

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/ablation_agent.png
   :alt: Ablation on lead agent and subagents
   :align: center

*Figure 5. Ablation on lead agent and subagents.*

.. image:: https://github.com/RLinf/misc/raw/main/pic/wideseek_r1/ablation_data.png
   :alt: Ablation on training data composition
   :align: center

*Figure 6. Ablation on training data composition.*

- **Lead Agent and Subagents:** best performance is achieved when both roles use WideSeek-R1-4B. Upgrading either role alone gives comparable gains, and combining both yields additional improvements, confirming role synergy under end-to-end MARL.
- **Training Data:** a hybrid dataset (wide + deep) consistently outperforms wide-only or deep-only training at equal data size, indicating complementary benefits between orchestration-focused and execution-focused data.

Standard QA Benchmarks
~~~~~~~~~~~~~~~~~~~~~~

To evaluate generalization beyond broad information seeking, we test on seven open-domain QA
benchmarks (three single-hop and four multi-hop). WideSeek-R1-4B reaches an average score of
59.0%, outperforming its backbone multi-agent Qwen3-4B by 7.7 points and surpassing larger
multi-agent systems such as OWL-8B and MiroFlow-8B.

.. list-table:: Table 2. Results on standard QA benchmarks (3 single-hop + 4 multi-hop)
   :header-rows: 1
   :widths: 12 18 8 8 10 8 8 10 10 8
   :align: left

   * - Setting
     - Model
     - Avg.
     - NQ
     - TriviaQA
     - PopQA
     - 2Wiki
     - HotpotQA
     - Bamboogle
     - MuSiQue
   * - Single Agent
     - SingleSeek-R1-4B
     - 57.0
     - 58.8
     - 78.3
     - 48.0
     - 70.9
     - 62.1
     - 54.6
     - 26.5
   * - Single Agent
     - Qwen3-4B
     - 48.3
     - 48.5
     - 68.7
     - 43.0
     - 58.9
     - 51.4
     - 48.2
     - 19.2
   * - Single Agent
     - Search-R1-7B
     - 55.4
     - 49.9
     - 78.0
     - 55.7
     - 58.1
     - 60.8
     - 58.4
     - 27.1
   * - Single Agent
     - ASearcher-7B
     - 61.0
     - 54.5
     - 79.3
     - 55.9
     - 77.6
     - 67.6
     - 60.0
     - 32.6
   * - Multi-Agent System
     - **WideSeek-R1-4B**
     - **59.0**
     - **56.1**
     - **78.5**
     - **48.5**
     - **75.0**
     - **64.2**
     - **61.8**
     - **28.9**
   * - Multi-Agent System
     - Qwen3-4B
     - 51.3
     - 49.6
     - 70.7
     - 44.9
     - 65.0
     - 54.3
     - 52.6
     - 21.7
   * - Multi-Agent System
     - AgentFlow-7B
     - 61.0
     - 58.5
     - 87.0
     - 52.5
     - 77.2
     - 57.0
     - 69.6
     - 25.3
   * - Multi-Agent System
     - OWL-8B
     - 57.2
     - 64.0
     - 74.2
     - 52.2
     - 62.6
     - 61.0
     - 55.8
     - 30.4
   * - Multi-Agent System
     - MiroFlow-8B
     - 50.0
     - 50.9
     - 73.1
     - 42.8
     - 58.6
     - 52.4
     - 50.8
     - 21.3

Citation
--------

.. code-block:: bibtex

   @article{xu2026wideseek,
     title={WideSeek-R1: Exploring Width Scaling for Broad Information Seeking via Multi-Agent Reinforcement Learning},
     author={Xu, Zelai and Xu, Zhexuan and Zhang, Ruize and Zhu, Chunyang and Yu, Shi and Liu, Weilin and Zhang, Quanlu and Ding, Wenbo and Yu, Chao and Wang, Yu},
     journal={arXiv preprint arXiv:2602.04634},
     year={2026},
   }

