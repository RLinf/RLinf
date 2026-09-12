DynaRL: Flexible and Dynamic Scheduling of Large-Scale Reinforcement Learning Training
=======================================================================================

**Paper:** `OSDI 2026 <https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing>`__ | `PDF <https://www.usenix.org/system/files/osdi26-wang-yuanqing.pdf>`__

Overview
--------

.. image:: https://github.com/RLinf/misc/raw/main/pic/dynarl/overview.png
   :alt: DynaRL system architecture
   :align: center

Modern RL workloads for large language models, long-horizon reasoning, and
agentic systems are highly dynamic: heavy-tailed rollouts, irregular multi-turn
tool interactions, and time-varying bottlenecks leave static GPU partitions idle
and prolong training.

**DynaRL** is the first RL system that dynamically reallocates computation,
memory, and communication across heterogeneous RL components. It models the
entire RL pipeline as a dynamic hypergraph that serves as a centralized,
continuously evolving control surface. With a unified resource-migration
interface and context-aware data routing, the scheduler moves GPUs from
overprovisioned components to the current bottleneck through multi-level
scheduling and fine-grained migration.

DynaRL is implemented atop RLinf and integrates with existing inference engines
(e.g., SGLang) and training backends (e.g., Megatron-LM). It is the system behind
RLinf's :doc:`dynamic scheduling <../../examples/agentic/dynamic_scheduling>` feature.

Results
-------

Evaluations on math-reasoning and multi-turn agentic RL (models from 1.5B to
32B) show:

- **Math-reasoning RL:** up to **1.98×** end-to-end throughput over state-of-the-art
  systems (verl, RLinf) across cluster scales.
- **Agentic RL:** **1.06×–1.53×** higher throughput than RLinf; with
  priority-aware request scheduling, **1.27×–1.64×**.
- **Overhead:** scheduling plans within 200 ms; each reallocation in 0.5–5 s;
  online scheduling overhead below **1%**.

Math-reasoning RL
~~~~~~~~~~~~~~~~~

Math-reasoning RL comprises rollout, inference, and training stages. Baseline
systems use static sequential allocation (all GPUs for one stage at a time).
DynaRL detects underutilized stages and reallocates capacity to the active
bottleneck.

.. image:: https://github.com/RLinf/misc/raw/main/pic/dynarl/math_throughput.png
   :alt: End-to-end throughput of math-reasoning RL
   :align: center

End-to-end throughput of math-reasoning RL under different cluster scales and
model sizes.

.. list-table:: Speedup vs. static baselines (math-reasoning RL)
   :header-rows: 1
   :widths: 18 28 28 26
   :align: left

   * - Setting
     - vs. verl / RLinf (64 GPUs)
     - vs. verl / RLinf (128 GPUs)
     - Notes
   * - 1.5B / 7B
     - 1.43×–1.55×
     - 1.40×–1.52×
     - Removes rollout underutilization
   * - 32B
     - 1.27× over RLinf
     - 1.98× / 1.40×
     - verl OOMs on 64 GPUs
   * - vs. RLHFuse
     - 1.21×–1.42×
     - 1.21×–1.42×
     - Schedules all three stages jointly

Multi-turn agentic RL
~~~~~~~~~~~~~~~~~~~~~

Agentic RL has multi-turn rollouts with tool calls, which make load less
predictable. DynaRL combines dynamic GPU allocation with priority-aware request
scheduling (preferring requests with more completed tool calls to reduce
stragglers and improve KV-cache reuse).

.. image:: https://github.com/RLinf/misc/raw/main/pic/dynarl/agent_throughput.png
   :alt: End-to-end throughput of multi-turn agentic RL
   :align: center

End-to-end throughput of multi-turn agentic RL under different cluster scales
and model sizes.

.. list-table:: Speedup vs. RLinf (multi-turn agentic RL)
   :header-rows: 1
   :widths: 20 35 45
   :align: left

   * - Setting
     - Dynamic allocation only
     - + Priority-aware request scheduling
   * - 64 GPUs (1.5B / 7B)
     - 1.06×–1.38×
     - 1.51×–1.53×
   * - 64 GPUs (32B)
     - —
     - 1.27×
   * - 128 GPUs
     - —
     - 1.40× / 1.64× / 1.58× (1.5B / 7B / 32B)

GPU allocation timeline
~~~~~~~~~~~~~~~~~~~~~~~

The figures below show how 64 GPUs are assigned to Trainer, Rollout, and
Inference over time under three modes: static allocation, dynamic allocation,
and dynamic allocation with priority-aware request scheduling.

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/allocation_static.png" alt="Static allocation" width="700"/>
         <br/><strong>Static allocation</strong>
       </td>
     </tr>
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/allocation_dynamic.png" alt="Dynamic allocation" width="700"/>
         <br/><strong>Dynamic allocation</strong>
       </td>
     </tr>
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/allocation_priority.png" alt="Dynamic allocation with priority-aware scheduling" width="700"/>
         <br/><strong>Dynamic + priority-aware request scheduling</strong>
       </td>
     </tr>
   </table>
   </div>

Scheduling overhead
~~~~~~~~~~~~~~~~~~~

Trainer migration cost scales mainly with model size (sub-ms for 1.5B to a few
seconds for 32B) and stays under **0.5%** of end-to-end latency even for 32B on
128 GPUs. Rollout migration completes within a few seconds. Scheduling decisions
finish within **200 ms**; total scheduling overhead per iteration is below
**0.5%**.

.. raw:: html

   <div align="center">
   <table border="0">
     <tr>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/migration_trainer.png" alt="Trainer migration cost" width="320"/>
         <br/><strong>Trainer migration cost</strong>
       </td>
       <td align="center">
         <img src="https://github.com/RLinf/misc/raw/main/pic/dynarl/migration_rollout.png" alt="Rollout migration cost" width="320"/>
         <br/><strong>Rollout migration cost</strong>
       </td>
     </tr>
   </table>
   </div>

Quick Start
-----------

- **Instruction:** :doc:`../../examples/agentic/dynamic_scheduling`

Citation
--------

.. code-block:: bibtex

   @inproceedings{wang2026dynarl,
     author = {Yuanqing Wang and Hao Lin and Junhao Hu and Chunyang Zhu and Quanlu Zhang and Zhen Guo and Yuchen Zhang and Xu Fu and Si Xu and Bo Dai and Zixiao Huang and Chao Yu and Boxun Li and Guohao Dai and Zhi Yang and Yu Wang},
     title = {{DynaRL}: Flexible and Dynamic Scheduling of {Large-Scale} Reinforcement Learning Training},
     booktitle = {20th USENIX Symposium on Operating Systems Design and Implementation (OSDI 26)},
     year = {2026},
     isbn = {978-1-939133-55-7},
     address = {Seattle, WA},
     pages = {847--862},
     url = {https://www.usenix.org/conference/osdi26/presentation/wang-yuanqing},
     publisher = {USENIX Association},
     month = jul
   }
