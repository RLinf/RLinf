Accelerating the "ImageNet Moment" of Embodied AI: RLinf Brings a 25× System Optimization to BEHAVIOR
=====================================================================================================

Last updated: 05/20/2026.

Related reading: :doc:`Reinforcement Learning Training with the BEHAVIOR Benchmark <../examples/embodied/behavior>`.

If the explosive growth of large language models was fueled by the internet’s massive text corpora, then the key bottleneck for embodied AI to reach its own “ImageNet moment” lies in the efficiency of agent interaction inside physical simulation environments. Initiated by Prof. Fei-Fei Li, the **BEHAVIOR** benchmark covers more than a thousand realistic long-horizon household tasks and is an important proving ground for training general-purpose home robots. But under the standard high-fidelity setup, its environment interaction stack is extremely heavy: the default end-to-end rollout latency can reach **1028.7 ms/step**.

Focusing on the real dataflow of reinforcement learning rollouts, we performed system-level optimization on the execution path across **BEHAVIOR / OmniGibson / Isaac Sim**. Through feature slimming, on-demand observation, and a hybrid-partition pipeline parallel strategy, we ultimately reduced end-to-end latency to **41.2 ms/step**, achieving a **25×** speedup. These optimizations have already been merged into **RLinf** as the default configuration and have also been integrated upstream by `StanfordVL/BEHAVIOR-1K <https://github.com/StanfordVL/BEHAVIOR-1K>`_.

1. Background: Why Optimize the BEHAVIOR Simulator?
----------------------------------------------------

In embodied AI, the core challenge has gradually shifted from “can a robot arm complete a single action?” to “can an agent continuously complete an entire real-world task?” These tasks are typically long-horizon: they are not a single grasp or placement, but a sequence of interdependent substeps. For example, “clean the bedroom” sounds simple, but actually involves object recognition, target localization, navigation, grasping, obstacle avoidance, placement, and many other sequential decision-making steps.

Because task chains are longer and state transitions are more complex, the environment itself becomes critical infrastructure for the training system. Compared with lighter environments such as RoboTwin and LIBERO, BEHAVIOR provides more complete physical simulation, more complex and realistic scene structure, and benchmark scale covering more than a thousand household tasks. As a result, it has long been regarded as one of the most representative “embodied ImageNet” platforms in the field.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-splash.png
   :alt: BEHAVIOR simulator environment
   :align: center

*Figure 1: BEHAVIOR simulator environment.*

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-scenes.png
   :alt: BEHAVIOR simulator supported scenes
   :align: center

*Figure 2: BEHAVIOR simulator supported scenes.*

The BEHAVIOR simulator provides a high-fidelity environment for long-horizon tasks.

But high fidelity also comes at a substantial cost. On a machine with **AMD EPYC 7542 + NVIDIA RTX 4090D 24 GiB**, using **OpenPI pi-05** as an example, we found that under the default configuration:

- End-to-end rollout takes **1028.7 ms/step** on average, or about **0.97 step/s**;
- A single chunk-step forward pass in the model rollout stage takes about **0.4-0.5 s**; normalized to a per-step basis, this is about **12-16 ms**, which means the main throughput bottleneck is still the environment rather than the model;
- A single environment requires roughly **10 CPU cores** and **12 GiB of GPU memory**;
- Because it depends on the full graphics rendering stack, BEHAVIOR cannot be deployed on pure compute cards as easily as purely numerical workloads can. In practice, it usually needs graphics-capable GPUs such as the RTX 4090 rather than cards such as the A100.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-render-4090.jpeg
   :alt: Rendering result on RTX 4090
   :align: center

*Figure 3: Rendering result on RTX 4090.*

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-render-a100.jpeg
   :alt: Rendering result on A100
   :align: center

*Figure 4: Rendering result on A100.*

Comparison of rendering results on RTX 4090 and A100.

This means we cannot simply “open more environments” and expect linear throughput scaling: CPU, GPU memory, and graphics resources all rise together and quickly hit hardware limits. So the real problem is not just that “the environment is slow,” but that the default execution path was never designed for large-scale RL rollouts.

With that in mind, we tried to answer three questions:

1. **Where exactly is BEHAVIOR slow?** Is the bottleneck in the physics engine itself, or in the combined effect of rendering, observation, threads, and the parallel execution structure?
2. **Which costs are truly necessary?** Which ones are merely inherited from the default interactive simulator path and are effectively wasted work during RL rollouts?
3. **Without changing simulation semantics, can we reorganize BEHAVIOR’s execution model?** In other words, can we turn it into infrastructure that is genuinely suitable for large-scale training?

The conclusion of this section is that BEHAVIOR’s problem is not that it “cannot be parallelized,” but that it is designed by default for single-machine interactive use and is not suitable to be used directly for large-scale reinforcement learning rollouts.

2. Root Cause Analysis: BEHAVIOR Is Slow Because of System-Level Bottlenecks
-------------------------------------------------------------------------------

A single environment step in BEHAVIOR is not just a simple transition from :math:`s_t` to :math:`s_{t+1}`. Under the hood, Isaac Sim handles physical simulation, GPU rendering, and the Omniverse runtime, while OmniGibson wraps scenes, robots, sensors, task logic, and evaluation interfaces. As a result, one step actually spans the full pipeline of physics simulation, sensor rendering, observation packaging, reward computation, and termination checking.

We used **Tracy** to profile a single step at fine granularity and found that the end-to-end latency of 1028 ms is not caused by one isolated physics kernel, but by the accumulation of multiple system components.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-timeline-slow-en.png
   :alt: Timeline of a single BEHAVIOR action step
   :align: center

*Figure 5: Timeline of a single BEHAVIOR action step.*

Timeline for processing a single action in the BEHAVIOR environment.

2.1 Heavy Observation Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BEHAVIOR / OmniGibson supports rich multimodal observations by default, including RGB, depth, semantic segmentation, instance segmentation, and more. But during actual training, the policy usually consumes only a small subset of these signals. For example, with OpenPI pi-05, the model only needs **RGB images + proprioceptive state**, not the full image annotation pipeline.

In other words, the default observation system is much heavier than the model’s real input requirements. For rollouts, a large portion of observation generation does not translate into actual training benefit.

2.2 Display Paths Are Still Kept During Headless Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One design goal of Isaac Sim is to support interactive development, so it keeps an independent viewport for each robot camera, allowing developers to inspect images in real time, debug sensors, and check scene state. But during unattended rollout, what we actually need is model input, not a window being displayed on screen.

Both performance profiling and code-path analysis show that the default execution stack still allocates extra resources to this “display path.” It contributes almost nothing to training, yet continuously consumes GPU memory and rendering time.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-isaac-viewport.png
   :alt: Isaac Sim GUI and viewport illustration
   :align: center

*Figure 6: Isaac Sim GUI and viewport illustration.*

The Isaac Sim GUI, where viewports let developers directly inspect live images from each camera.

2.3 Observation Generation Frequency Does Not Match Policy Consumption Frequency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mainstream embodied models do not necessarily read an image and run a forward pass at every simulation step. A more common design is to operate on **action chunks**: the model generates multiple consecutive actions at once, for example 32 actions in one batch.

However, under the default execution mode, the simulator still performs full rendering, copy, and observation packaging at every step inside the action chunk. For the intermediate frames that never enter model inference, this work is fundamentally wasted.

2.4 BEHAVIOR’s Parallel Structure Is Not Naturally Efficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

BEHAVIOR can scale throughput with multiple processes, and it can also manage multiple environment instances inside one process through vectorized environments. But these two parallelization modes have very different efficiency characteristics:

- **Multiprocessing** can provide nearly linear throughput scaling, but CPU, GPU memory, and graphics resources also grow nearly linearly;
- **Vectorized environments** are more resource-efficient, but multiple environments still share the same global physics / rendering step, so the critical path still contains significant serial or semi-serial segments.

So the issue is not just “how many environments to run,” but at what granularity and in what way those environments are organized and advanced.

The conclusion of this section is that BEHAVIOR’s performance problem is fundamentally a system-level bottleneck caused jointly by rendering, observation, threading, and the parallel execution structure.

3. System-Level Optimization Strategy and Implementation: First Remove Wasted Work, Then Overlap Necessary Work
-----------------------------------------------------------------------------------------------------------------

Based on the analysis above, our goal was no longer to “make every module a little faster,” but to reorganize the entire execution process around the real dataflow of rollouts. The overall strategy can be summarized in two sentences:

1. **Remove wasted work**: if an execution path only makes sense for interactive visualization and brings no benefit to training, it should not remain on the critical path of headless rollout;
2. **Overlap necessary work**: for unavoidable costs such as physics simulation, rendering, and environment stepping, we should not only optimize them individually, but also pipeline them more effectively with model rollout.

Following these principles, we ended up with three main optimization tracks: **feature slimming**, **on-demand observation**, and **hybrid-partition pipeline parallelism**.

3.1 Simple but Effective Optimizations: Feature Slimming and On-Demand Observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first two optimizations share one common characteristic: they are straightforward and both come directly from performance analysis.

First is **feature slimming**. If headless rollout still keeps the display path, redundant observation modalities, and default threading strategies that are not suitable for multi-instance concurrency, the most direct approach is to cut away everything the training system does by default but does not actually need. This includes:

- Disabling unnecessary Omniverse viewports during headless training, while keeping only the rendering path that is truly needed for sensor output;
- Turning off depth, semantic segmentation, and other multimodal observations that are not consumed by the model during rollout;
- Limiting the default thread count and enabling supporting optimizations such as numba JIT cache, so the system runs from the start like a “high-throughput training environment” rather than an “interactive debugging simulator.”

Second is **on-demand observation**. Since the policy consumes observations at action-chunk boundaries, the intermediate frames inside the action chunk that never reach the model should not trigger the full observation pipeline. Therefore, within each action chunk we still preserve:

- Physics simulation;
- Reward accumulation;
- Termination checking;

but generate observations only at the end of the chunk, when the policy truly needs new input. This aligns observation generation frequency with policy consumption frequency and directly removes a large amount of extra overhead from the critical path.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-timeline-fast-en.png
   :alt: Action-chunk timeline after on-demand observation
   :align: center

*Figure 7: Action-chunk timeline after on-demand observation.*

After trimming, the BEHAVIOR environment processes one action chunk with only the last action rendered, while all previous actions perform physics simulation only.

These first two optimizations are essentially “subtractions”: remove redundancy from the default path first, then move on to deeper execution reorganization.

3.2 Deep Optimization: A Hybrid-Partition Pipeline Parallel Strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Even after the “subtractions,” the system still contains a large amount of unavoidable work: physics simulation, rendering, environment synchronization, and more. If these remain strictly serialized with model rollout, the overall throughput ceiling is still limited. We therefore introduced a **hybrid-partition pipeline parallel strategy**.

From a system perspective, a natural idea is to pipeline model rollout and environment stepping: while one subset of environments is executing the current action chunk, the model can prepare the next batch of rollout results. But the real difficulty is not whether to use a pipeline, but **how to partition BEHAVIOR environments**.

If we partition only by process — assigning some environment processes to stage A and the rest to stage B — the scheme looks simple on paper, but it is inefficient for BEHAVIOR. The reason is that one process often corresponds to one vectorized environment, and multiple instances inside that vectorized environment still share the same global physics / rendering step. As a result, the critical path still contains large serial or semi-serial segments.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-parallelism-en.png
   :alt: Illustration of BEHAVIOR's two-level parallel structure
   :align: center

*Figure 8: Illustration of BEHAVIOR's two-level parallel structure.*

An illustration of BEHAVIOR’s two-level parallel structure: process-level parallelism and vectorized-environment-level parallelism.

This is why our final solution is not “pure process partitioning,” but a **hybrid partition strategy**:

- On one hand, it uses **process-level parallelism** to expand throughput;
- On the other hand, it pushes the partition boundary further down into the **shard / slice granularity inside vectorized environments**;
- Then it maps the environment slices belonging to the same pipeline stage onto the corresponding vectorized environments across all subprocesses.

As a result, one stage no longer “owns several complete processes.” Instead, each stage advances its assigned environment slices in parallel across all subprocesses. The key benefits are:

- A stage is no longer blocked by the long serial segment of one complete vectorized environment;
- Environment resources are utilized more evenly;
- Model forward passes and environment stepping overlap more fully;
- The system throughput ceiling is raised without linearly driving up resource cost.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-pipeline-stage-en.png
   :alt: Comparison of pipeline partition strategies
   :align: center

*Figure 9: Comparison of pipeline partition strategies.*

Comparison of pipeline partition schemes, using two processes with four vectorized environments per process as an example.

At the same time, finer partitioning does not mean simulation semantics can be changed arbitrarily. For a high-fidelity simulator like BEHAVIOR, physical time scale and task semantics must remain consistent. Therefore, after introducing multiple pipeline stages, we also adjust the physics frequency accordingly, so the overall physical time scale remains consistent with the original configuration rather than causing the “simulated world” to run faster simply because the number of stages increased.

In RLinf, this strategy has already been encapsulated into a configuration option. Users only need to set:

.. code-block:: yaml

   env:
      pipeline_stage_num: 2

RLinf will then automatically perform environment slicing, scheduling, and coordination with rollout on the environment side.

The conclusion of this section is that the key to this optimization is not making one isolated module faster, but reorganizing the work that must happen around the real dataflow of reinforcement learning rollouts: first remove wasted work, then restructure the unavoidable work in a way that better matches BEHAVIOR’s internal structure.

4. Performance Evaluation: Where the 25× Speedup Comes From
------------------------------------------------------------

To evaluate the practical effect of these optimizations, we conducted experiments on a server with an **AMD EPYC 7542 32-core CPU** and **4 × NVIDIA RTX 4090 D GPUs**. The model used was **OpenPI Comet 3B**, an open-weight model with strong performance on BEHAVIOR. Because simulator time dominates rollout cost, we used a disaggregated resource layout:

- **GPU 0** runs the OpenPI model;
- **GPU 1–3** each run 2 BEHAVIOR simulator processes;
- Each simulator process runs 2 vectorized environments;
- Each rollout epoch executes a fixed **2048 steps**.

4.1 End-to-End Rollout Time: From 1028.7 ms/step to 41.2 ms/step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We first focus on the most direct metric: under the same training setup, how much does end-to-end rollout latency change before and after optimization?

The result is very clear. Before optimization, the average rollout latency was **1028.7 ms/step**. After the full optimization stack, it dropped to **41.2 ms/step**, for a total speedup of **25×**. Looking at the BEHAVIOR simulator alone, the average execution time was about **473.4 ms/step** before optimization, and dropped to **13.2 ms/step** after optimization, for a total speedup of **36×**.

The significance of this result is not limited to “the environment became faster.” In embodied reinforcement learning, rollout throughput determines how fast the training system can collect interaction data, and therefore directly affects the cadence of model updates, experimental iteration speed, and the task scale that can be covered per unit time. After optimization, the bottleneck of the training system also changed fundamentally: the end-to-end latency, which was previously dominated entirely by the environment, was pulled back into a much more balanced regime across model, environment, and scheduling.

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-perf-rollout.png
   :alt: Ablation of each optimization item total rollout epoch time
   :align: center

*Figure 10: Ablation of each optimization item (total rollout epoch time).*

.. image:: https://github.com/rlinf/misc/raw/main/pic/behavior/behavior-perf-env.png
   :alt: Ablation of each optimization item average BEHAVIOR step cost
   :align: center

*Figure 11: Ablation of each optimization item (average BEHAVIOR step cost).*

Ablation results for each optimization item. The left figure shows the total time cost of each rollout epoch, and the right figure shows the average per-step cost of the BEHAVIOR environment.

4.2 Ablation by Optimization Component: How the Gains Accumulate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To answer the question “where exactly does the 25× speedup come from?”, we ran an ablation study over the three categories of optimization. The results show that this is not an accidental gain from one isolated trick, but the accumulated effect of multiple layers of optimization along the same critical path.

First, disabling viewports and removing unnecessary modalities (Slim) brought a **4.8×** end-to-end rollout speedup. This confirms that the GUI display path and multimodal observation redundancy identified in Section 2 are indeed very real burdens during rollout. These features are valuable for interactive debugging, of course, but in headless training they contribute cost with almost no throughput benefit.

Second, skipping intermediate observations inside the action chunk (SkipObs) brought a further **3.3×** improvement. This gain is even larger than many intuitive “low-level optimizations” because it directly eliminates a large block of wasted work in the default execution path: if intermediate frames that never enter model inference are still rendered, observed, and packaged as usual, the expensive visual pipeline is effectively being run for nothing. Once this work is removed, step latency drops immediately and substantially.

Finally, pipeline parallelism and scheduling optimization (All), on top of the first two “subtractive” optimizations, brought another **1.6×** improvement. Although this number looks less dramatic than the first two, its importance is equally high: it shows that once wasted work has been removed, the system’s main bottleneck shifts from “doing too much unnecessary work” to “whether the remaining necessary work can be organized more efficiently.” In that sense, hybrid-partition pipeline parallelism is more about raising the system ceiling than merely cleaning up local overhead.

4.3 Resource Usage Changes: Not Just Faster, Also More Efficient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A good system optimization should not simply trade more resources for higher throughput. Fortunately, what we observe is simultaneous improvement in both throughput and resource efficiency.

The most obvious change is on the CPU side. Before optimization, a single environment occupied **10 CPU cores** on average. After optimization, that number dropped to **2.1 cores**. This means we not only reduced per-step latency, but also greatly lowered the CPU cost of parallel scaling. For training scenarios with many environments and many concurrent experiments, this is extremely important, because it directly determines how many rollout workers the system can support under the same hardware budget.

The change on the GPU side more clearly shows that “removing wasted work” really works. By cutting redundant viewports and observation modalities, we reduced memory overhead related to rendering, annotators, and buffers. As a result, the GPU memory usage of a single simulator instance dropped from an average of **11.5 GiB** to **8.2 GiB**. With lower memory pressure, not only does each instance become lighter to run, but the whole system also scales better under multi-environment concurrency, and OOM events become much less frequent. In addition, model rollout GPU utilization increased by **69%**, because the environment no longer blocks model forward passes so frequently, and the entire system becomes more deeply pipelined.

4.4 An Open Question: What Is the Optimal Resource Layout?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After this round of optimization, a more interesting question begins to emerge: once the environment is no longer so slow that it overwhelms everything else, the optimal training-system configuration actually becomes more subtle.

The resource profiles of the environment and the model are different:

- The model tends to consume GPU compute and memory in a relatively stable way;
- The environment simultaneously depends on CPU, graphics rendering capability, and GPU memory;
- The optimum shifts with model size, action chunk length, number of environments, and the CPU / GPU ratio.

This means that the “best resource layout” is probably not one fixed answer, but a scheduling problem whose solution varies with the configuration. For example:

- With a small model and many environments, the environment side may become the tighter resource bottleneck;
- As the model grows or the action chunk length increases, rollout may again become the bottleneck;
- Different numbers of pipeline stages and different combinations of process count / vectorized-environment count will also push the system toward different equilibrium points.

In the long run, this is a highly worthwhile direction: automatic resource layout / auto-tuning could let the system discover a better execution plan based on the model, environment, and hardware configuration.

5. Conclusion: What High-Fidelity Embodied Environments Teach Us
----------------------------------------------------------------

One core lesson from this optimization work is that **the default execution path of a high-fidelity simulator is often not designed for large-scale reinforcement learning rollout**. Systems like BEHAVIOR need to simultaneously serve interaction, visualization, debugging, evaluation, and research. As a result, their default configurations naturally retain many paths that are friendly for developers but unfriendly for training throughput.

To truly use such a system for large-scale training, the first step is not to blindly add more machines, nor to immediately start micro-optimizing one local compute kernel. The first step is to understand clearly:

- Which work is truly required by training;
- Which work is only extra cost preserved for other historical use cases;
- Which necessary work can be reordered and overlapped in a way that better matches the real dataflow.

From this perspective, all three categories of optimizations in this post follow the same principle: **make the execution path obey the real dataflow, rather than the system’s default organization.**

Another important lesson is that environment optimization is not just optimization of the environment itself, but optimization of the entire training system. In embodied AI tasks, the coupling among environment, model, resource layout, pipeline structure, and scheduling is much stronger than in many traditional benchmarks. Once one bottleneck is relieved, higher-level coordination and scheduling problems start to surface.

The related performance patches have now been merged into **RLinf** and the upstream **StanfordVL/BEHAVIOR-1K** repository. For developers who want to start using BEHAVIOR directly, RLinf already provides a mature solution that can be used without having to rework the environment from scratch. We also hope this work can serve not only our own training system but the broader embodied AI community as well: with sufficient understanding of reinforcement learning dataflow and low-level system structure, even a complex physical simulation world can be rebuilt into efficient compute infrastructure.

Image / video credits: Physical Intelligence, BEHAVIOR-1K.
