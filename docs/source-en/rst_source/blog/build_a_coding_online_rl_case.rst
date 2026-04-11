A First Look at the "Last Mile" of Agent Deployment: Cursor Online Reinforcement Learning
=========================================================================================

Last updated: 10/20/2025.

Related reading: :doc:`Online RL for Code Completion <../examples/agentic/coding_online_rl>`.

1. Background
-------------

Recently, the Cursor team introduced a new Tab model based on **Online Reinforcement Learning (Online RL)** (`link <https://mp.weixin.qq.com/s/ShalRibfp9YSE5UFS0GLVg>`_). This model treats every user interaction (e.g., accepting or rejecting a suggestion) as a reinforcement signal that directly participates in online optimization. Driven by over 400 million daily requests, the model can learn and improve at very high frequency, becoming the first successful case of enhancing a production service with Online RL.

With the rise of the **Agentic AI** era, more and more agents are being deployed in real production environments and personalized via Online RL with adaptive performance optimization. This paradigm is likely to become the key technical path to realize the “last mile” of agent deployment.

Sensing this trend, our team reproduced Cursor’s Online RL approach on the large-scale RL framework **RLinf**, exploring whether Online RL can further improve the model’s code completion capability. Specifically, we use **Continue** as the editor-side component, where users can directly install the `Continue plugin <https://github.com/RLinf/continue>`_ in VSCode for development; **RLinf** serves as the backend system, responsible for both LLM serving and Online RL training. Notably, the LLM serving backend can also be replaced with an already deployed Agent service, while **RLinf** can flexibly plug in as a general component that provides Online RL capabilities to the system.

2. Preliminaries
----------------

2.1 Code Completion in Brief
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For intelligent programming editors like **Cursor**, one of the core capabilities is efficient, context-aware **code completion**. In a typical completion task, when the cursor stays at a position, the assistant proposes an insertion. A common way to implement this capability is the **FIM (Fill-In-the-Middle)** task: the model receives the prefix (above the cursor) and the suffix (below the cursor) and predicts the middle content. Most LLMs are pre-trained for FIM and only require the following input format:
`<|fim_prefix|>prefix<|fim_suffix|>suffix<|fim_middle|>`

The model then generates reasonable middle completions. This design naturally adapts LLMs to editor code completion.

2.2 Code Completion Logic in the Continue Plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In our experiments, we implemented a complete Online RL code completion case based on the open-source AI coding IDE **Continue**.

Continue natively supports FIM-based completion tasks, but the upstream version does not include human feedback reporting.

We therefore made a lightweight modification:

- When the user presses **Tab** to accept a suggestion, the plugin reports accepted=True;
- When the user does not press Tab within 10 seconds or performs other edits, it reports accepted=False.

This way, we can directly obtain reinforcement signals (reward) from real user interactions, without training a separate reward model to fit human preferences, achieving a **direct human-to-model reinforcement learning loop**.

3. System Setup
---------------

3.1 Flow Overview
~~~~~~~~~~~~~~~~~

The entire Online RL flow can be summarized in three core steps:

1. **Interaction and feedback collection**:

   The Coding Agent provides completion suggestions to the user; the user’s accept or reject forms a clear reinforcement signal.
2. **Immediate online updates**:

   User feedback is sent to the RLinf backend for training the generative model. The model is updated on-policy and synced back online to the Agent to obtain new interaction feedback.
3. **Effect validation and deployment**:

   After online training, use **A/B** testing to evaluate whether the acceptance rate of the new model surpasses the original. If improved, deploy to production.

3.1 RLinf-Online Setup
~~~~~~~~~~~~~~~~~~~~~~

Below is how to quickly build the flow with RLinf:

(1) RLinf Worker abstraction

The RLinf framework provides the Worker programming interface, the basic building block of RLinf. A Worker represents an executable component: at the large scale it can be an inference instance or a training framework; at the small scale it can be a data loader, etc. By subclassing Worker, we can abstract a concrete executable unit and enable interaction with other Workers, as well as scheduling, placement, and management by RLinf.

(2) RLinf Channel communication

The RLinf framework provides a high-performance, easy-to-use asynchronous communication abstraction, Channel. It adaptively uses optimized point-to-point backends (e.g., CUDA IPC and NCCL) and exposes a producer–consumer queue pattern. Thus, communication from Worker1 -> Worker2 can be implemented as follows:

.. code-block:: python

   self.comm_channel = Channel.create("Comm") # create a channel

   Handle1 = self.worker1.rollout(
       output_channel=self.comm_channel,
   ) # data generation

   Handle2 = self.worker2.run_inference(
       input_channel=self.comm_channel,
   ) # run inference

With just 3 lines of code, we can implement Worker1 -> Worker2 communication, greatly simplifying the logic.

(3) Building the Online RL training flow with RLinf

With Worker and Channel in hand, we can assemble the full Online RL training flow. The overall system architecture is shown below.

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/coding_online_rl_arch.png" width="800"/>

Assume the code completion Agent has been deployed as a complete online service, consisting of a **User Frontend** and a **Service Backend**. To enable Online RL capability, we introduce an independent component — **RLinf Runner** — at the **plugin** layer. Unlike long-running background services, RLinf Runner is not a resident process; it is a lightweight module that can be invoked on demand by the **Controller** in the online system. We design interaction interfaces between RLinf Runner and the online Agent to:

1. Ingest online data, including requests, responses, and user feedback (accept/reject);
2. Receive and update model weights to realize real-time policy optimization for the Agent.

Inside the RLinf Runner, we decompose the RL process into three core Workers:

- **Data Receiver**: receives and buffers interaction data from the online system;
- **Compute Reward**: computes immediate rewards based on user feedback;
- **PPO Loss + Actor Trainer**: performs policy optimization and model updates.

These Workers communicate via RLinf Channel, which provides high-performance, asynchronous data transfer so the entire online training process can run in a streaming manner. Once the Service Backend’s Controller launches the RLinf Runner, the Online RL process runs automatically: the system receives data from the online service, computes rewards, updates the policy model, and returns improved model weights to the service backend in real time. To ensure stability, Online RL can first be deployed and validated on a subset of users who opt in to new-model experiments.

4. Algorithm Design
-------------------

Beyond modular system design, we also explored **online RL algorithm design** in depth. In Online RL, each request typically corresponds to one response and one user feedback (accept/reject), so **GRPO** no longer applies because it relies on diverse response groups for the same input to compute relative preference. We therefore adopt an improved **PPO** without a **critic model**, where advantage estimation degenerates to **Monte Carlo return**. Although this can introduce higher training variance, PPO’s **clip mechanism** effectively limits the update magnitude and prevents collapse, yielding an **efficient and stable simplification**. In code completion Online RL training, the **reward comes from user feedback** (i.e., accept vs. reject).

Due to a lack of sufficiently large real Online usage scenarios at present, we adopt **LLM-as-a-Judge** to rate model completions. Concretely, we use the LLM (DeepSeek-V3.1) to score each completion from 0–10, and the average score serves as the aggregate performance metric on the test set.

5. Performance at a Glance
--------------------------

5.1 Training Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Dataset construction**

We select the **code-fim-v2** dataset, which contains code completion samples from multiple programming languages. We filter for Python samples and remove those with overly short completions, ending up with about **4,000 high-quality samples**. **3,000** are used for training and **1,000** for testing. Each sample contains prefix and suffix code snippets; the model must generate the middle completion based on context.

**Key parameters**

The base model is **Qwen2.5-Coder-1.5B**. As no KL regularization is used, an overly high learning rate may cause the model to forget its prior distribution, so we choose a small LR (2e-6) for stable convergence. We use bf16 training precision, which yields more stable gradient norms than fp16 in early training.

Additionally, to quickly verify RL’s effectiveness on this task, we also conduct an offline **GRPO (group size = 8)** experiment for comparison, evaluating performance changes under different training paradigms.

5.2 Experimental Results
~~~~~~~~~~~~~~~~~~~~~~~~

As shown in Figure 1, model performance steadily increases with Online RL. The test set results in Table 1 show that Qwen2.5-Coder-1.5B-RLinf improves significantly on the test set (4.532 -> 6.897), a gain of over 50%, even surpassing the 32B model in the same series. This indicates that Online RL can effectively improve deployment performance and that small models have great potential.

.. list-table::
   :widths: 50 50
   :header-rows: 0
   :align: center

   * - .. image:: https://github.com/RLinf/misc/raw/main/pic/coding_online_rl_offline_rewards.png
          :width: 100%
     - .. list-table::
          :header-rows: 1
          :align: center

          * - Model
            - Score
          * - Qwen2.5-Coder-1.5B
            - 4.532
          * - Qwen2.5-Coder-3B
            - 5.139
          * - Qwen2.5-Coder-7B
            - 5.68
          * - Qwen2.5-Coder-14B
            - 6.351
          * - Qwen2.5-Coder-32B
            - 6.545
          * - Qwen2.5-Coder-1.5B-RL
            - 6.897 (+52%)
   * - Figure 1 Reward during training
     - Table 1 Test set scores (0–10)

6. Outlook
----------

RLinf-online represents our team’s initial exploration into online optimization for agents. The current version simulates human performance via proxy, but the results already demonstrate the vast potential of Online RL. The team is putting this flow into production for testing in real business scenarios. Meanwhile, the RLinf team looks forward to collaborating with the community to jointly explore the boundaries of reinforcement learning in the era of large models!


