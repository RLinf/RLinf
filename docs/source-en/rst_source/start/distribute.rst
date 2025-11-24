Multi-node Training
===================

This guide shows how to launch a **4-node Ray cluster** (each node
has **8 GPUs**) and run distributed RL training on
the *math* task with the
`DeepSeek-R1-Distill-Qwen-1.5B <https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B>`_
model.  
The same procedure scales to any number of nodes/GPUs, as long as you customize the YAML configuration according to your needs.


Prerequisites
-------------

Before running, make sure to check the following:

* Clone RLinf to a shared filesystem accessible by all nodes.
* Ensure that each node has started the corresponding container image.



Step 1: Start a Ray Cluster
----------------------------

Clean up *old* cached state first:

.. code-block:: bash

   rm -f ray_utils/ray_head_ip.txt

Open a shell on *each* node and run:

==========================================  ==========================
node index                                  command
==========================================  ==========================
0 (head)                                    ``RANK=0 bash ray_utils/start_ray.sh``
1                                           ``RANK=1 bash ray_utils/start_ray.sh``
2                                           ``RANK=2 bash ray_utils/start_ray.sh``
3                                           ``RANK=3 bash ray_utils/start_ray.sh``
==========================================  ==========================


Once the scripts run successfully, the terminal on the **head node** should display output similar to the following (for simplicity, we only show the example of 2 nodes with 16 GPUs):

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/start-0.jpg" width="800"/>

On each **worker node**, the terminal should display:

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/start-1.jpg" width="800"/>

After all four startup scripts print *Ray started*, **remain** in the head node terminal and verify the total cluster size (in this example, ``4 × 8 = 32`` GPUs):

.. code-block:: bash

   bash ray_utils/check_ray.sh 32

.. note::

   The argument to ``check_ray.sh`` must equal the number of accelerators/GPUs in the cluster. 

If successful, your terminal should show:

.. raw:: html

   <img src="https://github.com/RLinf/misc/raw/main/pic/check.jpg" width="800"/>

Note: For simplicity, the images in this example only show a 2-node setup with 16 GPUs.


Step 2: Launch Training Tasks
------------------------------------

Here we provide startup examples in two modes: collocated mode and disaggregated mode.

Collocated 
^^^^^^^^^^^^^^

Every training stage (rollout, inference, actor) shares **all GPUs**.
Edit the sample YAML:

.. code-block:: yaml

   # examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron.yaml
   cluster:
     num_nodes: 4          # adapt to your cluster
     component_placement:
       actor,rollout: all  # “all” means the whole visible GPU set

Launch from the head node:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh \
        qwen2.5-1.5b-grpo-megatron


Disaggregated
^^^^^^^^^^^^^^^^^^

Different stages receive disjoint GPU ranges,
allowing fine-grained pipelining. Edit the pipeline YAML:

.. code-block:: yaml

   # examples/reasoning/config/math/qwen2.5-1.5b-grpo-megatron-pipeline.yaml
   cluster:
     num_nodes: 4
     component_placement:
       rollout:    0-19        # 20 GPUs
       inference:  20-23       # 4  GPUs
       actor:      24-31       # 8  GPUs

* ``rollout + inference + actor`` **must equal** the total GPU count
  (here ``32``).
* Ranges are inclusive.

Start the job:

.. code-block:: bash

   bash examples/reasoning/run_main_grpo_math.sh \
        qwen2.5-1.5b-grpo-megatron-pipeline