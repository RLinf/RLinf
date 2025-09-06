Online Scaling Mechanism
========================


Online scaling (also known as elastic training)  
is a powerful feature that enables dynamic scaling of training resources, with GPU swithching performed within 1 seconds.
This capability allows you to adjust the number of GPUs and nodes used for training in real time,  
based on cluster availability, workload demands, or resource optimization goals.

What is Online Scaling?
-----------------------

Online scaling refers to the ability to **scale up** (add more resources) or **scale down** (remove resources)  
during training while maintaining training continuity and model state consistency.  

In the context of RL training with Megatron-LM, this involves:

- **Scaling up**: Adding nodes/GPUs to increase training throughput  
- **Scaling down**: Releasing nodes/GPUs to free up resources for other tasks  
- **Parallel strategy adjustment**: Dynamically changing Megatron's parallel strategies (TP/PP/DP/CP)

The system automatically handles:

- Model parameter redistribution across the new parallel configuration  
- Optimizer state migration  
- Communication group reconstruction  
- Training state synchronization  

Why is Online Scaling Important?
--------------------------------

When using RLinf's disaggregated mode with fine-grained pipelining,  
the rollout and inference stages are completed before the actor stage finishes.  
At this point, the resources used for rollout and inference can be reallocated to the actor stage **within seconds**,  
accelerating actor training and improving overall system performance.

Benefits and Effects
--------------------

**Performance Benefits:**

- **Increased Throughput**: Adding more GPUs can significantly speed up training  
- **Better Resource Utilization**: Dynamic resource allocation ensures optimal usage  
- **Reduced Training Time**: Efficient scaling can reduce overall training time by 20–50%  

**Operational Benefits:**

- **Zero Training Interruption**: Scaling occurs seamlessly without halting training  
- **Consistent Training Progress**: Maintains convergence and model continuity throughout scaling  
