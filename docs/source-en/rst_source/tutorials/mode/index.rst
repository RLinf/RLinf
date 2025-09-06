Flexible Execution Modes
========================

.. Conventional RL post-training systems can typically be categorized—based on their GPU placement strategy—into two primary modes: :doc:`collocated` and :doc:`disaggregated`.

.. In collocated mode, all major components (i.e., workers), such as generator, actor inference, and actor training, share the same set of GPUs or nodes. 
.. In contrast, disaggregated mode places these components on separate GPUs or nodes. 

.. However, these modes cannot well deal with the RL workload such as embodied intelligence, which has more components (e.g., simulators) and more complex communication among components, e.g., fine-grained interactions between the **simulator** and **generator**.

.. To efficiently adapt to variable RL workloads, RLinf supports flexible component placement and execution mode, enabling diverse component orchestration. 
.. The components can be placed on any GPUs. 

.. - When two components are placed on the same set of GPUs, users can configure either the two components are resident in GPU memory or the two components swith the usage of the GPUs based on offloading/reloading mechanism. 

.. - When the two components use separate GPUs, they can run sequentially one after the other, which induces GPU idle time, or they can run in a pipelined manner, so all GPUs stay busy. 

.. :doc:`hybrid` enables flexible configuration of placement and execution mode.

Conventional RL post-training systems are typically classified—based on their GPU placement strategy—into two primary modes: :doc:`collocated` and :doc:`disaggregated`.

In the **collocated** mode, all major components (e.g., generator, actor inference, and actor training) share the same set of GPUs or nodes.  
In contrast, the **disaggregated** mode assigns these components to separate GPUs or nodes.

However, neither mode is well-suited for complex RL workloads such as embodied intelligence, which involve more components (e.g., simulators) and more intricate communication patterns—for instance, the fine-grained interactions between the **simulator** and the **generator**.

To better accommodate diverse and dynamic RL workloads, **RLinf** supports flexible component placement and execution modes, enabling users to orchestrate components in a highly adaptable way.  
In particular, components can be placed on **any GPUs** with configurable execution strategies:

- **Collocated on the same GPUs:**  
  Users may configure whether both components remain resident in GPU memory, or whether they switch usage of the GPUs via an offloading/reloading mechanism.

- **Distributed on separate GPUs:**  
  Components may either run sequentially—potentially causing GPU idle time—or execute in a pipelined fashion, ensuring that all GPUs remain busy.

The :doc:`hybrid` mode further extends this flexibility by supporting customized combinations of placement and execution strategies.


.. toctree::
   :hidden:
   :maxdepth: 1

   collocated
   disaggregated
   hybrid