Disaggregated Mode
==================

.. image:: ../../../_static/svg/disaggregated.svg
   :width: 600px
   :align: center
   :class: dis-img

Different RL tasks are mapped to different GPU groups according to their computation needs. There also two execution modes: the workers run sequentially one after another or the workers run concurrently with fine-grained pipelining.

**Pros**

* Flexible worker assignment.
* No requirement for offloading implementation.

**Cons**

* Data-flow dependencies lead to GPUs idle.  
* Pipelining should be implemented to reduce GPU idle time.

**Example configuration**

The workers are assigned to separate GPUs. The set of GPUs is specified using global GPU indices.

.. code:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       rollout: 0-9
       inference: 10-11
       actor: 12-15

.. note::
   The ``pipeline_stage_num`` configuration should be adjusted to achieve the desired pipelining effect. 

Refer to :doc:`../user/yaml` for compete configuration.

**ComponentPlacement programming**

As described in :doc:`collocated`, the placement configuration in the yaml file can be parsed by `ComponentPlacement` and enforced on workers. Refer to `Math RL training with pipelining <https://github.com/RLinf/RLinf/blob/main/examples/reasoning/main_grpo.py>`_ for the complete code.
