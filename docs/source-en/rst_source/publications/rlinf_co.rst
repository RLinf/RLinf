Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models
===================================================================================

**Paper:** `arXiv:2602.12628 <https://arxiv.org/abs/2602.12628>`__

Overview
--------


.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-co/overview.png
   :alt: RLinf-Co overview
   :align: center

Overview of the proposed two-stage sim-real co-training framework. We establish a digital-twin setup where :math:`T_{\text{sim}}` serves as a digital cousin to :math:`T_{\text{real}}` despite visual discrepancies. In **Stage I**, we initialize the VLA policy by supervising it on a mixture of real and simulated data (ratio :math:`\alpha`). This rapidly injects real-world knowledge and prepares the policy for simulation interaction. In **Stage II**, we perform RL fine-tuning in the simulator to explore and improve performance, simultaneously employing a real-world SFT loss as a regularizer to prevent the forgetting of real-world behaviors.

Results
-------

Main Results
~~~~~~~~~~~~

.. list-table:: Comparison of real-world success rates under different training paradigms
   :header-rows: 1
   :widths: 16 20 13 12 12 12 15
   :align: left

   * - VLA Model
     - Experiment Setting
     - Pick and Place
     - Push Cube
     - Open Drawer
     - Close Drawer
     - Avg.
   * - **OpenVLA**
     - Real-Only Training
     - 6.3 ± 0.0
     - 20.0 ± 13.3
     - 0.0 ± 0.0
     - 10.0 ± 10.0
     - 16.5 ± 13.3
   * -
     - SFT Co-Training
     - 23.4 ± 4.7
     - 51.7 ± 5.0
     - 0.0 ± 0.0
     - 85.0 ± 5.0
     - 40.0 ± 3.7
   * -
     - **RL-Co (Ours)**
     - **58.8 ± 10.0**
     - **68.3 ± 11.7**
     - **35.0 ± 15.0**
     - **95.0 ± 5.0**
     - **64.0 ± 0.7**
   * - **π₀.₅**
     - Real-Only Training
     - 71.9 ± 9.4
     - 0.0 ± 0.0
     - 0.0 ± 0.0
     - 35.0 ± 15.0
     - 26.7 ± 1.4
   * -
     - SFT Co-Training
     - 68.8 ± 9.4
     - 10.0 ± 3.3
     - 10.0 ± 0.0
     - 95.0 ± 5.0
     - 45.9 ± 4.4
   * -
     - **RL-Co (Ours)**
     - **81.3 ± 9.4**
     - **18.4 ± 1.7**
     - **65.0 ± 5.0**
     - **100.0 ± 0.0**
     - **66.2 ± 4.0**

Ablation Study
~~~~~~~~~~~~~~

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-co/success_rate.png
   :alt: Ablation study on simulation SFT initialization
   :align: center

Ablation study on simulation SFT initialization. We report the simulation success rate during RL training for models trained with and without simulation SFT initialization. Each RL training run uses three independent random seeds, and results are presented as mean success rate with shaded regions indicating standard deviation.

Data Efficiency
~~~~~~~~~~~~~~~

.. image:: https://github.com/RLinf/misc/raw/main/pic/rlinf-co/data_efficiency.png
   :alt: Effect of the number of real-world demonstrations
   :align: center

Effect of the number of real-world demonstrations. We vary the number of real-world demonstrations for the ``Open Drawer`` task and evaluate all training paradigms using the :math:`\pi_{0.5}` model. Performance is reported as success rate, with shaded regions indicating standard deviation.

Quickstart
----------

- **Instruction:** :doc:`../examples/embodied/co_training`

Citation
--------

.. code-block:: bibtex

    @article{shi2026rlinf,
      title={Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models},
      author={Shi, Liangzhi and Chen, Shuaihang and Gao, Feng and Chen, Yinuo and Chen, Kang and Zhang, Tonghe and Zhang, Hongzhi and Zhang, Weinan and Yu, Chao and Wang, Yu},
      journal={arXiv preprint arXiv:2602.12628},
      year={2026}
    }
