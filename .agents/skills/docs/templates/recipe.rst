.. Recipe page for an Examples gallery: copy this file, replace every <...>, and
.. delete these comments. Rules: .agents/skills/docs/rules/recipe.md. Match the
.. section names, card titles, and table columns of the other pages in the same
.. gallery; the cards below are the Models-gallery set.

RL with <Name>
==============

.. figure:: <upstream figure URL on raw.githubusercontent.com>
   :align: center
   :width: 90%

   <What the figure shows on this page, with credit for upstream images.>

<One paragraph: what <Name> is, what this page runs with RLinf, and the result you reach.>

Overview
--------

<One line: what you train or evaluate here.>

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      <every env this model supports>

   .. grid-item-card:: Algorithms
      :text-align: center

      <every algorithm>

   .. grid-item-card:: Tasks
      :text-align: center

      <suites or tasks>

   .. grid-item-card:: Hardware
      :text-align: center

      <nodes · GPUs · backends>

| **You'll do:** install → download the model → launch → watch ``<metric>``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · <other things to prepare>.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Environment
     - Task / Suite
     - Config / Weights
     - Focus
   * - <env>
     - <suite>
     - ``<config name or model repo>``
     - <what it tests, in one line>

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Field
     - Description
   * - Observation
     - <what the policy sees>
   * - Action
     - <what the policy outputs>
   * - Reward
     - <how success is scored>
   * - Prompt
     - <the task instruction>

Setup and Run
-------------

.. include:: _setup_common.rst

.. tab-set::

   .. tab-item:: <Model / Suite>

      **1. Install**

      .. code-block:: bash

         git clone https://github.com/RLinf/RLinf.git && cd RLinf
         bash requirements/install.sh embodied --model <model> --env <env> && source .venv/bin/activate

      **2. Download**

      .. code-block:: bash

         hf download <org>/<repo> --local-dir ./<checkpoint>

      **3. Launch**

      .. list-table::
         :header-rows: 1
         :widths: 30 40 30

         * - Setting
           - Config (``examples/embodiment/config/``)
           - Notes
         * - <model + algorithm>
           - ``<config_name>.yaml``
           - <one-line focus>

      .. code-block:: bash

         bash examples/embodiment/run_embodiment.sh <config_name> actor.model.model_path=$PWD/<checkpoint>

      .. include:: _model_path.rst

Visualization and Results
-------------------------

<Where to watch progress, for example env/success_once in TensorBoard, linked to the Training metrics page.>

<Results table: setting, metric, value, and the config that reproduces it.>
