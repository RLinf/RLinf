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

Installation
------------

.. include:: _setup_common.rst

<The recipe-specific image tag or install command, each introduced by what it does.>

Download the Model
------------------

<The download command for the exact weights this recipe uses.>

.. include:: _model_path.rst

Run It
------

<One sentence on what the command does.>

.. code-block:: bash

   bash examples/embodiment/run_embodiment.sh <config_name>

<What this command does, in order, and the config keys you may change.>

Visualization and Results
-------------------------

<Where to watch progress, for example env/success_once in TensorBoard, linked to the Training metrics page.>

<Results table: setting, metric, value, and the config that reproduces it.>
