.. This file is a reusable include, not a standalone page.
   It is shared by the embodied example recipes to avoid repeating the
   identical "clone the repo + choose an install method" boilerplate.
   Include it with ``.. include:: _setup_common.rst`` and then add the
   recipe-specific Docker tag / ``--env`` value below it.

First, clone the RLinf repository:

.. code:: bash

   # Mainland China users can use a mirror for faster cloning:
   # git clone https://ghfast.top/github.com/RLinf/RLinf.git
   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

Then set up the dependencies with **one** of the two methods below — a prebuilt
**Docker image** (recommended) or a **custom environment**. The general setup
(prerequisites, GPU drivers, the in-image ``switch_env`` helper, mirrors, and
troubleshooting) is documented once in :doc:`/rst_source/start/installation`;
the commands in this recipe only differ in the Docker **image tag** and the
``--env`` value.
