Training Visualisation
======================

RLinf support for real-time experiment tracking.
You can stream loss curves, accuracy, GPU utilization and arbitrary
custom metrics to one or more of the following backends:

- `TensorBoard <https://www.tensorflow.org/tensorboard>`_: 
  A widely used, open-source visualization tool 
  (from TensorFlow, also works with PyTorch, Hugging Face, etc.) 
  that lets you track metrics like loss and accuracy, visualize model graphs, embeddings, images, and more. 

- `Weights & Biases (W&B) <https://wandb.ai/site/>`_:
  A SaaS-based platform offering experiment tracking, hyperparameter sweeps, 
  artifacts (for model and data versioning), reporting, and collaborative features for teams. 

- `SwanLab <https://pypi.org/project/swanlab/>`_:
  An open-source, lightweight experiment logging and visualization tool designed for local or self-hosted use. 
  It provides intuitive Python APIs, logs metrics, hyperparameters, hardware and code info, 
  and supports experiment comparison through a clean UI — ideal for privacy-focused workflows. 

Enabling a back-end
-------------------

Add the desired logger(s) to ``runner.logger.logger_backends`` in your YAML:

.. code-block:: yaml

   runner:
     task_type: math
     logger:
       log_path: ${runner.output_dir}/${runner.experiment_name}
       project_name: rlinf
       experiment_name: ${runner.experiment_name}
       logger_backends: ["tensorboard", "wandb", "swanlab"]   # <─ choose any subset
     experiment_name: grpo-1.5b
     output_dir: ./logs

RLinf creates a sub-directory for each active back-end:

.. code-block:: text

   logs/grpo-1.5b/
   ├── checkpoints/
   ├── converted_ckpts/
   ├── log/                
   ├── swanlab/            # SwanLab event files
   ├── tensorboard/        # TensorBoard event files
   └── wandb/              # WandB run directory


TensorBoard
-----------


.. code-block:: bash

   tensorboard --logdir ./logs/grpo-1.5b/tensorboard --port 6006

Open `http://localhost:6006` in your browser
to inspect scalar plots, histograms and the computation graph.


Weights & Biases (WandB)
------------------------

#. Create a free account at `wandb.ai <https://wandb.ai>`__ and copy your
   *API key*.
#. Authenticate once per machine:

.. code-block:: bash

    wandb login          # paste API key when prompted

From now on RLinf will automatically start a new *run* and stream all
metrics. You can check the metrics through your dashboard.


SwanLab
-------


#. Register at `swanlab.ai <https://swanlab.ai>`__ and obtain an
   *access token*.
#. Authenticate:

.. code-block:: bash

    swanlab login        # paste access token

From now on RLinf will automatically start a new *run* and stream all
metrics. You can check the metrics through your dashboard.


.. tip::

   All three loggers run **in parallel**; feel free to mix and match.

