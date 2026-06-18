LoRA Integration
===================

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method that injects small low-rank matrices into existing layers, enabling efficient adaptation for large models while reducing memory and training cost without altering original weights.
This guide explains how to use LoRA in RLinf.

Configuration
-------------

LoRA can be configured in the actor model section of your YAML configuration:

.. code:: yaml

  actor:
    model:
      is_lora: True
      lora_rank: 32
      lora_path: null  # or path to existing LoRA weights

**Parameters:**

- ``is_lora``: Enable LoRA fine-tuning (True/False)
- ``lora_rank``: Rank of LoRA matrices (typically 8-64), LoRA trains two matrices A and B for each layer, with shapes [input-dim, lora-rank] and [lora-rank, output-dim] respectively
- ``lora_path``: Path to pre-trained LoRA weights (null for new training)

Target Modules
---------------

RLinf automatically applies LoRA to the following modules:

.. code:: python

  target_modules = [
      "proj",      # General projection layers
      "qkv",       # Query-Key-Value projections
      "fc1",       # Feed-forward layers
      "fc2",       # Vision-specific layers
      "q",         # Query projections
      "kv",        # Key-Value projections
      "fc3",       # Additional projection layers
      "q_proj",    # Query projection
      "k_proj",    # Key projection
      "v_proj",    # Value projection
      "o_proj",    # Output projection
      "gate_proj", # Gate projection (for SwiGLU)
      "up_proj",   # Up projection
      "down_proj", # Down projection
      "lm_head",   # Language model head
  ]

New LoRA Training
~~~~~~~~~~~~~~~~~

To start training with LoRA from scratch:

.. code:: yaml

  actor:
    model:
      is_lora: True
      lora_rank: 32
      lora_path: null

**Process:**

1. Load the base model
2. Apply LoRA configuration with specified rank
3. Initialize LoRA weights with Gaussian distribution
4. Train only LoRA parameters

Loading Pre-trained LoRA
~~~~~~~~~~~~~~~~~~~~~~~~~

To continue training with existing LoRA weights:

.. code:: yaml

  actor:
    model:
      is_lora: True
      lora_rank: 32
      lora_path: "/path/to/pretrained/lora/weights"

**Process:**

1. Load the base model
2. Load pre-trained LoRA weights from specified path
3. Set LoRA parameters as trainable
4. Continue training

Full Model Fine-tuning
~~~~~~~~~~~~~~~~~~~~~~

To disable LoRA and use full model fine-tuning:

.. code:: yaml

  actor:
    model:
      is_lora: False
      lora_rank: 32  # Ignored when is_lora=False

**Process:**

1. Load the base model
2. Make all parameters trainable
3. Train the entire model
