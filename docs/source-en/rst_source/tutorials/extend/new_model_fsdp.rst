Adding New Models with FSDP+HuggingFace
========================================

This document focus on using the HuggingFace Transformers library with PyTorch FSDP (Fully Sharded Data Parallel)  
to train and generate from models. It supports any model implemented in HuggingFace and compatible with PyTorch.
As an example, this section provides a step-by-step recipe for integrating a new HuggingFace model into RLinf, following the OpenVLA pattern.


Prerequisites
-------------

* Familiarity with **HuggingFace Transformers library**
* Understanding of the **RLinf** framework architecture
* Knowledge of **PyTorch** and distributed training

Step-by-Step Implementation
---------------------------

1. Model Configuration and Registration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Edit `rlinf/models/__init__.py` to extend `get_model_config_and_processor`. 
This registers your model’s `Config`, `ImageProcessor`, and `Processor` so RLinf can load them by name and wire up preprocessing automatically.

.. code-block:: python

  def get_model_config_and_processor(cfg: DictConfig):
      if cfg.model.model_name == "your_model_name":
          from your_package.configuration import YourModelConfig
          from your_package.processing import YourImageProcessor, YourProcessor

          AutoConfig.register("your_model", YourModelConfig)
          AutoImageProcessor.register(YourModelConfig, YourImageProcessor)
          AutoProcessor.register(YourModelConfig, YourProcessor)

          model_config = AutoConfig.from_pretrained(
              cfg.tokenizer.tokenizer_model
          )
          image_processor = YourImageProcessor.from_pretrained(
              cfg.tokenizer.tokenizer_model,
              trust_remote_code=True
          )
          tokenizer = AutoTokenizer.from_pretrained(
              cfg.tokenizer.tokenizer_model,
              trust_remote_code=True,
              padding_side="left"
          )
          input_processor = YourProcessor.from_pretrained(
              cfg.tokenizer.tokenizer_model,
              tokenizer=tokenizer,
              image_processor=image_processor,
              trust_remote_code=True
          )

      return model_config, input_processor

2. Model Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create your class in `rlinf/models/embodiment/your_model_action_model.py` inheriting from a HuggingFace base. 
Implement `predict_action_batch` to wrap generation, decoding, and optional value computation, keeping RL logic encapsulated.

.. code-block:: python

  from transformers import YourBaseModel
  from rlinf.models.embodiment.modules.value_head import ValueHead

  class YourModelForRLActionPrediction(YourBaseModel):
      def __init__(self, config, hidden_size, unnorm_key, vh_mode, action_dim):
          super().__init__(config)
          self._init_logits_processor()
          action_norm_stats = self.get_action_stats(unnorm_key)
          self.min_action = np.array(action_norm_stats["q01"])
          self.max_action = np.array(action_norm_stats["q99"])
          self.value_head = ValueHead(hidden_size)
          self.vh_mode = vh_mode
          self.action_dim = action_dim

      def _init_logits_processor(self):
          self.logits_processors = LogitsProcessorList()
          self.logits_processors.append(
              YourLogitsProcessor(self.config.n_action_bins)
          )

      @torch.no_grad()
      def predict_action_batch(
          self, input_ids=None, attention_mask=None, pixel_values=None,
          do_sample=True, **kwargs
      ):
          generated = self.generate(
              input_ids,
              attention_mask=attention_mask,
              pixel_values=pixel_values,
              output_scores=True,
              output_logits=True,
              output_hidden_states=True,
              return_dict_in_generate=True,
              do_sample=do_sample,
              logits_processor=self.logits_processors,
              **kwargs
          )
          sequences = generated.sequences
          actions = sequences[:, -self.action_dim:]
          logits = torch.stack(generated.logits, dim=1)
          if self.cfg.algorithm.require_values:
              values = self.compute_values(generated.hidden_states)
          else:
              values = torch.zeros_like(logits[..., :1])
          return actions, sequences, logits, values

3. Model Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify `get_model` in `rlinf/models/__init__.py` to call `from_pretrained` for your class when `cfg.model_name` matches. This ensures checkpoints load with the correct dtype, dimensions, and LoRA hooks.

.. code-block:: python

  def get_model(model_path, cfg: DictConfig, override_config_kwargs=None):
      torch_dtype = torch_dtype_from_precision(cfg.precision)

      if cfg.model_name == "your_model_name":
          from .embodiment.your_model_action_model import (
              YourModelForRLActionPrediction
          )
          model = YourModelForRLActionPrediction.from_pretrained(
              model_path,
              torch_dtype=torch_dtype,
              hidden_size=cfg.hidden_size,
              unnorm_key=cfg.unnorm_key,
              vh_mode=cfg.vh_mode,
              action_dim=cfg.action_token_len,
              attn_implementation=cfg.attn_implementation,
              low_cpu_mem_usage=cfg.low_cpu_mem_usage,
              trust_remote_code=cfg.trust_remote_code,
          )

      if cfg.is_lora:
          # Add LoRA support here
          pass

      return model

4. Environment Wrapper Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Add `wrap_observation_your_model` and `wrap_chunk_actions_your_model` in `rlinf/envs/your_env_wrapper.py`. 
These convert simulator data to model inputs and model outputs back to simulator actions in the required shape and device.

.. code-block:: python

  def wrap_observation_your_model(raw_obs, input_processor, model, precision):
      images = raw_obs["image"].permute(0,3,1,2).to(device="cuda:0", dtype=precision)
      prompts = [
          f"In: What action should the robot take to {t.lower()}?\nOut: "
          for t in raw_obs["task_description"]
      ]
      inputs = input_processor(
          prompts,
          images,
          padding="max_length",
          max_length=model.max_prompt_length
      ).to(device="cuda:0", dtype=precision)
      return inputs

  def wrap_chunk_actions_your_model(chunk_tokens, model, sim_precision):
      tokens = chunk_tokens.cpu().numpy()
      actions = []
      for step in range(tokens.shape[1]):
          decoded = wrap_single_step_actions(tokens[:, step], model)
          formatted = format_actions_for_simulator(decoded, model)
          actions.append(formatted)
      return torch.stack(actions, dim=1).to(device="cuda").to(sim_precision)

5. Worker Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Update `get_observation_action_wrapper_func` in `rlinf/workers/generation/hf/multi_step_worker.py` 
to return your wrappers when `cfg.env.train.wrapper` and `cfg.model_name` match. 

.. code-block:: python

  def get_observation_action_wrapper_func(cfg):
      if cfg.env.train.wrapper == "your_env":
          if cfg.actor.model.model_name == "your_model_name":
              from rlinf.envs.your_env_wrapper import (
                  wrap_observation_your_model,
                  wrap_chunk_actions_your_model,
              )
              return wrap_observation_your_model, wrap_chunk_actions_your_model
          raise NotImplementedError
      raise NotImplementedError

6. Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Create `examples/embodiment/config/your_config.yaml` with fields like `model_name`, `action_token_len`, and `precision`. 
This template exposes your model’s hyperparameters for easy experiment setup.

.. code-block:: yaml

  model:
    model_name: "your_model_name"
    action_token_len: 7
    action_chunks_len: 1
    unnorm_key: your_action_key
    micro_batch_size: 1
    val_micro_batch_size: 8
    precision: "bf16"
    vocab_size: 32000
    hidden_size: 4096
    vh_mode: "a0"
    image_size: [224, 224]
    is_lora: False
    num_images_in_input: 1
    attn_implementation: "flash_attention_2"
    low_cpu_mem_usage: True
    trust_remote_code: True
