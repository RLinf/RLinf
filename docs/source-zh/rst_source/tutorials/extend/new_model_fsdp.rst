使用 FSDP+HuggingFace 添加新模型
========================================

本文档重点介绍如何使用 HuggingFace Transformers 库与 PyTorch FSDP（Fully Sharded Data Parallel，全分片数据并行）  
来训练和生成模型。它支持 HuggingFace 中实现的任意模型，只要兼容 PyTorch 即可。  
作为示例，本节将提供一个逐步的操作流程，展示如何按照 OpenVLA 模式将一个新的 HuggingFace 模型集成到 RLinf 中。  

前置条件
-------------

* 熟悉 **HuggingFace Transformers 库**  
* 理解 **RLinf** 框架架构  
* 掌握 **PyTorch** 与分布式训练知识  

逐步实现
---------------------------

1. 模型配置与注册
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

编辑 `rlinf/models/__init__.py`，扩展 `get_model_config_and_processor`。  
这会注册你模型的 `Config`、`ImageProcessor` 和 `Processor`，使 RLinf 可以按名称加载它们并自动完成预处理。  

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

2. 模型实现
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `rlinf/models/embodiment/your_model_action_model.py` 中创建你的类，并继承自 HuggingFace 基类。  
实现 `predict_action_batch`，用于封装生成、解码和可选的数值计算，将 RL 逻辑保持在模型内部。  

.. code-block:: python

  from transformers import YourBaseModel
  from rlinf.models.embodiment.modules.value_head import ValueHead

  class YourModelForRLActionPrediction(YourBaseModel):
      def __init__(self, config, hidden_size, unnorm_key, action_dim):
          super().__init__(config)
          self._init_logits_processor()
          action_norm_stats = self.get_action_stats(unnorm_key)
          self.min_action = np.array(action_norm_stats["q01"])
          self.max_action = np.array(action_norm_stats["q99"])
          self.value_head = ValueHead(hidden_size)
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

1. 模型加载
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 `rlinf/models/__init__.py` 中的 `get_model`，当 `cfg.model_name` 匹配时调用 `from_pretrained` 加载你的类。  
这能确保检查点加载时保持正确的 dtype、维度和 LoRA hooks。  

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
              action_dim=cfg.action_token_len,
              attn_implementation=cfg.attn_implementation,
              low_cpu_mem_usage=cfg.low_cpu_mem_usage,
              trust_remote_code=cfg.trust_remote_code,
          )

      if cfg.is_lora:
          # 在此添加 LoRA 支持
          pass

      return model

1. 环境封装函数
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `rlinf/envs/your_env_wrapper.py` 中添加 `wrap_observation_your_model` 和 `wrap_chunk_actions_your_model`。  
这些函数用于将模拟器数据转换为模型输入，并将模型输出转换为模拟器动作，保持所需的形状和设备。  

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

5. Worker 集成
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

更新 `rlinf/workers/generation/hf/multi_step_worker.py` 中的 `get_observation_action_wrapper_func`，  
当 `cfg.env.train.wrapper` 和 `cfg.model_name` 匹配时返回你的封装函数。  

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

6. 配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `examples/embodiment/config/your_config.yaml` 中创建配置文件，  
包含 `model_name`、`action_token_len` 和 `precision` 等字段。  
该模板会暴露你模型的超参数，方便实验设置。  

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
    image_size: [224, 224]
    is_lora: False
    use_wrist_image: False
    attn_implementation: "flash_attention_2"
    low_cpu_mem_usage: True
    trust_remote_code: True
