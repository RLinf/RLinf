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
      if cfg.model.model_type == "your_model_type":
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
          if hasattr(self, "value_head"):
              values = self.value_head(generated.hidden_states)
          else:
              values = torch.zeros_like(logits[..., :1])
          return actions, sequences, logits, values

3. 模型加载
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改 `rlinf/models/__init__.py` 中的 `get_model`，当 `cfg.model_type` 匹配时调用 `from_pretrained` 加载你的类。  
这能确保检查点加载时保持正确的 dtype、维度和 LoRA hooks。  

.. code-block:: python

  def get_model(cfg: DictConfig, override_config_kwargs=None):
      torch_dtype = torch_dtype_from_precision(cfg.precision)
      model_path = cfg.model_path
      if cfg.model_type == "your_model_type":
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

4. 配置文件
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

在 `examples/embodiment/config/your_config.yaml` 中创建配置文件，  
包含 `model_type`、`action_token_len` 和 `precision` 等字段。  
该模板会暴露你模型的超参数，方便实验设置。  

.. code-block:: yaml

  model:
    model_type: "your_model_type"
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
    attn_implementation: "flash_attention_2"
    low_cpu_mem_usage: True
    trust_remote_code: True
