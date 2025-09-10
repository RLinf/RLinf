import contextlib
from typing import Any, Dict, Literal, Tuple
import torch
from torch import Tensor
from typing_extensions import override
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
import math
import pickle

class Pi0ForRLActionPrediction(PI0Policy):
    """Pi0 model for reinforcement learning action prediction.
    
    This is a template class that defines the interfaces needed for RL training.
    You need to implement all the methods marked with 'TODO: Implement'.
    """

    def __init__(
            self,
            config,
            dataset_stats: dict[str, dict[str, Tensor]] | None = None
        ):
            super().__init__(config, dataset_stats)
    # @property 
    # def _no_split_modules(self) -> list[str]:
    #     """List of modules that should not be split by FSDP."""
    #     # return ['PaliGemmaMultiModalProjector', 'GemmaRMSNorm', 'GemmaAttention', 'GemmaMLP', 'Linear', 'Embedding']
    #     return ['PaliGemmaForConditionalGeneration', 'Normalize', 'Unnormalize']

    def prepare_input(self, batch):
        batch = self.normalize_inputs(batch)
        # prepare
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        processed_input = {
            "images": images,
            "img_masks": img_masks,
            "state": state,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks
        }
        # return images, img_masks, state, lang_tokens, lang_masks
        return processed_input

    @override
    def prepare_language(self, batch):
        if 'lang_tokens' in batch.keys() and 'lang_masks' in batch.keys():
            lang_tokens = batch['lang_tokens']
            lang_masks = batch['lang_masks']
            return lang_tokens, lang_masks
        """Tokenize the text input"""
        device = batch["observation.state"].device
        tasks = batch["task"]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def preprocess_for_train(self, data):
        return data

    @torch.no_grad
    def predict_actions(self, processed_obs, sample_mode: Literal["ode", "sde"], denoise_steps: int):
        """
        Predict actions given environment observations. If the initial noise is not provided, it will be sampled from the model.
        Return samples:
            final_actions: [env_num,act_steps,action_dim] [5,5,7]
            chains: [env_num,denoise_steps + 1,act_steps,action_dim] [5,11,5,7]
            log_probs: [env_num,denoise_steps,act_steps,action_dim] [5,10,5,7]
        """
        # TODO: 这个需要放在这里吗？
        # processed_obs = self.prepare_input(batch)
        images = processed_obs["images"]
        img_masks = processed_obs["img_masks"]
        lang_tokens = processed_obs["lang_tokens"]
        lang_masks = processed_obs["lang_masks"]
        state = processed_obs["state"]
        # sample
        samples = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, sample_mode, denoise_steps)
        # decouple the samples, actions: [batch_size, n_action_steps, action_dim]
        actions, chains, log_probs, values = samples 
        # action dimension unpad # ! chains are not unpaded here
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        log_probs = log_probs[:,:,:,:original_action_dim]
        actions = self.unnormalize_outputs({"action": actions})["action"]
        # action step
        actions = actions[:, :self.config.act_steps]
        log_probs = log_probs[:,:, :self.config.act_steps]
        return actions, chains, log_probs, values
    
    # todo stop here
    def select_multi_actions(self, processed_obs: dict[str, Tensor], rollout_stage: bool = False):
        if rollout_stage:
            # result : actions, chains, log_probs, values, (depend on output_lang_tokens : lang_tokens, lang_masks)
            result = self.predict_actions(processed_obs,"sde",self.config.num_steps)
            return result
        else:
            result = self.predict_actions(processed_obs,"ode",self.config.num_steps_eval)
            return result

    
    def get_log_prob_value(self,batch,chains_pre,chains_next,denoise_inds):
        # prepare input
        processed_obs = self.prepare_input(batch)
        images = processed_obs["images"]
        img_masks = processed_obs["img_masks"]
        lang_tokens = processed_obs["lang_tokens"]
        lang_masks = processed_obs["lang_masks"]
        state = processed_obs["state"]
        # chains
        log_probs,value = self.model.get_log_prob_value(images,img_masks,lang_tokens,lang_masks,state,chains_pre,chains_next,denoise_inds)
        original_action_dim = self.config.action_feature.shape[0]
        log_probs = log_probs[:,:,:original_action_dim]
        log_probs = log_probs[:,:self.config.act_steps]
        return log_probs,value

    def forward(
        self, 
        data: dict[str, Tensor], 
        mode: Literal["train", "eval", "compute_logprob"] = "predict",
    ) -> Dict[str, Any]:
        """
        Unified forward function for Pi0 model that handles both prediction and logprob computation.
        
        Args:
            batch: Input batch dictionary containing observations (predict mode) or training data (compute_logprob mode)
            mode: "predict" for action prediction, "compute_logprob" for logprob computation
            sample_mode: Sampling mode for prediction ("ode" or "sde") - not used in current predict implementation
            rollout_stage: Whether in rollout stage - not used in current predict implementation  
            output_lang_tokens: Whether to output language tokens in prediction mode
            requires_grad: Whether to enable gradients (auto-detected if None)
        
        Returns:
            Dictionary containing:
            - For predict mode: actions, chains, log_probs, values, (optionally lang_tokens, lang_masks)
            - For compute_logprob mode: token_level_entropy, token_level_log_probs, action_level_logprobs
        """
        if mode == "train" or mode == "eval":
            # Action prediction mode using _generate_one_step
            # batch contains observation data
            batch_output = self._generate_one_step(data, mode)
            # Extract components from batch_output
            actions = batch_output["action"]
            chains = batch_output["chains"]
            log_probs = batch_output["prev_logprobs"]
            values = batch_output["values"]
            denoise_inds = batch_output["denoise_inds"]
            # for pi0 ppo value prediction
            # TODO: ppo
            values = torch.mean(values,dim = -1,keepdim = True)
            # values = values[:,2:3]
            return {
                "actions": actions,
                "chains": chains,
                "prev_logprobs": log_probs,
                "prev_values": values,
                "denoise_inds": denoise_inds
            }
            
        elif mode == "compute_logprob":
            result = self._forward_micro_batch(data)
            return {
                "entropy": result["entropy"],
                "logprobs": result["logprobs"],
                "values": result["values"]  
            }
        
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'rollout', 'eval' or 'compute_logprob'")


    # Pi0 do not need this function, so we leave it empty
    def setup_params(self, model_config, cfg):
        self.action_dim = cfg.actor.model.action_dim
        self.max_prompt_length = cfg.runner.max_prompt_length

    def _generate_one_step(self, obs: dict, mode: Literal["train", "eval"]) -> Dict[str, Any]:  
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        if mode == "train":
            rollout_stage = True
        else:
            rollout_stage = False
        
        actions, chains, token_level_log_probs, values = self.select_multi_actions(obs, rollout_stage=rollout_stage)
        actions = actions.to(torch.float32).detach().cpu().numpy()
        # Prepare PI0 diffusion chain data for PPO training
        batch_size = actions.shape[0]
        device = token_level_log_probs.device

        # Generate denoise indices for chains
        # chains shape: [batch_size, denoise_steps + 1, act_steps, action_dim]
        # log_probs shape: [batch_size, denoise_steps, act_steps, action_dim]
        denoise_steps = chains.shape[1] - 1  # chains includes initial noise, so subtract 1
        denoise_inds = torch.arange(denoise_steps, device=device).unsqueeze(0).repeat(batch_size, 1)
        batch_output = {
            "action": actions,
            "chains": chains,
            "denoise_inds": denoise_inds,
            "prev_logprobs": token_level_log_probs,
            "values": values,
        }
        return batch_output
    #! has some accuracy loss here
    def _forward_micro_batch(
        self, 
        data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # TODO: DEBUG, chenk modify here
        pi0_batch = {}
        pi0_obs_keys = ["observation.images.image", "observation.images.wrist_image", "observation.state",  "lang_tokens", "lang_masks", "task"]
        for key in pi0_obs_keys:
            if key in data.keys():
                pi0_batch[key] = data[key]

        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # denoise index
        length = data['prev_logprobs'].size(0)
        chains = data['chains']  # [length, denoise_steps + 1, act_steps, action_dim]
        denoise_inds = data['denoise_inds']
        chains_pre = chains[torch.arange(length), denoise_inds]  # [length, act_steps, action_dim]
        chains_next = chains[torch.arange(length), denoise_inds + 1]   # [length, act_steps, action_dim]
        # logprob calculation
        token_level_log_probs, values = self.get_log_prob_value(
            pi0_batch, chains_pre, chains_next, denoise_inds
        )
            # todo: debug log-prob
            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            #     loaded_dict = torch.load('debug_variables_1_0.pt', map_location='cuda')  # 加载到GPU
            #     pi0_batch = loaded_dict['batch']
            #     chains = loaded_dict['chains']
            #     denoise_inds = loaded_dict['denoise_idx'][torch.arange(length)]
            #     chains_pre = chains[torch.arange(length), denoise_inds]  # [length, act_steps, action_dim]
            #     chains_next = chains[torch.arange(length), denoise_inds + 1]   # [length, act_steps, action_dim]
            #     ref_log_probs = loaded_dict['ref_log_probs'][torch.arange(length)]
            #     token_level_log_probs, _ = self.get_log_prob_value(
            #         pi0_batch, chains_pre, chains_next, denoise_inds
            #     )
            #     token_level_log_probs = token_level_log_probs.sum(dim=-1)
            #     print(torch.exp(token_level_log_probs - data['prev_logprobs'][torch.arange(length), denoise_inds]))
            # todo: debug log-prob 
        return {"entropy": None, "logprobs": token_level_log_probs, "values": values}


    def process_tensor(self, tensor, pad_id):
        """Process tensor by removing padding (Pi0 version).
        
        Args:
            tensor: Input tensor with padding
            pad_id: Padding token id (not used in Pi0 but kept for compatibility)
            
        Returns:
            Tuple of (processed_tensor, valid_length)
        """
        # Pi0 doesn't use padding in the same way as token-based models
        # Return tensor as-is for compatibility
        if tensor is None:
            return None, 0
        return tensor, tensor.shape[-1] if len(tensor.shape) > 0 else 0
