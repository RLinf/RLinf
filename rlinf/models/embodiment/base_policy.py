import torch.nn as nn

class BasePolicy(nn.Module):
    def preprocess_env_obs(self, env_obs):
        return env_obs
    
    def forward(
        self, forward_type="default_forward", **kwargs
    ):
        if forward_type == "sac_forward":
            return self.sac_forward(**kwargs)
        elif forward_type == "sac_q_forward":
            return self.get_q_values(**kwargs)
        elif forward_type == "default_forward":
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError
        
    def sac_forward(self, **kwargs):
        raise NotImplementedError
    
    def get_q_values(self, **kwargs):
        raise NotImplementedError
    
    def default_forward(self, **kwargs):
        raise NotImplementedError