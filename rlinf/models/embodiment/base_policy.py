import torch.nn as nn

class BasePolicy(nn.Module):
    def preprocess_env_obs(self, env_obs):
        return env_obs