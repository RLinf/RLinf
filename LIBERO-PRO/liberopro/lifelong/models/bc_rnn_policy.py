import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn

from einops import rearrange, repeat
from libero.lifelong.models.modules.rgb_modules import *
from libero.lifelong.models.modules.language_modules import *
from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.policy_head import *


###############################################################################
#
# A model handling extra input modalities besides images at time t.
#
###############################################################################


class ExtraModalities:
    def __init__(
        self,
        use_joint=False,
        use_gripper=False,
        use_ee=False,
        extra_hidden_size=64,
        extra_embedding_size=32,
    ):

        self.use_joint = use_joint
        self.use_gripper = use_gripper
        self.use_ee = use_ee
        self.extra_embedding_size = extra_embedding_size

        joint_states_dim = 7
        gripper_states_dim = 2
        ee_dim = 3

        self.extra_low_level_feature_dim = (
            int(use_joint) * joint_states_dim
            + int(use_gripper) * gripper_states_dim
            + int(use_ee) * ee_dim
        )
        assert self.extra_low_level_feature_dim > 0, "[error] no extra information"

    def __call__(self, obs_dict):
        """
        obs_dict: {
            (optional) joint_stats: (B, T, 7),
            (optional) gripper_states: (B, T, 2),
            (optional) ee: (B, T, 3)
        }
        map above to a latent vector of shape (B, T, H)
        """
        tensor_list = []
        if self.use_joint:
            tensor_list.append(obs_dict["joint_states"])
        if self.use_gripper:
            tensor_list.append(obs_dict["gripper_states"])
        if self.use_ee:
            tensor_list.append(obs_dict["ee_states"])
        x = torch.cat(tensor_list, dim=-1)
        return x

    def output_shape(self, input_shape, shape_meta):
        return (self.extra_low_level_feature_dim,)


###############################################################################
#
# A RNN policy
#
###############################################################################


class BCRNNPolicy(BasePolicy):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, cfg, shape_meta):
        super().__init__(cfg, shape_meta)
        policy_cfg = cfg.policy

        ### 1. encode image
        rnn_input_size = 0
        image_embed_size = policy_cfg.image_embed_size
        self.image_encoders = {}
        for name in shape_meta["all_shapes"].keys():
            if "rgb" in name or "depth" in name:
                kwargs = policy_cfg.image_encoder.network_kwargs
                kwargs.input_shape = shape_meta["all_shapes"][name]
                kwargs.output_size = image_embed_size
                kwargs.language_dim = (
                    policy_cfg.language_encoder.network_kwargs.input_size
                )
                self.image_encoders[name] = {
                    "input_shape": shape_meta["all_shapes"][name],
                    "encoder": eval(policy_cfg.image_encoder.network)(**kwargs),
                }
                rnn_input_size += image_embed_size
        self.encoders = nn.ModuleList(
            [x["encoder"] for x in self.image_encoders.values()]
        )

        ### 2. encode language
        text_embed_size = policy_cfg.text_embed_size
        policy_cfg.language_encoder.network_kwargs.output_size = text_embed_size
        self.language_encoder = eval(policy_cfg.language_encoder.network)(
            **policy_cfg.language_encoder.network_kwargs
        )
        rnn_input_size += text_embed_size

        ### 3. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = ExtraModalities(
            use_joint=cfg.data.use_joint,
            use_gripper=cfg.data.use_gripper,
            use_ee=cfg.data.use_ee,
        )
        rnn_input_size += self.extra_encoder.extra_low_level_feature_dim

        self.rnn = nn.LSTM(
            input_size=rnn_input_size,
            hidden_size=policy_cfg.rnn_hidden_size,
            num_layers=policy_cfg.rnn_num_layers,
            batch_first=True,
            dropout=policy_cfg.rnn_dropout,
            bidirectional=policy_cfg.rnn_bidirectional,
        )

        ### 4. use policy head to output action
        self.D = 2 if policy_cfg.rnn_bidirectional else 1
        policy_head_kwargs = policy_cfg.policy_head.network_kwargs
        policy_head_kwargs.input_size = self.D * policy_cfg.rnn_hidden_size
        policy_head_kwargs.output_size = shape_meta["ac_dim"]

        self.policy_head = eval(policy_cfg.policy_head.network)(
            **policy_cfg.policy_head.loss_kwargs,
            **policy_cfg.policy_head.network_kwargs
        )
        self.eval_h0 = None
        self.eval_c0 = None

    def forward(self, data, train_mode=True):
        # 1. encode image
        encoded = []
        for img_name in self.image_encoders.keys():
            x = data["obs"][img_name]
            B, T, C, H, W = x.shape
            e = self.image_encoders[img_name]["encoder"](
                x.reshape(B * T, C, H, W),
                langs=data["task_emb"]
                .reshape(B, 1, -1)
                .repeat(1, T, 1)
                .reshape(B * T, -1),
            ).view(B, T, -1)
            encoded.append(e)

        # 2. add joint states, gripper info, etc.
        encoded.append(self.extra_encoder(data["obs"]))  # add (B, T, H_extra)
        encoded = torch.cat(encoded, -1)  # (B, T, H_all)

        # 3. language encoding
        lang_h = self.language_encoder(data)  # (B, H)
        encoded = torch.cat(
            [encoded, lang_h.unsqueeze(1).expand(-1, encoded.shape[1], -1)], dim=-1
        )

        # 4. apply temporal rnn
        if train_mode:
            h0 = torch.zeros(
                self.D * self.cfg.policy.rnn_num_layers,
                encoded.shape[0],
                self.cfg.policy.rnn_hidden_size,
            ).to(self.device)
            c0 = torch.zeros(
                self.D * self.cfg.policy.rnn_num_layers,
                encoded.shape[0],
                self.cfg.policy.rnn_hidden_size,
            ).to(self.device)
            output, (hn, cn) = self.rnn(encoded, (h0, c0))
        else:
            if self.eval_h0 is None:
                self.eval_h0 = torch.zeros(
                    self.D * self.cfg.policy.rnn_num_layers,
                    encoded.shape[0],
                    self.cfg.policy.rnn_hidden_size,
                ).to(self.device)
                self.eval_c0 = torch.zeros(
                    self.D * self.cfg.policy.rnn_num_layers,
                    encoded.shape[0],
                    self.cfg.policy.rnn_hidden_size,
                ).to(self.device)
            output, (h1, c1) = self.rnn(encoded, (self.eval_h0, self.eval_c0))
            self.eval_h0 = h1.detach()
            self.eval_c0 = c1.detach()

        dist = self.policy_head(output)
        return dist

    def get_action(self, data):
        self.eval()
        data = self.preprocess_input(data, train_mode=False)
        with torch.no_grad():
            dist = self.forward(data)
        action = dist.sample().detach().cpu()
        return action.view(action.shape[0], -1).numpy()

    def reset(self):
        self.eval_h0 = None
        self.eval_c0 = None
