import dataclasses
import logging
import socket
import asyncio
import os
import http
import logging
import time
import traceback
import torch
import tyro
from einops import rearrange
import datetime

from groot.vla.model.n1_5.sim_policy import GrootSimPolicy
from groot.vla.data.schema import EmbodimentTag
import imageio
import numpy as np

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from rlinf.models.embodiment.dreamzero.dreamzero_policy import (
    DreamZeroForRLActionPrediction,
)


class DreamZeroPolicy:
    def __init__(self, model_path: str, video_output_dir: str):
        self.policy = DreamZeroForRLActionPrediction(model_path=model_path, video_output_dir=video_output_dir)
    def infer(self, obs: dict):
        """Infer actions from observations.
        
        Args:
            obs: Observations from the environment.
        Returns:
            Actions from the policy.
        """

        actions = self.policy.eval(obs)
        print("--------------from infer: actions------------------")
        print(actions)
        print("--------------------------------")
        return actions
        
    def _reset_state(self, save_video: bool = False):
        self.policy._reset_state(save_video=save_video)

def setup_policy(model_path: str, video_output_dir: str):
    return DreamZeroPolicy(model_path, video_output_dir)

__all__ = ["setup_policy"]