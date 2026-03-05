import logging
import os
import typing
from typing import Optional, Union

import pandas as pd
import torch
from omegaconf.dictconfig import DictConfig
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from rlinf.data.io_struct import RolloutRequest
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.data_iter_utils import split_list
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.runner_utils import check_progress, local_mkdir_safe
from rlinf.utils.timers import Timer




