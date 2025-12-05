import copy
import json
import os
import random
from pathlib import Path

import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
from hydra.utils import to_absolute_path
from thop import profile
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, logging


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def safe_device(x, device="cpu"):
    if device == "cpu":
        return x.cpu()
    elif "cuda" in device:
        if torch.cuda.is_available():
            return x.to(device)
        else:
            return x.cpu()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def torch_save_model(model, model_path, cfg=None, previous_masks=None):
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "previous_masks": previous_masks,
        },
        model_path,
    )


def torch_load_model(model_path, map_location=None):
    model_dict = torch.load(model_path, map_location=map_location)
    cfg = None
    if "cfg" in model_dict:
        cfg = model_dict["cfg"]
    if "previous_masks" in model_dict:
        previous_masks = model_dict["previous_masks"]
    return model_dict["state_dict"], cfg, previous_masks


def get_train_test_loader(
    dataset, train_ratio, train_batch_size, test_batch_size, num_workers=(0, 0)
):

    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        num_workers=num_workers[0],
        shuffle=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=num_workers[1],
        shuffle=False,
    )
    return train_dataloader, test_dataloader


def confidence_interval(p, n):
    return 1.96 * np.sqrt(p * (1 - p) / n)


def compute_flops(algo, dataset, cfg):
    model = copy.deepcopy(algo.policy)
    tmp_loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)
    data = next(iter(tmp_loader))
    data = TensorUtils.map_tensor(data, lambda x: safe_device(x, device=cfg.device))
    macs, params = profile(model, inputs=(data,), verbose=False)
    GFLOPs = macs * 2 / 1e9
    MParams = params / 1e6
    del model
    return GFLOPs, MParams


def create_experiment_dir(cfg):
    prefix = "experiments"
    if cfg.pretrain_model_path != "":
        prefix += "_finetune"
    if cfg.data.task_order_index > 0:
        prefix += f"_permute{cfg.data.task_order_index}"
    if cfg.task_embedding_format == "one-hot":
        prefix += f"_onehot"
    if cfg.task_embedding_format == "clip":
        prefix += f"_clip"
    if cfg.task_embedding_format == "gpt2":
        prefix += f"_gpt2"
    if cfg.task_embedding_format == "roberta":
        prefix += f"_roberta"

    experiment_dir = (
        f"./{prefix}/{cfg.benchmark_name}/{cfg.lifelong.algo}/"
        + f"{cfg.policy.policy_type}_seed{cfg.seed}"
    )

    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # look for the most recent run
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run_")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except BaseException:
            pass
    experiment_id += 1

    experiment_dir += f"/run_{experiment_id:03d}"
    cfg.experiment_dir = experiment_dir
    cfg.experiment_name = "_".join(cfg.experiment_dir.split("/")[2:])
    os.makedirs(cfg.experiment_dir, exist_ok=True)
    return True


def get_task_embs(cfg, descriptions):
    logging.set_verbosity_error()

    if cfg.task_embedding_format == "one-hot":
        # offset defaults to 1, if we have pretrained another model, this offset
        # starts from the pretrained number of tasks + 1
        offset = cfg.task_embedding_one_hot_offset
        descriptions = [f"Task {i+offset}" for i in range(len(descriptions))]

    if cfg.task_embedding_format == "bert" or cfg.task_embedding_format == "one-hot":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    elif cfg.task_embedding_format == "gpt2":
        tz = AutoTokenizer.from_pretrained("gpt2")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("gpt2")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["last_hidden_state"].detach()[:, -1]
    elif cfg.task_embedding_format == "clip":
        tz = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model.get_text_features(**tokens).detach()
    elif cfg.task_embedding_format == "roberta":
        tz = AutoTokenizer.from_pretrained("roberta-base")
        tz.pad_token = tz.eos_token
        model = AutoModel.from_pretrained("roberta-base")
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        task_embs = model(**tokens)["pooler_output"].detach()
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    return task_embs
