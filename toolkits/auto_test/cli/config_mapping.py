# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""配置映射：任务类型和资源环境的映射关系"""

from typing import Optional

# 任务类型到环境配置的映射
TASK_TYPE_MAPPING = {
    "pretrain": {
        "image_id": "im-db5yl3csf7ulsudj",  # TODO: 替换为实际镜像 ID
        "framework_id": "fw-c6q6a7sfyhoeb5xi",  # TODO: 替换为实际框架 ID
        "pool_id": "po-c77m6nj3hteszywp",  # 宁夏B的资源池ID
    },
    "supervised_learning": {
        "image_id": "im-db5yl3csf7ulsudj",  # TODO: 替换为实际镜像 ID
        "framework_id": "fw-c6q6a7sfyhoeb5xi",  # TODO: 替换为实际框架 ID
        "pool_id": "po-c77m6nj3hteszywp",  # 宁夏B的资源池ID
    },
    "reinforcement_learning": {
        "image_id": "im-db5yl3csf7ulsudj",  # TODO: 替换为实际镜像 ID
        "framework_id": "fw-c6q6a7sfyhoeb5xi",  # TODO: 替换为实际框架 ID (这里是ray)
        "pool_id": "po-c77m6nj3hteszywp",  # 宁夏B的资源池ID
    },
}

IMAGE_NAME_MAPPING = {"behavior": "im-db5yl3csf7ulsudj"}

# 资源可用区到资源规格和存储卷的映射
POOL_MAPPING = {
    "po-c77m6nj3hteszywp": {
        "load_spec_id": "ls-c7ankietdy2zcogp",  # 包年包月资源使用 load_spec_id
        "resource_spec_id": "rs-dah77zmbxylz7qkd",  # Spot 资源使用 resource_spec_id
        "mount": {
            "path": "/mnt/public",  # 默认挂载路径
            "volume_id": "vo-c77m6nj4qru75pqs",  # 存储卷 ID
            "rw_setting": "can_write",  # 默认只读
        },
    },
    "po-dals27bs2sx4gj2a": {
        "load_spec_id": "ls-dajcsoxf7ze25vxi",  # 包年包月资源使用 load_spec_id
        "resource_spec_id": "rs-da4xgqypkdox3plj",  # Spot 资源使用 resource_spec_id
        "mount": {
            "path": "/mnt/public/hzl",  # 默认挂载路径
            "volume_id": "vo-dals27bupd4mba22",  # 存储卷 ID
            "rw_setting": "can_write",  # 默认只读
        },
    },
    "po-da73jexmoe4rfgej": {
        "resource_spec_id": "rs-dba3vcqqr2b2jfko",  # 浙江D 当前仅接入 Spot 规格
    },
}

# 资源区域映射
REGION_RESOURCE_MAPPING = {
    "ningxiaB": {
        "region_id": "rg-c7anfhyiq3cf24s7",
        "pool_id": "po-c77m6nj3hteszywp",
        "aicoder_mount_path": "/mnt/public",
        "specs": {
            0: {
                8: "ls-c7ankietdy2zcogp",
                4: "ls-c7ankietsjbjokv4",
                2: "ls-c7ankieuaxxpj52i",
                1: "ls-c7ankieupt5pdkim",
            },
            1: {
                8: "rs-dah77zmbxylz7qkd",
                4: "rs-dah77zmeyzctfvih",
                2: "rs-dah77zmhwqforehv",
                1: "rs-dah77zmkuflgc5a6",
            },
        },
    },
    "beijingD": {
        "region_id": "rg-dajcq7thhraidy6p",
        "pool_id": "po-dals27bs2sx4gj2a",
        "aicoder_mount_path": "/mnt/public",
        "specs": {
            0: {
                8: "ls-dajcsoxf7ze25vxi",
                4: "ls-dajcsoxho4jtz5uw",
                2: "ls-dajcsoxi5q4h7sk3",
                1: "ls-dajcsoxkmd4ngfpo",
            },
            1: {
                8: "rs-da4xgqypkdox3plj",
                4: "rs-da4xgqyuujuqty24",
                2: "rs-da4xgqyygkkhwjwp",
                1: "rs-da4xgqy3wpnu6s6z",
            },
        },
    },
    "zhejiangD": {
        "region_id": "rg-da5azznpr4i4fdrx",
        "pool_id": "po-da73jexmoe4rfgej",
        "specs": {
            1: {
                8: "rs-dba3vcqqr2b2jfko",
                4: "rs-dba3vcqtwwetgt2s",
                2: "rs-dba3vcqxfl3icerj",
                1: "rs-dba3vcq2k4po4o5g",
            },
        },
    },
}

REGION_RESOURCE_ALIASES = {
    "ningxiab": "ningxiaB",
    "宁夏b": "ningxiaB",
    "beijingd": "beijingD",
    "北京d": "beijingD",
    "zhejiangd": "zhejiangD",
    "浙江d": "zhejiangD",
}

# AICoder 上传预设
AICODER_UPLOAD_PRESET_MAPPING = {
    name: {
        "upload_method": "aicoder_scp",
        "volume_id": pool_config["mount"]["volume_id"],
        "region_id": config["region_id"],
        "aicoder_mount_path": config["aicoder_mount_path"],
    }
    for name, config in REGION_RESOURCE_MAPPING.items()
    for pool_config in [POOL_MAPPING.get(config["pool_id"], {})]
    if isinstance(pool_config.get("mount"), dict)
    and pool_config["mount"].get("volume_id")
    and config.get("aicoder_mount_path")
}

AICODER_UPLOAD_PRESET_ALIASES = REGION_RESOURCE_ALIASES.copy()

RESOURCE_TYPE_MAPPING = {
    0: 0,
    1: 1,
    "0": 0,
    "1": 1,
    "reserved": 0,
    "reservedpool": 0,
    "annual": 0,
    "包年包月": 0,
    "spot": 1,
    "ondemand": 1,
    "按需": 1,
}

# 默认配置
DEFAULT_CONFIG = {
    "resource_type": 0,  # 0=reserved, 1=spot（默认 reserved）
    "worker_num": 1,
    "rdma_enable": False,
}


def get_task_config(task_type: str) -> dict:
    """
    根据任务类型获取配置

    Args:
        task_type: 任务类型（如 "预训练"、"强化学习训练"）

    Returns:
        包含 image_id, framework_id, pool_id, device_type 的字典

    Raises:
        ValueError: 如果任务类型不存在
    """
    if task_type not in TASK_TYPE_MAPPING:
        available_types = ", ".join(TASK_TYPE_MAPPING.keys())
        raise ValueError(
            f"未知的任务类型: {task_type}。\n可用的任务类型: {available_types}"
        )
    return TASK_TYPE_MAPPING[task_type].copy()


def get_pool_config(pool_id: str) -> dict:
    """
    根据资源池 ID 获取配置

    Args:
        pool_id: 资源池 ID（如 "北京D"、"北京J02"）

    Returns:
        包含 resource_spec_id, volume_id 的字典

    Raises:
        ValueError: 如果资源池不存在
    """
    if pool_id not in POOL_MAPPING:
        available_pools = ", ".join(POOL_MAPPING.keys())
        raise ValueError(f"未知的资源池: {pool_id}。\n可用的资源池: {available_pools}")
    return POOL_MAPPING[pool_id].copy()


def get_full_config(task_type: str) -> dict:
    """
    根据任务类型获取完整配置（合并任务类型和资源池配置）

    Args:
        task_type: 任务类型

    Returns:
        完整的配置字典
    """
    task_config = get_task_config(task_type)
    pool_id = task_config["pool_id"]
    pool_config = get_pool_config(pool_id)

    # 合并配置
    full_config = {
        **DEFAULT_CONFIG,
        **task_config,
        **pool_config,
    }

    return full_config


def list_task_types() -> list:
    """列出所有可用的任务类型"""
    return list(TASK_TYPE_MAPPING.keys())


def list_pools() -> list:
    """列出所有可用的资源池"""
    return list(POOL_MAPPING.keys())


def _normalize_mapped_name(name: str) -> str:
    """规范化映射名称，便于兼容中英文别名。"""
    return "".join(ch for ch in name.strip() if ch.isalnum()).lower()


def normalize_resource_type(resource_type) -> int:
    """将资源类型规范化为 API 使用的整数值。"""
    if resource_type in {0, 1}:
        return int(resource_type)

    if isinstance(resource_type, str):
        normalized_name = _normalize_mapped_name(resource_type)
        mapped_value = RESOURCE_TYPE_MAPPING.get(normalized_name)
        if mapped_value is not None:
            return mapped_value

    raise ValueError(
        f"未知的资源类型: {resource_type}。\n"
        "可用的值: reserved, spot, 0, 1（兼容别名: 包年包月, 按需）"
    )


def get_resource_region_name(training_project: dict) -> Optional[str]:
    """从训练配置中提取资源区域选择。"""
    for field_name in ("resource_region", "region"):
        field_value = training_project.get(field_name)
        if field_value:
            return str(field_value)

    compute_config = training_project.get("compute", {})
    if isinstance(compute_config, dict):
        for field_name in ("region", "cluster_name"):
            field_value = compute_config.get(field_name)
            if field_value:
                return str(field_value)

    return None


def normalize_resource_gpu_num(resource_gpu_num) -> Optional[int]:
    """将卡数规范化为正整数。"""
    if resource_gpu_num in {None, ""}:
        return None

    try:
        normalized_value = int(resource_gpu_num)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"未知的卡数配置: {resource_gpu_num}") from exc

    if normalized_value <= 0:
        raise ValueError(f"卡数必须为正整数，当前值: {resource_gpu_num}")

    return normalized_value


def get_resource_gpu_num(training_project: dict) -> Optional[int]:
    """从训练配置中提取单节点卡数。"""
    for field_name in ("resource_gpu_num", "gpu_num"):
        field_value = training_project.get(field_name)
        if field_value not in {None, ""}:
            return normalize_resource_gpu_num(field_value)

    compute_config = training_project.get("compute", {})
    if isinstance(compute_config, dict):
        for field_name in ("gpu_num", "gpu_count"):
            field_value = compute_config.get(field_name)
            if field_value not in {None, ""}:
                return normalize_resource_gpu_num(field_value)

    return None


def get_region_resource_config(name: str) -> dict:
    """根据区域名获取完整的资源配置。"""
    normalized_name = _normalize_mapped_name(name)
    region_name = REGION_RESOURCE_ALIASES.get(normalized_name)
    if region_name is None:
        available_regions = ", ".join(REGION_RESOURCE_MAPPING.keys())
        raise ValueError(
            f"未知的资源区域: {name}。\n可用的资源区域: {available_regions}"
        )

    region_config = REGION_RESOURCE_MAPPING[region_name].copy()
    pool_id = region_config["pool_id"]
    pool_config = get_pool_config(pool_id)

    return {
        "region_name": region_name,
        **region_config,
        **pool_config,
    }


def apply_region_resource_config(training_project: dict) -> dict:
    """根据区域选择自动覆盖 pool/spec/mount 配置。"""
    region_name = get_resource_region_name(training_project)
    if not region_name:
        return training_project

    normalized_resource_type = normalize_resource_type(
        training_project.get("resource_type", DEFAULT_CONFIG["resource_type"])
    )
    requested_gpu_num = get_resource_gpu_num(training_project)
    region_config = get_region_resource_config(region_name)
    available_specs = region_config.get("specs", {}).get(normalized_resource_type, {})
    spec_field_name = (
        "load_spec_id" if normalized_resource_type == 0 else "resource_spec_id"
    )
    other_spec_field_name = (
        "resource_spec_id" if normalized_resource_type == 0 else "load_spec_id"
    )

    selected_spec_id = region_config.get(spec_field_name)
    if requested_gpu_num is not None:
        selected_spec_id = available_specs.get(requested_gpu_num)
        if not selected_spec_id:
            available_gpu_nums = ", ".join(
                str(gpu_num) for gpu_num in sorted(available_specs.keys())
            )
            resource_type_name = "包年包月" if normalized_resource_type == 0 else "Spot"
            raise ValueError(
                f"资源区域 {region_config['region_name']} 在 {resource_type_name} 模式下不支持 {requested_gpu_num} 卡。\n"
                f"可用卡数: {available_gpu_nums or '无'}"
            )
    elif not selected_spec_id:
        resource_type_name = "包年包月" if normalized_resource_type == 0 else "Spot"
        available_gpu_nums = ", ".join(
            str(gpu_num) for gpu_num in sorted(available_specs.keys())
        )
        raise ValueError(
            f"资源区域 {region_config['region_name']} 当前没有可用的 {resource_type_name} 默认规格配置。\n"
            f"可用卡数: {available_gpu_nums or '无'}"
        )

    training_project["resource_region"] = region_config["region_name"]
    training_project["resource_type"] = normalized_resource_type
    training_project["region_id"] = region_config["region_id"]
    training_project["pool_id"] = region_config["pool_id"]
    if "load_spec_id" in region_config:
        training_project["load_spec_id"] = region_config["load_spec_id"]
    else:
        training_project.pop("load_spec_id", None)
    if "resource_spec_id" in region_config:
        training_project["resource_spec_id"] = region_config["resource_spec_id"]
    else:
        training_project.pop("resource_spec_id", None)
    training_project[spec_field_name] = selected_spec_id
    training_project.pop(other_spec_field_name, None)
    if requested_gpu_num is not None:
        training_project["resource_gpu_num"] = requested_gpu_num

    existing_mount = training_project.get("mount")
    existing_volume_id = training_project.get("volume_id")
    mount_config = region_config.get("mount")
    if isinstance(mount_config, dict):
        training_project["mount"] = mount_config.copy()
        volume_id = mount_config.get("volume_id")
        if volume_id:
            training_project["volume_id"] = volume_id
    else:
        if isinstance(existing_mount, list) and existing_mount:
            training_project["mount"] = [
                item.copy() for item in existing_mount if isinstance(item, dict)
            ]
            if existing_volume_id:
                training_project["volume_id"] = existing_volume_id
            elif training_project["mount"]:
                first_volume_id = training_project["mount"][0].get("volume_id")
                if first_volume_id:
                    training_project["volume_id"] = first_volume_id
        elif existing_volume_id:
            training_project["volume_id"] = existing_volume_id
            training_project.pop("mount", None)
        else:
            training_project.pop("mount", None)
            training_project.pop("volume_id", None)

    return training_project


def get_aicoder_upload_preset(name: str) -> dict:
    """根据简写名称获取 AICoder 上传预设。"""
    normalized_name = _normalize_mapped_name(name)
    preset_name = AICODER_UPLOAD_PRESET_ALIASES.get(normalized_name)
    if preset_name is None or preset_name not in AICODER_UPLOAD_PRESET_MAPPING:
        available_presets = ", ".join(AICODER_UPLOAD_PRESET_MAPPING.keys())
        raise ValueError(
            f"未知的上传预设: {name}。\n可用的上传预设: {available_presets}"
        )
    return AICODER_UPLOAD_PRESET_MAPPING[preset_name].copy()


def list_aicoder_upload_presets() -> list:
    """列出所有可用的 AICoder 上传预设。"""
    return list(AICODER_UPLOAD_PRESET_MAPPING.keys())
