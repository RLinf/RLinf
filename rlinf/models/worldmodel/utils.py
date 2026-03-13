# Copyright (2024) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
import math
import torch.distributed as dist
import numpy as np
import base64
import glob

from io import BytesIO
from PIL import Image
import uuid
import time
import os
from openai import OpenAI


def remote_server_reward(task_text, vllm_model, video_path, base_url="http://172.31.208.6:8000/v1"):
    """
    请求远程 Qwen3-VL 服务器评估机器人任务。
    返回模型原始生成的字符串，以便后续进行匹配。

    Args:
        task_text: 任务描述文本
        video_path: 视频路径，可以是 mp4 文件或包含帧图片的文件夹目录
        base_url: 远程服务器地址

    Returns:
        模型生成的原始文本响应
    """
    client = OpenAI(
        api_key="EMPTY",
        base_url=base_url,
        timeout=3600
    )

    text_template = (
        f"You will be shown a video. Determine if the robot succeeds at {task_text} Output the final answer strictly as one of : Success or Failure."
    )

    # 判断 video_path 是文件还是目录
    if os.path.isfile(video_path) and video_path.endswith('.mp4'):
        # mp4 文件：使用 video_url 类型
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": "file://" + video_path},
                    },
                    {"type": "text", "text": text_template},
                ],
            }
        ]
    elif os.path.isdir(video_path):
        # 文件夹目录：读取帧图片并转换为 base64
        frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
        if not frames:
            # 如果没有 jpg，尝试 png
            frames = sorted(glob.glob(os.path.join(video_path, "*.png")))

        content = []
        for f in frames:
            with open(f, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        content.append({"type": "text", "text": text_template})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]
    else:
        raise ValueError(f"video_path 必须是 mp4 文件或包含帧图片的文件夹目录: {video_path}")

    # 直接执行调用，不捕获异常
    response = client.chat.completions.create(
        model=vllm_model,
        messages=messages,
        max_tokens=50,
        temperature=0.0
    )

    # 返回原始文本（去空格）
    return response.choices[0].message.content.strip()


def generate_video_path(save_as='images',output_dir="temp_videos"):
    os.makedirs(output_dir, exist_ok=True)

    # 方案 A: UUID (最安全，适合并发)
    unique_id = uuid.uuid4().hex

    # 方案 B: 时间戳 + 随机数 (可读性好)
    # timestamp = time.strftime("%Y%m%d_%H%M%S")
    # unique_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
    if save_as == 'images':
        return os.path.abspath(os.path.join(output_dir, f"v_{unique_id}"))
    return os.path.abspath(os.path.join(output_dir, f"v_{unique_id}.mp4"))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate(
            [np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def b64_2_img(data: str):
    image_b64 = base64.b64decode(data)
    img = Image.open(BytesIO(image_b64)).convert("RGB")
    return img


def get_continuous_action(d_acts, c_act_max, c_act_min, n_bins):
    c_act_max = c_act_max.to(d_acts.device)
    c_act_min = c_act_min.to(d_acts.device)
    c_acts = d_acts / (n_bins - 1) * (c_act_max - c_act_min) + c_act_min
    return c_acts


def alpha2rotm(a):
    """Alpha euler angle to rotation matrix."""
    rotm = np.array([
        [1, 0, 0],
        [0, np.cos(a), -np.sin(a)],
        [0, np.sin(a),  np.cos(a)]
    ])
    return rotm


def beta2rotm(b):
    """Beta euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(b), 0, np.sin(b)],
        [0, 1, 0],
        [-np.sin(b), 0, np.cos(b)]
    ])
    return rotm


def gamma2rotm(c):
    """Gamma euler angle to rotation matrix."""
    rotm = np.array([
        [np.cos(c), -np.sin(c), 0],
        [np.sin(c),  np.cos(c), 0],
        [0, 0, 1]
    ])
    return rotm


def euler2rotm(euler_angles):
    """Euler angle (ZYX) to rotation matrix."""
    alpha = euler_angles[0]
    beta = euler_angles[1]
    gamma = euler_angles[2]

    rotm_a = alpha2rotm(alpha)
    rotm_b = beta2rotm(beta)
    rotm_c = gamma2rotm(gamma)

    rotm = rotm_c @ rotm_b @ rotm_a

    return rotm


def isRotm(R):
    # Checks if a matrix is a valid rotation matrix.
    # Forked from Andy Zeng
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotm2euler(R):
    # Forked from: https://learnopencv.com/rotation-matrix-to-euler-angles/
    # R = Rz * Ry * Rx
    assert isRotm(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    # (-pi , pi]
    while x > np.pi:
        x -= (2 * np.pi)
    while x <= -np.pi:
        x += (2 * np.pi)
    while y > np.pi:
        y -= (2 * np.pi)
    while y <= -np.pi:
        y += (2 * np.pi)
    while z > np.pi:
        z -= (2 * np.pi)
    while z <= -np.pi:
        z += (2 * np.pi)
    return np.array([x, y, z])


def get_converted_fp32_paths(deepspeed_ckpt_path):
    deepspeed_ckpt_path = deepspeed_ckpt_path.rstrip('/')
    ckpt_dir = os.path.dirname(deepspeed_ckpt_path)
    ckpt_name = os.path.basename(deepspeed_ckpt_path)
    fp32_ckpt_name = f"{ckpt_name}.fp32.pt"
    converted_path = os.path.join(ckpt_dir, fp32_ckpt_name)
    return converted_path


def quat2rotm(quat):
    """Quaternion to rotation matrix.

    Args:
        quat (4, numpy array): quaternion x, y, z, w
    Returns:
        rotm (3x3 numpy array): rotation matrix
    """
    w = quat[3]
    x = quat[0]
    y = quat[1]
    z = quat[2]

    s = w*w + x*x + y*y + z*z

    rotm = np.array([[1-2*(y*y+z*z)/s, 2*(x*y-z*w)/s,   2*(x*z+y*w)/s],
                     [2*(x*y+z*w)/s,   1-2*(x*x+z*z)/s, 2*(y*z-x*w)/s],
                     [2*(x*z-y*w)/s,   2*(y*z+x*w)/s,   1-2*(x*x+y*y)/s]
                     ])

    return rotm

def save_video_frame(video_frames: list, save_dir: str, fps: int = 8):
    """
    将视频帧保存为带数字编号的jpg文件。
    
    Args:
        video_frames: 视频帧列表，每帧可以是numpy数组或PIL Image
        save_dir: 保存目录
        fps: 帧率参数（保留以保持接口兼容，实际不使用）
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, frame in enumerate(video_frames):
        # 转换为PIL Image
        if isinstance(frame, np.ndarray):
            # 确保维度正确并转换为uint8
            if frame.shape[0] == 3 and len(frame.shape) == 3:  # CHW格式
                frame = frame.transpose(1, 2, 0)
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
            frame_pil = Image.fromarray(frame)
        elif isinstance(frame, torch.Tensor):
            frame_np = frame.cpu().numpy()
            if frame_np.shape[0] == 3:
                frame_np = frame_np.transpose(1, 2, 0)
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
            frame_pil = Image.fromarray(frame_np)
        else:
            frame_pil = frame
        
        # 保存为带数字编号的jpg，格式：0001.jpg, 0002.jpg, ...
        frame_path = os.path.join(save_dir, f"{i+1:04d}.jpg")
        frame_pil.save(frame_path, quality=95)