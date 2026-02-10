# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from typing import Optional, Union

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

libero_type = os.environ.get("LIBERO_TYPE", "standard")

if libero_type == "pro":
    try:
        import liberopro.liberopro.benchmark as benchmark
        from liberopro.liberopro.benchmark import Benchmark
    except ImportError:
        print("[Utils] Warning: LIBERO_TYPE=pro but 'liberopro' not found. Falling back to 'libero'.")
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

elif libero_type == "plus":
    try:
        import liberoplus.liberoplus.benchmark as benchmark
        from liberoplus.liberoplus.benchmark import Benchmark
    except ImportError:
        print("[Utils] Warning: LIBERO_TYPE=plus but 'liberoplus' not found. Falling back to 'libero'.")
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

else:
    try:
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark
    except ImportError:
        try:
            import liberopro.liberopro.benchmark as benchmark
            from liberopro.liberopro.benchmark import Benchmark
        except ImportError:
            try:
                import liberoplus.liberoplus.benchmark as benchmark
                from liberoplus.liberoplus.benchmark import Benchmark
            except ImportError:
                raise ImportError("No valid LIBERO package (libero, liberopro, or liberoplus) found.")


def tile_images(
    images: list[Union[np.ndarray, torch.Tensor]], nrows: int = 1
) -> Union[np.ndarray, torch.Tensor]:
    """
    Copied from maniskill https://github.com/haosulab/ManiSkill
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns.
    """
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns: list[list[Union[np.ndarray, torch.Tensor]]] = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched : 2 + batched]
    columns.append(column)

    total_width = sum(x[0].shape[1 + batched] for x in columns)
    is_torch = isinstance(images[0], torch.Tensor) if torch is not None else False

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    
    output_image = torch.zeros(output_shape, dtype=images[0].dtype) if is_torch else np.zeros(output_shape, dtype=images[0].dtype)
    
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        column_image = torch.concatenate(column, dim=0 + batched) if is_torch else np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image

def put_text_on_image(image: np.ndarray, lines: list[str], max_width: int = 200) -> np.ndarray:
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)

    new_lines = []
    for line in lines:
        words = line.split()
        current_line = []
        for word in words:
            test_line = " ".join(current_line + [word])
            test_width = font.getlength(test_line)
            if test_width <= max_width:
                current_line.append(word)
            else:
                new_lines.append(" ".join(current_line))
                current_line = [word]
        if current_line:
            new_lines.append(" ".join(current_line))

    y = -10
    for line in new_lines:
        bbox = draw.textbbox((0, 0), text=line)
        textheight = bbox[3] - bbox[1]
        y += textheight + 10
        x = 10
        draw.text((x, y), text=line, fill=(0, 0, 0))
    return np.array(image)

def put_info_on_image(image: np.ndarray, info: dict[str, float], extras: Optional[list[str]] = None, overlay: bool = True) -> np.ndarray:
    lines = [f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in info.items()]
    if extras is not None:
        lines.extend(extras)
    return put_text_on_image(image, lines)

def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # rotate 180 degrees
    return img

def get_libero_wrist_image(obs: dict[str, np.ndarray], resize_size: Union[int, tuple[int, int]] = 224) -> np.ndarray:
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # rotate 180 degrees
    return img

def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    if quat[3] > 1.0: quat[3] = 1.0
    elif quat[3] < -1.0: quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def save_rollout_video(rollout_images: list[np.ndarray], output_dir: str, video_name: str, fps: int = 30) -> None:
    os.makedirs(output_dir, exist_ok=True)
    mp4_path = os.path.join(output_dir, f"{video_name}.mp4")
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()

def get_benchmark_overridden(benchmark_name) -> Benchmark:
    """
    Return the Benchmark class for a given name.
    Incorporates the dynamic aggregation for "libero_130".
    """
    name = str(benchmark_name).lower()
    
    if name != "libero_130":
        return benchmark.get_benchmark(benchmark_name)

    libero_cls = benchmark.BENCHMARK_MAPPING.get("libero_130", None)
    if libero_cls is not None:
        return libero_cls

    aggregated_task_map: dict[str, benchmark.Task] = {}
    suites = getattr(benchmark, "libero_suites", [])
    
    for suite_name in suites:
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name, task in suite_map.items():
            if task_name not in aggregated_task_map:
                aggregated_task_map[task_name] = task

    class LIBERO_ALL(Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = "libero_130"
            self._make_benchmark()

        def _make_benchmark(self):
            tasks = list(aggregated_task_map.values())
            self.tasks = tasks
            self.n_tasks = len(self.tasks)

    benchmark.BENCHMARK_MAPPING["libero_130"] = LIBERO_ALL
    return LIBERO_ALL
    
    