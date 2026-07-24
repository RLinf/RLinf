# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from rlinf.envs.utils import put_text_on_image


def cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def cfg_require(cfg: Any, key: str) -> Any:
    value = cfg_get(cfg, key, None)
    if value is None:
        raise KeyError(f"Missing required gen_reward config field: {key}")
    return value


def normalize_type(value: Any, prefixes: tuple[str, ...] = ()) -> str:
    type_name = str(value).lower().replace(":", ".")
    for prefix in prefixes:
        prefix = prefix if prefix.endswith(".") else f"{prefix}."
        if type_name.startswith(prefix):
            return type_name[len(prefix) :]
    return type_name


def media_to_uint8_nhwc(media: torch.Tensor | np.ndarray | list[Any]) -> np.ndarray:
    """Convert image/video batches to uint8 channel-last format."""
    if isinstance(media, list):
        arrays = []
        for item in media:
            if isinstance(item, Image.Image):
                arrays.append(np.asarray(item.convert("RGB"), dtype=np.uint8))
            else:
                arrays.append(np.asarray(item))
        media = np.stack(arrays, axis=0)
    elif isinstance(media, torch.Tensor):
        media = media.detach().cpu().float().numpy()
    else:
        media = np.asarray(media)

    if media.ndim == 3:
        media = media[None]
    if media.ndim not in (4, 5):
        raise ValueError(f"Expected image/video batch, got shape {media.shape}.")
    if np.issubdtype(media.dtype, np.floating):
        media = np.clip(media, 0.0, 1.0) * 255.0
    media = np.rint(media).clip(0, 255).astype(np.uint8)

    if media.ndim == 4:
        if media.shape[1] in (1, 3, 4) and media.shape[-1] not in (1, 3, 4):
            media = np.transpose(media, (0, 2, 3, 1))
    elif media.shape[-1] not in (1, 3, 4):
        if media.shape[2] in (1, 3, 4) and (
            media.shape[1] not in (1, 3, 4)
            or (media.shape[2] == 3 and media.shape[1] != 3)
        ):
            media = np.transpose(media, (0, 1, 3, 4, 2))
        elif media.shape[1] in (1, 3, 4):
            media = np.transpose(media, (0, 2, 3, 4, 1))

    if media.shape[-1] == 1:
        media = np.repeat(media, 3, axis=-1)
    if media.shape[-1] == 4:
        media = media[..., :3]
    return media


def resize_video(video: np.ndarray, height: int, width: int) -> np.ndarray:
    size = (width, height)
    frames = [
        np.asarray(Image.fromarray(frame).resize(size, Image.BILINEAR))
        for frame in video
    ]
    return np.stack(frames, axis=0)


def record_gt_video(record: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            media_to_uint8_nhwc(record["main_image"]),
            media_to_uint8_nhwc(record["future_video"]),
        ],
        axis=0,
    )


def prepare_video_pair(
    pred_video: np.ndarray,
    record: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    gt_video = record_gt_video(record)
    if pred_video.shape[0] != gt_video.shape[0]:
        raise ValueError(
            "Pred/GT videos must have the same number of frames, "
            f"got pred={pred_video.shape[0]} gt={gt_video.shape[0]}."
        )
    if pred_video.shape[0] == 0:
        raise ValueError("Pred/GT videos are empty.")
    if pred_video.shape[1:3] != gt_video.shape[1:3]:
        gt_video = resize_video(gt_video, *pred_video.shape[1:3])
    return pred_video, gt_video


def make_side_by_side_video(
    left_video: np.ndarray,
    right_video: np.ndarray,
    left_label: str,
    right_label: str,
) -> np.ndarray:
    if left_video.shape[0] != right_video.shape[0]:
        raise ValueError(
            "Side-by-side video comparison requires the same number of frames, "
            f"got left={left_video.shape[0]} right={right_video.shape[0]}."
        )

    if left_video.shape[1:3] != right_video.shape[1:3]:
        right_video = resize_video(right_video, *left_video.shape[1:3])

    frames = []
    left_max_width = max(120, left_video.shape[2] - 20)
    right_max_width = max(120, right_video.shape[2] - 20)
    for left_frame, right_frame in zip(left_video, right_video, strict=True):
        left_frame = put_text_on_image(
            left_frame.copy(), [left_label], max_width=left_max_width
        )
        right_frame = put_text_on_image(
            right_frame.copy(), [right_label], max_width=right_max_width
        )
        frames.append(np.concatenate([left_frame, right_frame], axis=1))
    return np.stack(frames, axis=0)


def make_future_video_comparison(
    pred_videos: np.ndarray,
    records: list[dict[str, Any]],
) -> np.ndarray:
    return np.stack(
        [
            make_side_by_side_video(
                *prepare_video_pair(pred_video, record),
                "pred",
                "gt",
            )
            for pred_video, record in zip(pred_videos, records, strict=True)
        ],
        axis=0,
    )


def put_header_text(image: np.ndarray, text: str, max_width: int) -> np.ndarray:
    image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=20)
    lines = []
    current_line = []
    for word in str(text).split():
        candidate = " ".join(current_line + [word])
        if font.getlength(candidate) <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))

    line_gap = 8
    line_heights = [
        draw.textbbox((0, 0), line, font=font)[3]
        - draw.textbbox((0, 0), line, font=font)[1]
        for line in lines
    ]
    total_height = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    y = max(4, (image.height - total_height) // 2)
    for line, line_height in zip(lines, line_heights, strict=True):
        draw.text((10, y), text=line, fill=(0, 0, 0), font=font)
        y += line_height + line_gap
    return np.asarray(image)


def put_video_text(
    media: np.ndarray,
    task_descriptions: list[str] | None = None,
    score_curves: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    media = media.copy()
    max_width = max(120, media.shape[3] - 20)
    if task_descriptions is not None:
        media_with_header = []
        for batch_idx in range(media.shape[0]):
            header = np.full((112, media.shape[3], 3), 245, dtype=np.uint8)
            if batch_idx < len(task_descriptions):
                header = put_header_text(
                    header,
                    task_descriptions[batch_idx],
                    max_width=max_width,
                )
            frames = [
                np.concatenate([header, media[batch_idx, frame_idx]], axis=0)
                for frame_idx in range(media.shape[1])
            ]
            media_with_header.append(np.stack(frames, axis=0))
        media = np.stack(media_with_header, axis=0)

    if not score_curves:
        return media

    media_with_curves = []
    for batch_idx in range(media.shape[0]):
        frames = []
        for frame_idx in range(media.shape[1]):
            curve_panel = make_score_curve_panel(
                score_curves,
                batch_idx=batch_idx,
                frame_idx=frame_idx,
                width=media.shape[3],
                num_frames=media.shape[1],
            )
            frames.append(
                np.concatenate([media[batch_idx, frame_idx], curve_panel], axis=0)
            )
        media_with_curves.append(np.stack(frames, axis=0))
    return np.stack(media_with_curves, axis=0)


def make_score_curve_panel(
    score_curves: dict[str, np.ndarray],
    batch_idx: int,
    frame_idx: int,
    width: int,
    num_frames: int,
) -> np.ndarray:
    row_height = 65
    right = min(12, max(1, width // 10))
    left = min(max(40, width // 5), max(1, width - right - 1))
    plot_width = max(1, width - left - right)
    colors = [
        (79, 145, 116),
        (76, 114, 176),
        (221, 151, 77),
        (128, 128, 128),
        (211, 94, 96),
        (129, 114, 179),
    ]
    panel = np.full(
        (row_height * (len(score_curves) + 1), width, 3), 255, dtype=np.uint8
    )
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)

    for row_idx, (name, values) in enumerate(score_curves.items()):
        color = colors[row_idx % len(colors)]
        values = values[batch_idx].astype(np.float32)
        row_top = row_idx * row_height
        plot_top = row_top + 8
        plot_bottom = row_top + row_height - 10
        vmin = float(values.min())
        vmax = float(values.max())
        if vmin == vmax:
            vmin -= 1.0
            vmax += 1.0

        def point(idx: int, value: float) -> tuple[int, int]:
            x = left + int(round(idx * plot_width / max(1, len(values) - 1)))
            y = plot_bottom - int(
                round((float(value) - vmin) * (plot_bottom - plot_top) / (vmax - vmin))
            )
            return x, y

        draw.rectangle(
            [left, plot_top, width - right, plot_bottom],
            outline=(226, 226, 226),
        )
        mid_y = (plot_top + plot_bottom) // 2
        draw.line(
            [(left, mid_y), (width - right, mid_y)],
            fill=(235, 235, 235),
            width=1,
        )
        points = [point(idx, value) for idx, value in enumerate(values)]
        if len(points) > 1:
            draw.line(points, fill=color, width=3)
        score_idx = int(round(frame_idx * (len(values) - 1) / max(1, num_frames - 1)))
        marker_x, marker_y = point(score_idx, values[score_idx])
        draw.ellipse(
            [marker_x - 4, marker_y - 4, marker_x + 4, marker_y + 4],
            fill=color,
        )
        draw.text((8, row_top + 16), f"{name}: {values[score_idx]:.3f}", fill=color)
        if row_idx == 0:
            draw.text(
                (max(left, width - 115), row_top + 16),
                f"frame: {frame_idx + 1}/{num_frames}",
                fill=(80, 80, 80),
            )

    return np.asarray(image)


def extract_quoted_text(text: str) -> str:
    match = re.search(r'"([^"]+)"', str(text))
    return match.group(1) if match else str(text)


__all__ = [
    "cfg_get",
    "cfg_require",
    "extract_quoted_text",
    "make_future_video_comparison",
    "make_side_by_side_video",
    "resize_video",
    "make_score_curve_panel",
    "record_gt_video",
    "media_to_uint8_nhwc",
    "put_video_text",
    "normalize_type",
]
