import os
from typing import Dict, List, Optional, Union

import imageio
import numpy as np
import torch
import tqdm
from PIL import Image, ImageDraw, ImageFont
Array = Union[torch.Tensor, np.ndarray]


def _to_numpy(array: Union[Array]) -> np.ndarray:
    if isinstance(array, (dict)):
        return {k: _to_numpy(v) for k, v in array.items()}
    if isinstance(array, torch.Tensor):
        return array.cpu().numpy()
    if (
        isinstance(array, np.ndarray)
        or isinstance(array, bool)
        or isinstance(array, str)
        or isinstance(array, float)
        or isinstance(array, int)
    ):
        return array
    else:
        return np.array(array)


def to_numpy(array: Union[Array], dtype=None) -> np.ndarray:
    array = _to_numpy(array)
    if dtype is not None:
        return array.astype(dtype)
    return array


def torch_clone_dict(data: dict) -> dict:
    """
    Recursively clones all torch tensors in a dictionary.
    If the input was a torch tensor, it will return a clone of the tensor.
    """
    if isinstance(data, torch.Tensor):
        return data.clone()

    output_dict = {}
    for key, value in data.items():
        if isinstance(value, dict):
            output_dict[key] = torch_clone_dict(value)
        elif isinstance(value, torch.Tensor):
            output_dict[key] = value.clone()
        else:
            output_dict[key] = value
    return output_dict


def images_to_video(
    images: List[Array],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    verbose: bool = True,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    References:
        https://github.com/facebookresearch/habitat-lab/blob/main/habitat/utils/visualizations/utils.py
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    output_path = os.path.join(output_dir, video_name)
    writer = imageio.get_writer(
        output_path, fps=fps, quality=quality, **kwargs)
    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm.tqdm(images)
    else:
        images_iter = images
    for im in images_iter:
        writer.append_data(im)
    writer.close()


def tile_images(images: List[Array], nrows=1) -> Array:
    """
    Tile multiple images to a single image comprised of nrows and an appropriate number of columns to fit all the images.
    The images can also be batched (e.g. of shape (B, H, W, C)), but give images must all have the same batch size.

    if nrows is 1, images can be of different sizes. If nrows > 1, they must all be the same size.
    """
    # Sort images in descending order of vertical height
    batched = False
    if len(images[0].shape) == 4:
        batched = True
    if nrows == 1:
        images = sorted(
            images, key=lambda x: x.shape[0 + batched], reverse=True)

    columns = []
    if batched:
        max_h = images[0].shape[1] * nrows
        cur_h = 0
        cur_w = images[0].shape[2]
    else:
        max_h = images[0].shape[0] * nrows
        cur_h = 0
        cur_w = images[0].shape[1]

    # Arrange images in columns from left to right
    column = []
    for im in images:
        if cur_h + im.shape[0 + batched] <= max_h and cur_w == im.shape[1 + batched]:
            column.append(im)
            cur_h += im.shape[0 + batched]
        else:
            columns.append(column)
            column = [im]
            cur_h, cur_w = im.shape[0 + batched: 2 + batched]
    columns.append(column)

    # Tile columns
    total_width = sum(x[0].shape[1 + batched] for x in columns)

    is_torch = False
    if torch is not None:
        is_torch = isinstance(images[0], torch.Tensor)

    output_shape = (max_h, total_width, 3)
    if batched:
        output_shape = (images[0].shape[0], max_h, total_width, 3)
    if is_torch:
        output_image = torch.zeros(output_shape, dtype=images[0].dtype)
    else:
        output_image = np.zeros(output_shape, dtype=images[0].dtype)
    cur_x = 0
    for column in columns:
        cur_w = column[0].shape[1 + batched]
        next_x = cur_x + cur_w
        if is_torch:
            column_image = torch.concatenate(column, dim=0 + batched)
        else:
            column_image = np.concatenate(column, axis=0 + batched)
        cur_h = column_image.shape[0 + batched]
        output_image[..., :cur_h, cur_x:next_x, :] = column_image
        cur_x = next_x
    return output_image


TEXT_FONT = None


def put_text_on_image(image: np.ndarray, lines: List[str]):
    global TEXT_FONT
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    if TEXT_FONT is None:
        TEXT_FONT = ImageFont.truetype(
            os.path.join(os.path.dirname(__file__),
                         "UbuntuSansMono-Regular.ttf"),
            size=16,
        )
    y = -10
    for line in lines:
        bbox = draw.textbbox((0, 0), text=line)
        textheight = bbox[3] - bbox[1]
        y += textheight + 10
        x = 10
        draw.text((x, y), text=line, fill=(0, 255, 0), font=TEXT_FONT)
    return np.array(image)


def put_info_on_image(image, info: Dict[str, float], extras=None, overlay=True):
    lines = [
        f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}"
        for k, v in info.items()
    ]
    if extras is not None:
        lines.extend(extras)
    return put_text_on_image(image, lines)
