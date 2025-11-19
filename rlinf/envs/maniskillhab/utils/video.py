import os
from typing import Dict, List, Optional

import cv2
from tqdm import tqdm

import numpy as np

from mani_skill.utils.structs.types import Array


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

    h, w, c = images[-1].shape
    out = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h), **kwargs
    )

    if verbose:
        print(f"Video created: {output_path}")
        images_iter = tqdm(images)
    else:
        images_iter = images

    for img in images_iter:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out.write(img)
    out.release()


def put_text_on_image(
    image: Array,
    lines: List[str],
    rgb=(0, 255, 0),
    font_thickness=1,
):
    assert image.dtype == np.uint8, image.dtype
    image = image.copy()

    font_size = 0.5
    font_thickness = font_thickness
    font = cv2.FONT_HERSHEY_SIMPLEX

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            image,
            line,
            (x, y),
            font,
            font_size,
            rgb,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def append_text_to_image(
    image: Array,
    lines: List[str],
    rgb=(255, 255, 255),
    font_thickness=1,
):
    r"""Appends text left to an image of size (height, width, channels).
    The returned image has white text on a black background.
    Args:
        image: the image to put text
        text: a string to display
    Returns:
        A new image with text inserted left to the input image
    See also:
        habitat.utils.visualization.utils
    """
    # h, w, c = image.shape
    font_size = 0.5
    font_thickness = font_thickness
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = np.zeros(image.shape, dtype=np.uint8)

    y = 0
    for line in lines:
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        y += textsize[1] + 10
        x = 10
        cv2.putText(
            blank_image,
            line,
            (x, y),
            font,
            font_size,
            rgb,
            font_thickness,
            lineType=cv2.LINE_AA,
        )
    # text_image = blank_image[0 : y + 10, 0:w]
    # final = np.concatenate((image, text_image), axis=0)
    final = np.concatenate((blank_image, image), axis=1)
    return final


def put_info_on_image(
    image,
    info: Dict[str, float],
    extras=None,
    overlay=True,
    rgb=(0, 255, 0),
    font_thickness=1,
):
    lines = [f"{k}: {v:.3f}" for k, v in info.items()]
    if extras is not None:
        lines.extend(extras)
    if overlay:
        return put_text_on_image(image, lines, rgb=rgb, font_thickness=font_thickness)
    else:
        return append_text_to_image(image, lines, rgb=rgb)
