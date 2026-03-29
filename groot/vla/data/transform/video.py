from typing import Any, Callable, ClassVar, Optional, Literal

import albumentations as A
import cv2
from einops import rearrange
import functools
import numpy as np
from pydantic import Field, PrivateAttr, field_validator
import torch
import torchvision.transforms.v2 as T

from groot.vla.data.schema import DatasetMetadata
from groot.vla.data.transform.base import ModalityTransform


class VideoTransform(ModalityTransform):
    # Configurable attributes
    backend: str = Field(
        default="torchvision", description="The backend to use for the transformations"
    )

    # Model variables
    _train_transform: Callable | None = PrivateAttr(default=None)
    _eval_transform: Callable | None = PrivateAttr(default=None)
    _original_resolutions: dict[str, tuple[int, int]] = PrivateAttr(default_factory=dict)

    # Model constants
    _INTERPOLATION_MAP: ClassVar[dict[str, dict[str, Any]]] = PrivateAttr(
        {
            "nearest": {
                "albumentations": cv2.INTER_NEAREST,
                "torchvision": T.InterpolationMode.NEAREST,
            },
            "linear": {
                "albumentations": cv2.INTER_LINEAR,
                "torchvision": T.InterpolationMode.BILINEAR,
            },
            "cubic": {
                "albumentations": cv2.INTER_CUBIC,
                "torchvision": T.InterpolationMode.BICUBIC,
            },
            "area": {
                "albumentations": cv2.INTER_AREA,
                "torchvision": None,  # Torchvision does not support this interpolation mode
            },
            "lanczos4": {
                "albumentations": cv2.INTER_LANCZOS4,  # Lanczos with a 4x4 filter
                "torchvision": T.InterpolationMode.LANCZOS,  # Torchvision does not specify filter size, might be different from 4x4
            },
            "linear_exact": {
                "albumentations": cv2.INTER_LINEAR_EXACT,
                "torchvision": None,  # Torchvision does not support this interpolation mode
            },
            "nearest_exact": {
                "albumentations": cv2.INTER_NEAREST_EXACT,
                "torchvision": T.InterpolationMode.NEAREST_EXACT,
            },
            "max": {
                "albumentations": cv2.INTER_MAX,
                "torchvision": None,
            },
        }
    )

    @property
    def train_transform(self) -> Callable:
        assert (
            self._train_transform is not None
        ), "Transform is not set. Please call set_metadata() before calling apply()."
        return self._train_transform

    @train_transform.setter
    def train_transform(self, value: Callable):
        self._train_transform = value

    @property
    def eval_transform(self) -> Callable | None:
        return self._eval_transform

    @eval_transform.setter
    def eval_transform(self, value: Callable | None):
        self._eval_transform = value

    @property
    def original_resolutions(self) -> dict[str, tuple[int, int]]:
        assert (
            self._original_resolutions is not None
        ), "Original resolutions are not set. Please call set_metadata() before calling apply()."
        return self._original_resolutions

    @original_resolutions.setter
    def original_resolutions(self, value: dict[str, tuple[int, int]]):
        self._original_resolutions = value

    def check_input(self, data: dict[str, Any]):
        if self.backend == "torchvision":
            for key in self.apply_to:
                assert isinstance(data[key], torch.Tensor), f"Video {key} is not a torch tensor"
                assert data[key].ndim in [
                    4,
                    5,
                ], f"Expected video {key} to have 4 or 5 dimensions (T, C, H, W or T, B, C, H, W), got {data[key].ndim}"
        elif self.backend == "albumentations":
            for key in self.apply_to:
                assert isinstance(data[key], np.ndarray), f"Video {key} is not a numpy array"
                assert data[key].ndim in [
                    4,
                    5,
                ], f"Expected video {key} to have 4 or 5 dimensions (T, C, H, W or T, B, C, H, W), got {data[key].ndim}"
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        super().set_metadata(dataset_metadata)
        self.original_resolutions = {}
        for key in self.apply_to:
            split_keys = key.split(".")
            assert len(split_keys) == 2, f"Invalid key: {key}. Expected format: modality.key"
            sub_key = split_keys[1]
            if sub_key in dataset_metadata.modalities.video:
                self.original_resolutions[key] = dataset_metadata.modalities.video[
                    sub_key
                ].resolution
            else:
                raise ValueError(
                    f"Video key {sub_key} not found in dataset metadata. Available keys: {dataset_metadata.modalities.video.keys()}"
                )
        train_transform = self.get_transform(mode="train")
        eval_transform = self.get_transform(mode="eval")
        if self.backend == "albumentations":
            self.train_transform = A.ReplayCompose(transforms=[train_transform])  # type: ignore
            if eval_transform is not None:
                self.eval_transform = A.ReplayCompose(transforms=[eval_transform])  # type: ignore
        else:
            assert train_transform is not None, "Train transform must be set"
            self.train_transform = train_transform
            self.eval_transform = eval_transform

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.training:
            transform = self.train_transform
        else:
            transform = self.eval_transform
            if transform is None:
                return data
        assert (
            transform is not None
        ), "Transform is not set. Please call set_metadata() before calling apply()."
        try:
            self.check_input(data)
        except AssertionError as e:
            raise ValueError(
                f"Input data does not match the expected format for {self.__class__.__name__}: {e}"
            ) from e

        # Concatenate views
        views = [data[key] for key in self.apply_to]
        num_views = len(views)
        is_batched = views[0].ndim == 5
        bs = views[0].shape[0] if is_batched else 1
        if isinstance(views[0], torch.Tensor):
            views = torch.cat(views, 0)
        elif isinstance(views[0], np.ndarray):
            views = np.concatenate(views, 0)
        else:
            raise ValueError(f"Unsupported view type: {type(views[0])}")
        if is_batched:
            views = rearrange(views, "(v b) t c h w -> (v b t) c h w", v=num_views, b=bs)
        # Apply the transform
        if self.backend == "torchvision":
            views = transform(views)
        elif self.backend == "albumentations":
            assert isinstance(transform, A.ReplayCompose), "Transform must be a ReplayCompose"
            first_frame = views[0]
            transformed = transform(image=first_frame)
            replay_data = transformed["replay"]
            transformed_first_frame = transformed["image"]

            if len(views) > 1:
                # Apply the same transformations to the rest of the frames
                transformed_frames = [
                    transform.replay(replay_data, image=frame)["image"] for frame in views[1:]
                ]
                # Add the first frame back
                transformed_frames = [transformed_first_frame] + transformed_frames
            else:
                # If there is only one frame, just make a list with one frame
                transformed_frames = [transformed_first_frame]

            # Delete the replay data to save memory
            del replay_data
            views = np.stack(transformed_frames, 0)

        else:
            raise ValueError(f"Backend {self.backend} not supported")
        # Split views
        if is_batched:
            views = rearrange(views, "(v b t) c h w -> v b t c h w", v=num_views, b=bs)
        else:
            views = rearrange(views, "(v t) c h w -> v t c h w", v=num_views)
        for key, view in zip(self.apply_to, views):
            data[key] = view
        return data

    @classmethod
    def _validate_interpolation(cls, interpolation: str):
        if interpolation not in cls._INTERPOLATION_MAP:
            raise ValueError(f"Interpolation mode {interpolation} not supported")

    def _get_interpolation(self, interpolation: str, backend: str = "torchvision"):
        """
        Get the interpolation mode for the given backend.

        Args:
            interpolation (str): The interpolation mode.
            backend (str): The backend to use.

        Returns:
            Any: The interpolation mode for the given backend.
        """
        return self._INTERPOLATION_MAP[interpolation][backend]

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        raise NotImplementedError(
            "set_transform is not implemented for VideoTransform. Please implement this function to set the transforms."
        )


class VideoCrop(VideoTransform):
    height: int | None = Field(default=None, description="The height of the input image")
    width: int | None = Field(default=None, description="The width of the input image")
    scale: float = Field(
        ...,
        description="The scale of the crop. The crop size is (width * scale, height * scale)",
    )

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the transform for the given mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: If mode is "train", return a random crop transform. If mode is "eval", return a center crop transform.
        """
        # 1. Check the input resolution
        assert (
           len(set(self.original_resolutions.values())) == 1
        ), f"All video keys must have the same resolution, got: {self.original_resolutions}"
        if self.height is None:
            assert self.width is None, "Height and width must be either both provided or both None"
            self.width, self.height = self.original_resolutions[self.apply_to[0]]
        else:
            assert (
                self.width is not None
            ), "Height and width must be either both provided or both None"
        # 2. Create the transform
        size = (int(self.height * self.scale), int(self.width * self.scale))
        if self.backend == "torchvision":
            if mode == "train":
                return T.RandomCrop(size)
            elif mode == "eval":
                return T.CenterCrop(size)
            else:
                raise ValueError(f"Crop mode {mode} not supported")
        elif self.backend == "albumentations":
            if mode == "train":
                return A.RandomCrop(height=size[0], width=size[1], p=1)
            elif mode == "eval":
                return A.CenterCrop(height=size[0], width=size[1], p=1)
            else:
                raise ValueError(f"Crop mode {mode} not supported")
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict[str, Any]):
        super().check_input(data)
        # Check the input resolution
        for key in self.apply_to:
            if self.backend == "torchvision":
                height, width = data[key].shape[-2:]
            elif self.backend == "albumentations":
                height, width = data[key].shape[-3:-1]
            else:
                raise ValueError(f"Backend {self.backend} not supported")
            assert (
                height == self.height and width == self.width
            ), f"Video {key} has invalid shape {height, width}, expected {self.height, self.width}"


class VideoRandomErasing(VideoTransform):
    """Adds random rectangles overlaying the video.

    This discourages overfitting to the background.
    """

    probability: float = Field(default=0.2, description="Probability of applying the transform")
    scale: tuple[float, float] = Field(default=(0.02, 0.33), description="Scale of the rectangle")
    ratio: tuple[float, float] = Field(
        default=(0.3, 3.3), description="Aspect ratio of the rectangle"
    )
    value: Literal["random"] | tuple[float, float, float] = Field(
        default="random", description="Color to fill the erased region with"
    )

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the transform for the given mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: If mode is "train", return a transform that adds random rectangles. If mode is "eval", return a no-op.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomErasing(
                p=self.probability, scale=self.scale, ratio=self.ratio, value=self.value
            )
        elif self.backend == "albumentations":
            return A.Erasing(
                p=self.probability, scale=self.scale, ratio=self.ratio, value=self.value
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoResize(VideoTransform):
    height: int = Field(..., description="The height of the resize")
    width: int = Field(..., description="The width of the resize")
    interpolation: str = Field(default="linear", description="The interpolation mode")
    antialias: bool = Field(default=True, description="Whether to apply antialiasing")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the resize transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The resize transform.
        """
        interpolation = self._get_interpolation(self.interpolation, self.backend)
        if interpolation is None:
            raise ValueError(
                f"Interpolation mode {self.interpolation} not supported for torchvision"
            )
        if self.backend == "torchvision":
            size = (self.height, self.width)
            return T.Resize(size, interpolation=interpolation, antialias=self.antialias)
        elif self.backend == "albumentations":
            return A.Resize(
                height=self.height,
                width=self.width,
                interpolation=interpolation,
                p=1,
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomRotation(VideoTransform):
    degrees: float | tuple[float, float] = Field(
        ..., description="The degrees of the random rotation"
    )
    interpolation: str = Field("linear", description="The interpolation mode")

    @field_validator("interpolation")
    def validate_interpolation(cls, v):
        cls._validate_interpolation(v)
        return v

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the random rotation transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: The random rotation transform. None for eval mode.
        """
        if mode == "eval":
            return None
        interpolation = self._get_interpolation(self.interpolation, self.backend)
        if interpolation is None:
            raise ValueError(
                f"Interpolation mode {self.interpolation} not supported for torchvision"
            )
        if self.backend == "torchvision":
            return T.RandomRotation(self.degrees, interpolation=interpolation)  # type: ignore
        elif self.backend == "albumentations":
            return A.Rotate(limit=self.degrees, interpolation=interpolation, p=1)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoHorizontalFlip(VideoTransform):
    p: float = Field(..., description="The probability of the horizontal flip")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the horizontal flip transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a horizontal flip transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomHorizontalFlip(self.p)
        elif self.backend == "albumentations":
            return A.HorizontalFlip(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoGrayscale(VideoTransform):
    p: float = Field(..., description="The probability of the grayscale transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the grayscale transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a grayscale transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomGrayscale(self.p)
        elif self.backend == "albumentations":
            return A.ToGray(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoColorJitter(VideoTransform):
    brightness: float | tuple[float, float] = Field(
        ..., description="The brightness of the color jitter"
    )
    contrast: float | tuple[float, float] = Field(
        ..., description="The contrast of the color jitter"
    )
    saturation: float | tuple[float, float] = Field(
        ..., description="The saturation of the color jitter"
    )
    hue: float | tuple[float, float] = Field(..., description="The hue of the color jitter")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the color jitter transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a color jitter transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
            )
        elif self.backend == "albumentations":
            return A.ColorJitter(
                brightness=self.brightness,
                contrast=self.contrast,
                saturation=self.saturation,
                hue=self.hue,
                p=1,
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomGrayscale(VideoTransform):
    p: float = Field(..., description="The probability of the grayscale transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the grayscale transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a grayscale transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomGrayscale(self.p)
        elif self.backend == "albumentations":
            return A.ToGray(p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoRandomPosterize(VideoTransform):
    bits: int = Field(..., description="The number of bits to posterize the image")
    p: float = Field(..., description="The probability of the posterize transformation")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable | None:
        """Get the posterize transform, only used in train mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable | None: If mode is "train", return a posterize transform. If mode is "eval", return None.
        """
        if mode == "eval":
            return None
        if self.backend == "torchvision":
            return T.RandomPosterize(bits=self.bits, p=self.p)
        elif self.backend == "albumentations":
            return A.Posterize(num_bits=self.bits, p=self.p)
        else:
            raise ValueError(f"Backend {self.backend} not supported")


class VideoToTensor(VideoTransform):

    output_on_cuda: bool = Field(
        default=False,
        description="Output the tensor on CUDA if True.",
    )

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to tensor transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to tensor transform.
        """
        if self.backend == "torchvision":
            return functools.partial(
                self.__class__.to_tensor,
                output_on_cuda=self.output_on_cuda,
            )
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict):
        """Check if the input data has the correct shape.
        Expected video shape: [T, H, W, C], dtype np.uint8
        """
        for key in self.apply_to:
            assert key in data, f"Key {key} not found in data. Available keys: {data.keys()}"
            assert data[key].ndim in [
                4,
                5,
            ], f"Video {key} must have 4 or 5 dimensions, got {data[key].ndim}"
            assert (
                data[key].dtype == np.uint8
            ), f"Video {key} must have dtype uint8, got {data[key].dtype}"
            input_resolution = data[key].shape[-3:-1][::-1]
            if key in self.original_resolutions:
                expected_resolution = self.original_resolutions[key]
            else:
                expected_resolution = input_resolution
            assert (
                input_resolution == expected_resolution
            ), f"Video {key} has invalid resolution {input_resolution}, expected {expected_resolution}. Full shape: {data[key].shape}"

    @staticmethod
    def to_tensor(frames: np.ndarray, output_on_cuda: bool) -> torch.Tensor:
        """Convert numpy array to tensor efficiently.

        Args:
            frames: numpy array of shape [T, H, W, C] in uint8 format
            output_on_cuda: whether to output the tensor on CUDA if True.
        Returns:
            tensor of shape [T, C, H, W] in range [0, 1]
        """
        frames = torch.from_numpy(frames)
        if output_on_cuda:
            frames = frames.cuda()
        frames = frames.to(torch.float32) / 255.0
        return frames.permute(0, 3, 1, 2)  # [T, C, H, W]


class VideoToNumpy(VideoTransform):
    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the to numpy transform. Same transform for both train and eval.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The to numpy transform.
        """
        if self.backend == "torchvision":
            return self.__class__.to_numpy
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    @staticmethod
    def to_numpy(frames: torch.Tensor) -> np.ndarray:
        """Convert tensor back to numpy array efficiently.

        Args:
            frames: tensor of shape [T, C, H, W] in range [0, 1]
        Returns:
            numpy array of shape [T, H, W, C] in uint8 format
        """
        frames = (frames.permute(0, 2, 3, 1) * 255).to(torch.uint8)
        return frames.cpu().numpy()


class VideoMergeTimeBatch(ModalityTransform):
    """
    Merge the batch and time dimensions of the video.
    """

    apply_to: list[str] = Field(..., description="The keys of the modalities to merge")

    def apply(self, data: dict) -> dict:
        warnings.warn(
            "VideoMergeTimeBatch is deprecated. Use ComposedModalityTransform instead.",
            DeprecationWarning,
        )
        for key in self.apply_to:
            data[key] = rearrange(data[key], "b t ... -> (b t) ...")
        return data


class VideoSplitTimeBatch(ModalityTransform):
    """
    Split the batch and time dimensions of the video.
    """

    apply_to: list[str] = Field(..., description="The keys of the modalities to split")
    time_dim: int = Field(..., description="The dimension of the time dimension")

    def apply(self, data: dict) -> dict:
        warnings.warn(
            "VideoSplitTimeBatch is deprecated. Use ComposedModalityTransform instead.",
            DeprecationWarning,
        )
        for key in self.apply_to:
            data[key] = rearrange(data[key], "(b t) ... -> b t ...", t=self.time_dim)
        return data


class VideoFocusRect(ModalityTransform):
    """
    Given a rectangle area in the video, apply focus effects on the target
    rectangle, by applying blur and noise to the surrounding region.

    Mainly useful for EgoView
    """

    # Region coordinates in normalized space [0,1]
    xtl: float = Field(2 / 12, description="Top-left x coordinate (normalized)", ge=0.0, le=1.0)
    ytl: float = Field(3 / 8, description="Top-left y coordinate (normalized)", ge=0.0, le=1.0)
    xbr: float = Field(10 / 12, description="Bottom-left x coordinate (normalized)", ge=0.0, le=1.0)
    ybr: float = Field(1.0, description="Bottom-left y coordinate (normalized)", ge=0.0, le=1.0)

    # Content region parameters (in pixel coordinates, None means auto-detect)
    content_y_min: Optional[int] = Field(
        None, description="Top coordinate of content region (pixels)"
    )
    content_y_max: Optional[int] = Field(
        None, description="Bottom coordinate of content region (pixels)"
    )
    content_x_min: Optional[int] = Field(
        None, description="Left coordinate of content region (pixels)"
    )
    content_x_max: Optional[int] = Field(
        None, description="Right coordinate of content region (pixels)"
    )

    # Jitter amount for coordinates (in normalized space)
    jitter: float = Field(0.05, description="Amount of random jitter to apply to coordinates")

    # Effect parameters
    blur_kernel: int = Field(95, description="Gaussian blur kernel size")
    noise_std: float = Field(0.3, description="Standard deviation of Gaussian noise")
    blend_size: float = Field(0.1, description="Size of blending region as fraction of image size")

    # Effect probabilities during training
    p_blur: float = Field(0.2, description="Probability of applying blur")
    p_noise: float = Field(0.2, description="Probability of applying noise")

    def detect_padding(self, image: np.ndarray) -> tuple[slice, slice]:
        """
        Detect padding in the image by finding non-black regions.
        Returns slices for the content region (y_slice, x_slice).
        """
        H, W = image.shape[:2]

        # If all content region parameters are provided, use them
        if all(
            param is not None
            for param in [
                self.content_y_min,
                self.content_y_max,
                self.content_x_min,
                self.content_x_max,
            ]
        ):
            y_min = max(0, self.content_y_min)
            y_max = min(H, self.content_y_max)
            x_min = max(0, self.content_x_min)
            x_max = min(W, self.content_x_max)
            return slice(y_min, y_max), slice(x_min, x_max)

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Find non-black regions
        rows = np.any(gray > 0.01, axis=1)
        cols = np.any(gray > 0.01, axis=0)

        # find first and last non-black pixel indices
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return slice(rmin, rmax + 1), slice(cmin, cmax + 1)

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Do nothing in eval mode
            return data

        for key in self.apply_to:
            video = data[key]

            # Handle numpy array case
            assert isinstance(
                video, np.ndarray
            ), f"Expected numpy array or torch tensor for {key}, got {type(video)}"
            assert video.ndim in {
                3,
                4,
            }, f"Expected [H, W, C] or [T, H, W, C] array for {key}, got shape {video.shape}"
            transformed = self._transform_video(video)

            data[key] = transformed

        return data

    def _transform_video(self, video: np.ndarray) -> np.ndarray:
        """
        Apply the focus rectangle transformation to a video.

        Args:
            video (np.ndarray): Video tensor of shape [T, H, W, C]

        Returns:
            np.ndarray: Transformed video
        """
        # Handle both single frame and video inputs
        is_single_frame = video.ndim == 3
        if is_single_frame:
            video = video[np.newaxis]

        T, H, W, C = video.shape

        assert (
            self.p_blur + self.p_noise <= 1.0
        ), "Sum of blur and noise probabilities must be <= 1.0"
        r = np.random.random()
        apply_blur = r < self.p_blur
        apply_noise = self.p_blur <= r < self.p_blur + self.p_noise
        alpha = random.uniform(0.0, 1.0)  # Noise blending factor

        # Apply jitter once to rectangle
        xtl = self.xtl + np.random.uniform(-self.jitter, self.jitter)
        ytl = self.ytl + np.random.uniform(-self.jitter, self.jitter)
        xbr = self.xbr + np.random.uniform(-self.jitter, self.jitter)
        ybr = self.ybr + np.random.uniform(-self.jitter, self.jitter)
        xtl, ytl, xbr, ybr = [np.clip(x, 0.0, 1.0) for x in [xtl, ytl, xbr, ybr]]

        # Detect padding from first frame (assume consistent across frames)
        y_slice, x_slice = self.detect_padding(video[0])
        content_h = y_slice.stop - y_slice.start
        content_w = x_slice.stop - x_slice.start

        # Convert normalized coordinates relative to pixel space
        x1 = int(xtl * content_w) + x_slice.start
        y1 = int(ytl * content_h) + y_slice.start
        x2 = int(xbr * content_w) + x_slice.start
        y2 = int(ybr * content_h) + y_slice.start

        # Create mask for the inner rectangle
        mask = np.zeros((H, W), dtype=np.float32)
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)
        cv2.fillPoly(mask, [pts], color=1.0)

        # Create a smooth blend mask around the target rectangle using distance transform
        content_mask = np.zeros((H, W), dtype=np.uint8)
        content_mask[y_slice, x_slice] = 1
        dist = cv2.distanceTransform(1 - (mask > 0).astype(np.uint8) * content_mask, cv2.DIST_L2, 3)
        blend_radius = int(min(content_h, content_w) * self.blend_size)
        blend_mask = np.clip(1.0 - dist / blend_radius, 0, 1)
        blend_mask *= content_mask
        blend_mask = blend_mask[..., np.newaxis]

        # Process all frames with same transformations
        result = np.zeros_like(video)
        for t in range(T):
            frame = video[t]
            modified = frame.copy()
            content = modified[y_slice, x_slice]

            if apply_blur:
                content = cv2.GaussianBlur(content, (self.blur_kernel, self.blur_kernel), 0)

            if apply_noise:
                background = np.random.randint(0, 256, content.shape, dtype=np.uint8) / 255.0
                content = alpha * content + (1 - alpha) * background
                content = np.clip(content, 0, 1)

            modified[y_slice, x_slice] = content
            result[t] = frame * blend_mask + modified * (1 - blend_mask)

        return result[0] if is_single_frame else result


class VideoNormalize(VideoTransform):
    mean: list[float] = Field(..., description="Mean for normalization")
    std: list[float] = Field(..., description="Standard deviation for normalization")

    def get_transform(self, mode: Literal["train", "eval"] = "train") -> Callable:
        """Get the normalization transform. Same for train and eval mode.

        Args:
            mode (Literal["train", "eval"]): The mode to get the transform for.

        Returns:
            Callable: The normalization transform.
        """
        print("Using VideoNormalize transform")
        if self.backend == "torchvision":
            return T.Normalize(mean=self.mean, std=self.std)
        elif self.backend == "albumentations":
            return A.Normalize(mean=self.mean, std=self.std, max_pixel_value=1.0, p=1.0)
        else:
            raise ValueError(f"Backend {self.backend} not supported")

    def check_input(self, data: dict):
        for key in self.apply_to:
            assert key in data, f"Key {key} not found in data"
            assert isinstance(data[key], torch.Tensor), f"Video {key} is not a torch tensor"
            assert data[key].ndim in [4, 5], f"Video {key} must have 4 or 5 dimensions, got {data[key].ndim}"
            assert data[key].dtype == torch.float32, f"Video {key} must be float32, got {data[key].dtype}"
            assert data[key].min() >= 0.0 and data[key].max() <= 1.0, (
                f"Video {key} must be in [0,1] range before normalization"
            )

# class VideoTransformLegacy(ModalityTransform):
#     def __init__(
#         self,
#         modality_keys: list[str],
#         backend: str = "torchvision",
#         crop_cfg: CropConfig | None = None,
#         resize_cfg: ResizeConfig | None = None,
#         random_rotation_cfg: RandomRotationConfig | None = None,
#         horizontal_flip_cfg: HorizontalFlipConfig | None = None,
#         grayscale_cfg: GrayscaleConfig | None = None,
#         color_jitter_cfg: ColorJitterConfig | None = None,
#         strong_vision_aug: bool = False,
#     ):
#         """
#         Initialize the video transform.
#         With the default settings, the input will be (T, H, W, C) where T is the number of frames.
#         The output will be (K, T, C, H, W) where K is the number of video keys.

#         Args:
#             modality_keys (list[str]): The keys of the modalities to load and transform.
#             backend (str): The backend to use for the transformations. The default is "torchvision".
#             crop_cfg (CropConfig | None): Configuration for the crop transformation. See CropConfig for more details.
#             resize_cfg (ResizeConfig | None): Configuration for the resize transformation. See ResizeConfig for more details.
#             random_rotation_cfg (RandomRotationConfig | None): Configuration for the random rotation transformation. See RandomRotationConfig for more details.
#             horizontal_flip_cfg (HorizontalFlipConfig | None): Configuration for the horizontal flip transformation. See HorizontalFlipConfig for more details.
#             grayscale_cfg (GrayscaleConfig | None): Configuration for the grayscale transformation. See GrayscaleConfig for more details.
#             color_jitter_cfg (ColorJitterConfig | None): Configuration for the color jitter transformation. See ColorJitterConfig for more details.
#             strong_vision_aug (bool): Whether to apply strong vision augmentation. The default is False.
#         """
#         super().__init__(modality_keys)
#         self.backend = backend
#         self.crop_cfg = crop_cfg
#         self.resize_cfg = resize_cfg
#         self.random_rotation_cfg = random_rotation_cfg
#         self.horizontal_flip_cfg = horizontal_flip_cfg
#         self.grayscale_cfg = grayscale_cfg
#         self.color_jitter_cfg = color_jitter_cfg
#         self.strong_vision_aug = strong_vision_aug
#         self.transforms = None

#     def set_metadata(self, dataset_metadata: TrainableDatasetMetadata_V1_1):
#         super().set_metadata(dataset_metadata)
#         # Get the original height and width
#         video_metadata = dataset_metadata.modalities.video
#         resolutions = {}
#         for key in self.modality_keys:
#             split_keys = key.split(".")
#             assert len(split_keys) == 2, f"Invalid key: {key}. Expected format: modality.key"
#             sub_key = split_keys[1]
#             resolutions[key] = video_metadata[sub_key].resolution
#         assert (
#             len(set(resolutions.values())) == 1
#         ), f"All video keys must have the same resolution, got: {resolutions}"
#         width, height = resolutions[self.modality_keys[0]]

#         transforms = []
#         if self.crop_cfg is not None:
#             self.crop_cfg.set_original_height_width(height, width)
#             transforms.append(self.crop_cfg.get_transform(self.backend))
#         if self.resize_cfg is not None:
#             transforms.append(self.resize_cfg.get_transform(self.backend))
#         if self.random_rotation_cfg is not None:
#             transforms.append(self.random_rotation_cfg.get_transform(self.backend))
#         if self.horizontal_flip_cfg is not None:
#             transforms.append(self.horizontal_flip_cfg.get_transform(self.backend))
#         if self.grayscale_cfg is not None:
#             transforms.append(self.grayscale_cfg.get_transform(self.backend))
#         if self.color_jitter_cfg is not None:
#             transforms.append(self.color_jitter_cfg.get_transform(self.backend))

#         if self.backend == "torchvision":
#             if len(transforms) == 0:
#                 transforms.append(T.Identity())
#             self.transforms = T.Compose(transforms)
#         else:
#             raise ValueError(f"Backend {self.backend} not supported")

#         if self.strong_vision_aug:
#             import kornia.augmentation as K
#             from kornia.augmentation import ImageSequential

#             assert (
#                 self.backend == "torchvision"
#             ), "Temporarily only support torchvision backend for strong augmentation"
#             self.strong_transform = ImageSequential(
#                 K.RandomErasing(p=0.5, scale=(0.005, 0.01), ratio=(0.3, 1.3)),
#                 K.RandomSaltAndPepperNoise(p=0.5, amount=0.05, salt_vs_pepper=0.5),
#                 K.RandomCutMixV2(p=0.5, num_mix=1, cut_size=(0.98, 1.0)),
#                 random_apply=1,
#                 keepdim=True,
#                 same_on_batch=True,
#             )

#     def __call__(self, data: dict) -> dict[str, torch.Tensor | np.ndarray | Image.Image]:
#         # Batch frames along the first dimension
#         frames = [data[key] for key in self.modality_keys]  # view x [T, H, W, C]
#         n_view, n_frames = len(frames), len(frames[0])
#         frames = np.concatenate(frames, 0)  # [view*T, H, W, C]

#         if self.backend == "torchvision":
#             transformed_frames = self.transform_torchvision(frames)
#         else:
#             raise ValueError(f"Backend {self.backend} not supported")

#         # De-batch the frames
#         transformed_frames = np.array(transformed_frames)  # [view*T, H, W, C]
#         H, W, C = transformed_frames.shape[-3:]
#         transformed_frames = {
#             key: x
#             for key, x in zip(
#                 self.modality_keys, transformed_frames.reshape(n_view, n_frames, H, W, C)
#             )
#         }

#         return transformed_frames

#     def check_input(self, data: dict):
#         for key in self.modality_keys:
#             assert key in data, f"Key {key} not found in data"
#             video = data[key]
#             assert isinstance(video, np.ndarray), f"Video {key} is not a numpy array"
#             assert video.ndim == 4, f"Video {key} must have 4 dimensions, got {video.ndim}"
#             assert video.dtype == np.uint8, f"Video {key} must have dtype uint8, got {video.dtype}"
#             shape = video.shape
#             expected_resolution = self.dataset_metadata.modalities.video[key].resolution
#             assert (
#                 shape[1:3] == expected_resolution
#             ), f"Video {key} has invalid shape {shape}, expected {expected_resolution}"

#     def transform_torchvision(
#         self, frames: np.ndarray
#     ) -> list[torch.Tensor | np.ndarray | Image.Image]:
#         """
#         frames: [view * T, H, W, C], np.uint8
#         """
#         if self.transforms is None:
#             raise ValueError(
#                 "Transform is not set. Please call set_metadata() before calling __call__()"
#             )
#         # Convert to batched tensor, using ToTensor() is too slow
#         frames_tensor = torch.from_numpy(frames).to(torch.float32) / 255.0
#         frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # [view * T, C, H, W]
#         transformed_frames = self.transforms(frames_tensor)
#         if self.strong_vision_aug:
#             transformed_frames = self.strong_transform(transformed_frames)

#         to_pil = T.ToPILImage()
#         transformed_frames = [to_pil(frame) for frame in transformed_frames]
#         return transformed_frames  # type: ignore


# class IdentityTransform(ModalityTransform):
#     def __call__(self, data: dict) -> dict:  # type: ignore
#         warnings.warn("IdentityTransform is used, further transformations is required.")
#         output = {}
#         for key in self.modality_keys:
#             output[key] = data[key]
#         return output
