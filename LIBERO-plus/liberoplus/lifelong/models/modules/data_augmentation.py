import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from robomimic.models.base_nets import CropRandomizer


class IdentityAug(nn.Module):
    def __init__(self, input_shape=None, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x

    def output_shape(self, input_shape):
        return input_shape


class TranslationAug(nn.Module):
    """
    Utilize the random crop from robomimic.
    """

    def __init__(
        self,
        input_shape,
        translation,
    ):
        super().__init__()

        self.pad_translation = translation // 2
        pad_output_shape = (
            input_shape[0],
            input_shape[1] + translation,
            input_shape[2] + translation,
        )

        self.crop_randomizer = CropRandomizer(
            input_shape=pad_output_shape,
            crop_height=input_shape[1],
            crop_width=input_shape[2],
        )

    def forward(self, x):
        if self.training:
            batch_size, temporal_len, img_c, img_h, img_w = x.shape
            x = x.reshape(batch_size, temporal_len * img_c, img_h, img_w)
            out = F.pad(x, pad=(self.pad_translation,) * 4, mode="replicate")
            out = self.crop_randomizer.forward_in(out)
            out = out.reshape(batch_size, temporal_len, img_c, img_h, img_w)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class ImgColorJitterAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class ImgColorJitterGroupAug(torch.nn.Module):
    """
    Conduct color jittering augmentation outside of proposal boxes
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.05,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        raise NotImplementedError
        if self.training and np.random.rand() > self.epsilon:
            out = self.color_jitter(x)
        else:
            out = x
        return out

    def output_shape(self, input_shape):
        return input_shape


class BatchWiseImgColorJitterAug(torch.nn.Module):
    """
    Color jittering augmentation to individual batch.
    This is to create variation in training data to combat
    BatchNorm in convolution network.
    """

    def __init__(
        self,
        input_shape,
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.3,
        epsilon=0.1,
    ):
        super().__init__()
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
        )
        self.epsilon = epsilon

    def forward(self, x):
        out = []
        for x_i in torch.split(x, 1):
            if self.training and np.random.rand() > self.epsilon:
                out.append(self.color_jitter(x_i))
            else:
                out.append(x_i)
        return torch.cat(out, dim=0)

    def output_shape(self, input_shape):
        return input_shape


class DataAugGroup(nn.Module):
    """
    Add augmentation to multiple inputs
    """

    def __init__(self, aug_list):
        super().__init__()
        self.aug_layer = nn.Sequential(*aug_list)

    def forward(self, x_groups):
        split_channels = []
        for i in range(len(x_groups)):
            split_channels.append(x_groups[i].shape[1])
        if self.training:
            x = torch.cat(x_groups, dim=1)
            out = self.aug_layer(x)
            out = torch.split(out, split_channels, dim=1)
            return out
        else:
            out = x_groups
        return out
