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

import torch
import torch.nn as nn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = None
        self.conv = doubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        if self.up is None:
            self.up = nn.Upsample(
                size=x2.size()[2:], mode="bilinear", align_corners=True
            )
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


def doubleConv(in_channels, out_channels, mid_channels=None):
    if mid_channels is None:
        mid_channels = out_channels
    layer = []
    layer.append(
        nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
    )
    layer.append(nn.BatchNorm2d(mid_channels))
    layer.append(nn.ReLU(inplace=True))
    layer.append(
        nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
    )
    layer.append(nn.BatchNorm2d(out_channels))
    layer.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layer)


def down(in_channels, out_channels):
    layer = []
    layer.append(nn.MaxPool2d(2, stride=2))
    layer.append(doubleConv(in_channels, out_channels))
    return nn.Sequential(*layer)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channel=64):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = 5
        self.in_conv = doubleConv(self.in_channels, base_channel)
        self.downs = [None] * self.num_layers
        self.ups = [None] * self.num_layers
        for i in range(self.num_layers):
            down_in_channel = base_channel * 2**i
            down_out_channel = (
                down_in_channel * 2 if i < self.num_layers - 1 else down_in_channel
            )
            up_in_channel = base_channel * 2 ** (self.num_layers - i)
            up_out_channel = (
                up_in_channel // 4 if i < self.num_layers - 1 else base_channel
            )
            self.downs[i] = down(down_in_channel, down_out_channel)
            self.ups[i] = Up(up_in_channel, up_out_channel)
        self.downs = nn.Sequential(*self.downs)
        self.ups = nn.Sequential(*self.ups)
        self.out = nn.Conv2d(
            in_channels=base_channel, out_channels=self.out_channels, kernel_size=1
        )

    def forward(self, x):
        x = self.in_conv(x)
        xs = [x]
        for down in self.downs:
            x = down(xs[-1])
            xs.append(x)

        x_out = None
        for x, up in zip(xs[::-1][1:], self.ups):
            if x_out is None:
                x_out = up(xs[-1], xs[-2])
            else:
                x_out = up(x_out, x)
        out = self.out(x_out)
        return out


class BottleNeckResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                stride=stride,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * 4,
                    stride=stride,
                    kernel_size=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * 4),
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, output_dim=14, input_channels=3, *args, **kwargs):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                input_channels,
                64,
                kernel_size=(7, 7),
                stride=(2, 2),
                padding=(3, 3),
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.conv2_x = self._make_layer(64, 3, 1)
        self.conv3_x = self._make_layer(128, 4, 2)
        self.conv4_x = self._make_layer(256, 6, 2)
        self.conv5_x = self._make_layer(512, 3, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, output_dim)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                BottleNeckResidualBlock(self.in_channels, out_channels, stride)
            )
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


class IDM(nn.Module):
    def __init__(self, model_name, *args, **kwargs):
        super(IDM, self).__init__()
        match model_name:
            case "mask":
                self.model = Mask(*args, **kwargs)
            case _:
                raise ValueError(f"Unsupported model name: {model_name}")
        if self.model.output_dim == 14:
            train_mean = torch.tensor(
                [
                    -0.26866713,
                    0.83559588,
                    0.69520934,
                    -0.29099351,
                    0.18849116,
                    -0.01014598,
                    1.41953145,
                    0.35073715,
                    1.05651613,
                    0.8930193,
                    -0.37493264,
                    -0.18510782,
                    -0.0272574,
                    1.35274259,
                ]
            )
            train_std = torch.tensor(
                [
                    0.25945241,
                    0.65903812,
                    0.52147207,
                    0.42150272,
                    0.32029947,
                    0.28452226,
                    1.78270006,
                    0.29091741,
                    0.67675932,
                    0.58250554,
                    0.42399049,
                    0.28697442,
                    0.31100304,
                    1.67651926,
                ]
            )
        else:
            train_mean = torch.tensor(
                [
                    -0.0011163579765707254,
                    0.3502498269081116,
                    0.010254094377160072,
                    -2.0258395671844482,
                    0.06505978852510452,
                    2.3033766746520996,
                    0.8659588098526001,
                    0.026907790452241898,
                    -0.027255306020379066,
                ]
            )
            train_std = torch.tensor(
                [
                    0.12338999658823013,
                    0.35243555903434753,
                    0.17533640563488007,
                    0.43524453043937683,
                    0.416223406791687,
                    0.31947872042655945,
                    0.6888905167579651,
                    0.014177982695400715,
                    0.014080556109547615,
                ]
            )
        self.register_buffer("train_mean", train_mean)
        self.register_buffer("train_std", train_std)

    def normalize(self, x):
        x = (x - self.train_mean) / self.train_std
        return x

    def forward(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        if isinstance(output, tuple):
            return output[0] * self.train_std + self.train_mean, *output[1:]
        else:
            return output * self.train_std + self.train_mean


class Mask(nn.Module):
    def __init__(self, output_dim: int = 14, *args, **kwargs):
        super().__init__()
        self.output_dim = output_dim
        self.mask_model = UNet(3, 1)
        self.resnet_model = ResNet(output_dim, 3)

    def forward(self, images, return_mask=False, *args, **kwargs):
        mask = (1 + torch.tanh(self.mask_model(images))) / 2
        mask_hard = torch.where(mask < 0.5, 0.0, 1.0)
        inputs = images * ((mask_hard - mask).detach() + mask)
        outputs = self.resnet_model(inputs)
        if return_mask:
            return outputs, mask
        else:
            return outputs, None
