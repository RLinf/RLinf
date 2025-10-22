# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Dict, Any, Optional, Tuple
import numpy as np


class RollingStats:
    def __init__(self, feature_size: int):
        self.count = 0
        self.mean = np.zeros(feature_size, dtype=np.float64)
        self.M2 = np.zeros(feature_size, dtype=np.float64)
        self.min = np.full(feature_size, np.inf, dtype=np.float64)
        self.max = np.full(feature_size, -np.inf, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        for row in x:
            self.count += 1
            delta = row - self.mean
            self.mean += delta / self.count
            delta2 = row - self.mean
            self.M2 += delta * delta2
            self.min = np.minimum(self.min, row)
            self.max = np.maximum(self.max, row)

    def finalize(self, reshape: Optional[Tuple[int, int, int]] = None) -> Dict[str, Any]:
        var = np.zeros_like(self.mean)
        if self.count > 1:
            var = self.M2 / (self.count - 1)
        std = np.sqrt(np.maximum(var, 0.0))
        # Apply reshape if specified
        if reshape is not None:
            mean = self.mean.reshape(reshape)
            std = std.reshape(reshape)
            min_val = self.min.reshape(reshape)
            max_val = self.max.reshape(reshape)
        else:
            mean = self.mean
            std = std
            min_val = self.min
            max_val = self.max
        
        return {
            "count": [int(self.count)],
            "mean": mean.tolist(),
            "std": std.tolist(),
            "min": min_val.tolist(),
            "max": max_val.tolist(),
        }


def compute_image_channel_stats(images: np.ndarray) -> Dict[str, Any]:
    """
    images: array of shape [T, H, W, C] in uint8 or float.
    Returns per-channel mean/std/min/max.
    """
    if images.size == 0:
        return {"count": [0], "mean": [], "std": [], "min": [], "max": []}
    if images.dtype != np.float32 and images.dtype != np.float64:
        arr = images.astype(np.float32) / 255.0
    else:
        arr = images
    # flatten spatial, keep channels
    c = arr.shape[-1]
    rs = RollingStats(c)
    # reshape to [T*H*W, C] to accumulate per-channel
    reshaped = arr.reshape(-1, c)
    rs.update(reshaped)
    return rs.finalize(reshape=(c, 1, 1))


