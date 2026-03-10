"""Shared signal-processing utilities for the glove driver."""

import numpy as np


class LowPassFilter:
    """Simple delta-clipping low-pass filter.

    Each call to :meth:`filter` limits the per-joint change to at most
    ``±delta``, preventing abrupt jumps in the output signal.

    Args:
        delta: Maximum allowed change per call.
        num_joints: Number of joints (unused, kept for compatibility).
    """

    def __init__(self, delta: float = 0.1, num_joints: int = 6):
        self.delta = delta
        self.num_joints = num_joints
        self.filtered_values = None

    def filter(self, values):
        """Apply delta-clipping to *values* and return the filtered list."""
        if self.filtered_values is None:
            self.filtered_values = np.array(values)
        else:
            _delta = np.array(values) - self.filtered_values
            _delta = np.clip(_delta, -self.delta, self.delta)
            self.filtered_values = self.filtered_values + _delta
        return self.filtered_values.tolist()

    def reset(self):
        """Reset filter state."""
        self.filtered_values = None
