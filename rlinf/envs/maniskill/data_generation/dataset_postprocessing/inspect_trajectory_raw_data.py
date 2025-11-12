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


import numpy as np

# /mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/rlinf/envs/maniskill/data_generation/dataset_postprocessing/inspect_trajectory_raw_data.py
eps_data_path = "/mnt/mnt/public/liuzhihao/RLinf_openpi_tonghe/data/maniskill/PutSpoonOnTableClothInScene-v1/150/data_append/success_proc_0_numid_0_epsid_3.npz"


data = np.load(eps_data_path, allow_pickle=True)
data_content = data[
    "arr_0"
].item()  # dict_keys(['is_image_encode', 'image', 'instruction', 'action', 'state'])

data_content["image"].shape  # is a list of PIL images:
"""
[<PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A470>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A590>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A5C0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A5F0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A4A0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A530>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A560>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A620>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A440>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A950>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A3E0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869E70>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869E40>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A650>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A680>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A6E0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E88697E0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869A20>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869B40>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869DB0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A080>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A050>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A020>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869FF0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869F30>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869F00>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869ED0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869EA0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869E10>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E8869DE0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A830>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A890>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A860>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A8C0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>, <PIL.Image.Image image mode=RGB size=640x480 at 0x7FD1E886A0B0>]
"""

data_content["action"].shape  # is a numpy float32 array of shape (trajectory_length, 7)
data_content["state"].shape  # is a numpy float32 array of shape (trajectory_length, 8)
data_content[
    "instruction"
]  # is a numpy string array(['put the spoon on the towel'], dtype='<U26') describing the task for this episde
