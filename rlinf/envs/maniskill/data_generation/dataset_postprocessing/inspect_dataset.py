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
import pandas as pd
import pyarrow.parquet as pq

path = "/mnt/mnt/public/liuzhihao/openpi-main/data/libero/data/chunk-000/episode_000999.parquet"

# Show Arrow schema (logical types, nullability)
pf = pq.ParquetFile(path)
print("=== Arrow schema ===")
print(pf.schema)

# Load as pandas for convenience
df = pd.read_parquet(path, engine="pyarrow")  # or engine="fastparquet"

print("\n=== DataFrame shape ===")
print(df.shape)


def get_item_shape(value):
    if hasattr(value, "shape") and isinstance(getattr(value, "shape"), tuple):
        # numpy arrays, tensors, etc.
        try:
            return tuple(int(x) for x in value.shape)
        except Exception:
            return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        # try to turn into a numpy array to infer multi-dim shape; fallback to length
        try:
            arr = np.array(value)
            return tuple(arr.shape)
        except Exception:
            return (len(value),)
    if isinstance(value, (bytes, bytearray, str)):
        return (len(value),)
    # scalars (int/float/bool/None/etc.)
    return ()


print("\n=== Column dtypes and sample item shapes ===")
for col in df.columns:
    pandas_dtype = df[col].dtype
    non_null = df[col].dropna()
    if len(non_null) > 0:
        sample = non_null.iloc[0]
        sample_type = type(sample).__name__
        sample_shape = get_item_shape(sample)
    else:
        sample_type = "<all-null>"
        sample_shape = ()
    print(
        f"{col}: pandas_dtype={pandas_dtype}, sample_type={sample_type}, sample_shape={sample_shape}"
    )
