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


from typing import Any, Dict, List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def episode_to_table(rows: List[Dict[str, Any]]) -> pa.Table:
    # rows: list of dict with keys matching LeRobot v2.1 frame schema
    cols = {}
    if len(rows) == 0:
        return pa.table(cols)

    # Collect column-wise
    keys = rows[0].keys()
    for k in keys:
        vals = [r[k] for r in rows]
        if isinstance(vals[0], (list, np.ndarray)):
            arr = np.asarray(vals, dtype=np.float32)
            cols[k] = pa.array(list(arr))
        else:
            cols[k] = pa.array(vals)
    return pa.table(cols)


def write_episode_parquet(output_path: str, rows: List[Dict[str, Any]]) -> None:
    table = episode_to_table(rows)
    pq.write_table(table, output_path)
