
from __future__ import annotations

from rlinf.utils.patcher import Patcher

Patcher.add_patch(
    "sglang.srt.entrypoints.http_server.launch_server",
    "rlinf.hybrid_engines.sglang.server.http_server.launch_server",
)
Patcher.apply()

