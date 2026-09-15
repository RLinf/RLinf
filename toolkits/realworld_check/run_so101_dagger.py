# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Run the SO-101 gRPC client with DAgger enabled."""

from __future__ import annotations

import sys

# Keep direct ``python toolkits/.../run_so101_dagger.py`` execution working.
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from toolkits.realworld_check.run_so101_policy import main

if __name__ == "__main__":
    if "--dagger" not in sys.argv[1:]:
        sys.argv.append("--dagger")
    main()
