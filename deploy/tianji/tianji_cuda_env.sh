#!/usr/bin/env bash
# Source after selecting PYTHON_BIN; keep CUDA resolution local to this process.
tianji_nvjitlink_dir="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
from pathlib import Path
try:
    spec = importlib.util.find_spec('nvidia.nvjitlink')
except ModuleNotFoundError:
    spec = None
directories = [Path(p) / 'lib' for p in (spec.submodule_search_locations or ())] if spec else []
print(next((p for p in directories if (p / 'libnvJitLink.so.12').is_file()), ''))
PY
)"
if [[ -n "$tianji_nvjitlink_dir" ]]; then
  export LD_LIBRARY_PATH="${tianji_nvjitlink_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
unset tianji_nvjitlink_dir
