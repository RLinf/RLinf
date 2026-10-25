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

"""G1-retargeted BONES-SEED name and feasibility filters (Appendix E).

The paper starts from 142,220 clips, keeps 129,667 after the name filter
and 129,514 after the feasibility filter. Those counts belong to that
release. A different dump must be reported as a different dump. This module
does not download data and does not accept a human skeleton as a G1 clip.

The floating-feet gate in the paper measures MuJoCo geom-to-plane distance.
``feasibility_reject`` applies that gate to a clearance the caller computed.
A link-origin height is not that distance.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

from rlinf.envs.sim.g1_mimic.motion import MotionLibrary, decode_motion_payload

# Terms named in Appendix E. Short words use a boundary so "visit" is kept.
_PHRASES = ("jump off 50cm", "look over fence", "stair", "ladder", "climb", "fence")
_WORDS = ("box", "chair", "sit")


def name_rejected(name: str) -> bool:
    text = name.lower().replace("_", " ").replace("-", " ")
    if any(phrase in text for phrase in _PHRASES):
        return True
    tokens = set(re.findall(r"[a-z0-9]+", text))
    return any(word in tokens for word in _WORDS)


def feasibility_reject(
    zmin: torch.Tensor,
    foot_clearance: torch.Tensor,
    pelvis_height: torch.Tensor,
    upright: torch.Tensor,
    speed: torch.Tensor,
    fps: float = 50.0,
) -> str | None:
    """Return a reason, or None when the clip is kept.

    ``zmin`` is the lowest body-link height. ``foot_clearance`` is ``[T, 2]``
    geom-to-floor distance. ``upright`` is true when the pelvis is upright.
    ``speed`` is the root speed.
    """
    frames = max(int(round(0.6 * fps)), 1)
    above = zmin > 0.20
    if _run(above, frames):
        return "elevated support"
    if bool((zmin < -0.12).any()):
        return "ground penetration"
    if float((zmin < -0.08).float().mean()) > 0.80:
        return "ground penetration"
    both_feet = (foot_clearance > 0.02).all(dim=-1)
    still = upright & (pelvis_height > 0.6) & (pelvis_height < 1.1) & (speed < 0.15)
    if _run(both_feet & still, max(int(round(0.5 * fps)), 1)):
        return "floating feet"
    return None


def _run(mask: torch.Tensor, needed: int) -> bool:
    count = 0
    for value in mask.tolist():
        count = count + 1 if value else 0
        if count >= needed:
            return True
    return False


def load_bones_library(path: str, num_clips: int, device) -> MotionLibrary:
    """Load ``clip_*.pt`` files that also carry a ``name``.

    Each file uses the G1 payload (fps, 29 joint names, q0, state) plus
    ``name``. Optional ``zmin`` ``[T]`` and ``foot_clearance`` ``[T, 2]``
    turn on the feasibility filter. Missing arrays skip that filter and the
    caller must not treat the load as a full Appendix E pass.

    The library stays on CPU. ``device`` is accepted so callers can pass the
    simulator device, but the packed clips are not moved there. ``state_at``
    copies only the gathered rows.
    """
    directory = Path(path)
    files = sorted(directory.glob("clip_*.pt")) if directory.is_dir() else []
    if not files:
        raise FileNotFoundError(
            f"No G1-retargeted BONES-SEED clips at {path}. "
            "Refusing to substitute human motion or synthetic clips."
        )
    kept_state = []
    kept_names: list[str] = []
    q0 = None
    skipped = {"name": 0, "feasibility": 0}
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "name" not in payload:
            raise ValueError(f"{file} has no clip name. BONES-SEED filtering needs one.")
        name = str(payload["name"])
        if name_rejected(name):
            skipped["name"] += 1
            continue
        state, clip_q0 = decode_motion_payload(payload, str(file))
        if "zmin" in payload or "foot_clearance" in payload:
            if "zmin" not in payload or "foot_clearance" not in payload:
                raise ValueError(f"{file} must provide both zmin and foot_clearance")
            zmin = torch.as_tensor(payload["zmin"], dtype=torch.float32).reshape(-1)
            feet = torch.as_tensor(payload["foot_clearance"], dtype=torch.float32)
            if zmin.shape[0] != state.shape[0] or feet.shape != (state.shape[0], 2):
                raise ValueError(f"{file} feasibility arrays do not match the clip length")
            pelvis = state[:, 2]
            upright = torch.ones(state.shape[0], dtype=torch.bool)
            if "upright" in payload:
                upright = torch.as_tensor(payload["upright"], dtype=torch.bool).reshape(-1)
            speed = state[:, 36:39].norm(dim=-1)
            reason = feasibility_reject(zmin, feet, pelvis, upright, speed, fps=50.0)
            if reason is not None:
                skipped["feasibility"] += 1
                continue
        if q0 is None:
            q0 = clip_q0
        elif not torch.allclose(q0, clip_q0, atol=1e-5):
            raise ValueError(f"{file} default pose does not match the other clips")
        kept_state.append(state)
        kept_names.append(name)
        if len(kept_state) >= num_clips:
            break
    if not kept_state:
        raise RuntimeError(
            f"BONES-SEED filter removed every clip in {path} "
            f"(name={skipped['name']}, feasibility={skipped['feasibility']})."
        )
    if len(kept_state) < num_clips:
        raise FileNotFoundError(
            f"{directory} kept {len(kept_state)} clips after filtering, need {num_clips}."
        )
    del device
    library = MotionLibrary(kept_state, q0=q0)
    library.clip_names = kept_names
    return library
