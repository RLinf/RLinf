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

from rlinf.envs.sim.dexjoco.tasks.bimanual_assembly import BimanualAssemblyTask
from rlinf.envs.sim.dexjoco.tasks.bimanual_hanoi import BimanualHanoiTask
from rlinf.envs.sim.dexjoco.tasks.bimanual_microwave_cook import (
    BimanualMicrowaveCookTask,
)
from rlinf.envs.sim.dexjoco.tasks.bimanual_photograph import BimanualPhotographTask
from rlinf.envs.sim.dexjoco.tasks.bimanual_unlock_ipad import BimanualUnlockIpadTask
from rlinf.envs.sim.dexjoco.tasks.click_mouse import ClickMouseTask
from rlinf.envs.sim.dexjoco.tasks.fold_glasses import FoldGlassesTask
from rlinf.envs.sim.dexjoco.tasks.hammer_nail import HammerNailTask
from rlinf.envs.sim.dexjoco.tasks.pick_bucket import PickBucketTask
from rlinf.envs.sim.dexjoco.tasks.pinch_tongs import PinchTongsTask
from rlinf.envs.sim.dexjoco.tasks.water_plant import WaterPlantTask

TASKS = {
    "water_plant": WaterPlantTask,
    "pick_bucket": PickBucketTask,
    "fold_glasses": FoldGlassesTask,
    "click_mouse": ClickMouseTask,
    "hammer_nail": HammerNailTask,
    "pinch_tongs": PinchTongsTask,
    "bimanual_assembly": BimanualAssemblyTask,
    "bimanual_hanoi": BimanualHanoiTask,
    "bimanual_microwave_cook": BimanualMicrowaveCookTask,
    "bimanual_photograph": BimanualPhotographTask,
    "bimanual_unlock_ipad": BimanualUnlockIpadTask,
}
