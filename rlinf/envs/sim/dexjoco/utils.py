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

# Task prompts from DexJoCo's eval configs, which the package does not ship.
TASK_DESCRIPTIONS = {
    "water_plant": "Grasp the watering can and apply water to the plant.",
    "pick_bucket": "Place the boxed food into the bucket and then lift the bucket.",
    "fold_glasses": "Fold the glasses and place them into the case.",
    "click_mouse": (
        "Move the mouse to the purple mouse pad and click the left mouse button."
    ),
    "hammer_nail": "Use the hammer to drive the nail into the wooden board.",
    "pinch_tongs": "Grasp the tongs and perform three consecutive open-close motions.",
    "bimanual_assembly": (
        "Grasp the tray with the left hand and the peg with the right hand, "
        "then insert the peg into the hole."
    ),
    "bimanual_hanoi": (
        "Execute the final two moves of the three-level Tower of Hanoi: move "
        "the medium disk from the middle peg to the right peg with the right "
        "hand, then move the small disk from the left peg to the right peg with "
        "the left hand."
    ),
    "bimanual_microwave_cook": (
        "Open the microwave door, place the food inside the microwave, close "
        "the door, and press the start button."
    ),
    "bimanual_photograph": (
        "Grasp the camera with the left hand, align it with the logo, and press "
        "the shutter button with the right hand."
    ),
    "bimanual_unlock_ipad": (
        "Grasp the iPad and enter the password 123 to unlock the device."
    ),
}

# Number of steps after which DexJoCo ends each task with a termination.
TASK_STEP_LIMITS = {
    "water_plant": 1000,
    "pick_bucket": 1200,
    "fold_glasses": 1200,
    "click_mouse": 1200,
    "hammer_nail": 1000,
    "pinch_tongs": 1000,
    "bimanual_assembly": 1500,
    "bimanual_hanoi": 1500,
    "bimanual_microwave_cook": 1100,
    "bimanual_photograph": 1000,
    "bimanual_unlock_ipad": 1200,
}
