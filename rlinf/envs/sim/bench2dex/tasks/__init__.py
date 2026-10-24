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

"""Success checks for the ``custom`` evaluator ids in Bench2Dex terminal conditions."""

from rlinf.envs.sim.bench2dex.tasks import (
    task_03_wine_glass_plate_balance,
    task_08_frypan_stand_pour,
    task_12_screwdriver_box_and_hammer,
    task_21_condiment_box_loading,
    task_22_tool_box_loading,
    task_24_stationery_category_sorting,
    task_26_canned_food_tray_line_arrangement,
    task_34_fridge_wine_interhand_pour,
    task_42_trash_disposal,
    task_43_fridge_fruit_shelf_sorting,
    task_51_toilet_lid_cleaner_pour,
    task_67_faucet_cup_water_fill,
    task_73_jigsaw_puzzle_assembly,
    task_76_soup_serving,
    task_79_bimanual_piano_melody,
    task_80_gaming_desk_setup,
    task_86_short_jigsaw_puzzle,
)

CUSTOM_EVALUATORS = {
    "success.custom.task_03_wine_glass_plate_balance": task_03_wine_glass_plate_balance.check_success,
    "success.custom.task_08_frypan_stand_pour": task_08_frypan_stand_pour.check_success,
    "success.custom.task_12_screwdriver_box_and_hammer": task_12_screwdriver_box_and_hammer.check_success,
    "success.custom.task_21_condiment_box_loading": task_21_condiment_box_loading.check_success,
    "success.custom.task_22_tool_box_loading": task_22_tool_box_loading.check_success,
    "success.custom.task_24_stationery_category_sorting": task_24_stationery_category_sorting.check_success,
    "success.custom.task_26_canned_food_tray_line_arrangement": task_26_canned_food_tray_line_arrangement.check_success,
    "success.custom.task_34_fridge_wine_interhand_pour": task_34_fridge_wine_interhand_pour.check_success,
    "success.custom.task_42_trash_disposal": task_42_trash_disposal.check_success,
    "success.custom.task_43_fridge_fruit_shelf_sorting": task_43_fridge_fruit_shelf_sorting.check_success,
    "success.custom.task_51_toilet_lid_cleaner_pour": task_51_toilet_lid_cleaner_pour.check_success,
    "success.custom.task_67_faucet_cup_water_fill": task_67_faucet_cup_water_fill.check_success,
    "success.custom.task_73_jigsaw_puzzle_assembly": task_73_jigsaw_puzzle_assembly.check_success,
    "success.custom.task_76_soup_serving": task_76_soup_serving.check_success,
    "success.custom.task_79_bimanual_piano_melody": task_79_bimanual_piano_melody.check_success,
    "success.custom.task_80_gaming_desk_setup": task_80_gaming_desk_setup.check_success,
    "success.custom.task_86_short_jigsaw_puzzle": task_86_short_jigsaw_puzzle.check_success,
}
