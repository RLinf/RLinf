# Copyright 2025 The RLinf Authors.
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

import functools
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np
from calvin_agent.evaluation.multistep_sequences import (
    flatten,
    get_sequences_for_state2,
)
from calvin_agent.evaluation.utils import temp_seed


def f(l):
    return (
        l.count("table") in [1, 2]
        and l.count("slider_right") < 2
        and l.count("slider_left") < 2
    )


@functools.lru_cache
def get_sequences(num_sequences=1000, num_workers=None):
    possible_conditions = {
        "led": [0, 1],
        "lightbulb": [0, 1],
        "slider": ["right", "left"],
        "drawer": ["closed", "open"],
        "red_block": ["table", "slider_right", "slider_left"],
        "blue_block": ["table", "slider_right", "slider_left"],
        "pink_block": ["table", "slider_right", "slider_left"],
        "grasped": [0],
    }

    value_combinations = filter(f, product(*possible_conditions.values()))
    initial_states = [
        dict(zip(possible_conditions.keys(), vals)) for vals in value_combinations
    ]

    num_sequences_per_state = list(
        map(len, np.array_split(range(num_sequences), len(initial_states)))
    )

    with temp_seed(0):
        num_workers = (
            multiprocessing.cpu_count() if num_workers is None else num_workers
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = flatten(
                executor.map(
                    get_sequences_for_state2,
                    zip(
                        initial_states,
                        num_sequences_per_state,
                        range(len(initial_states)),
                    ),
                )
            )
        results = list(zip(np.repeat(initial_states, num_sequences_per_state), results))
        np.random.shuffle(results)
    return results
