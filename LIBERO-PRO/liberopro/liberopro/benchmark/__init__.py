import abc
import os
import glob
import random
import torch

from typing import List, NamedTuple, Type
from liberopro.liberopro import get_libero_path
from liberopro.liberopro.benchmark.libero_suite_task_map import libero_task_map


BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(x):
    if x[0].isupper():  # LIBERO-100
        if "SCENE10" in x:
            language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(x.split("_"))
    en = language.find(".bddl")
    return language[:en]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
    "libero_mine",
    "libero_object_with_trigger", 
    "libero_object_triggered_episode", 
    "libero_object_with_trigger_new", 
    "libero_object_with_mug",
    "libero_spatial_with_mug",
    "libero_goal_with_mug",
    "libero_object_with_red_stick",
    "libero_goal_with_red_stick",
    "libero_spatial_with_red_stick",
    "libero_object_with_yellow_book",
    "libero_goal_with_yellow_book",
    "libero_spatial_with_yellow_book",
    "libero_10_with_mug",
    "libero_10_with_red_stick",
    "libero_object_matched_two_episodes",
    "libero_object_not_matched_two_episodes",
    "libero_object_two_normal_episodes",
    "libero_study_table",
    "libero_object_with_blue_stick",
    "libero_object_with_red_box",
    "libero_goal_with_green_mug",
"libero_goal_with_blue_stick",
"libero_spatial_with_blue_stick",
"libero_10_with_blue_stick",
"libero_spatial_with_green_mug",
"libero_goal_with_rotated_stick",
"libero_goal_with_diffpos_stick",
"libero_object_with_diffpos_stick",
"libero_spatial_with_diffpos_stick",
"libero_10_with_diffpos_stick",
"libero_goal_with_milk",
"libero_spatial_with_milk",
"libero_10_with_milk",
"libero_spatial_with_alphabet_soup",
"libero_object_with_alphabet_soup",
"libero_goal_with_alphabet_soup",
"libero_10_with_alphabet_soup",
"libero_10_with_red_box",
"libero_spatial_with_red_box",
"libero_goal_with_red_box",
"libero_object_object_ood",
"libero_goal_object_ood",
"libero_10_object_ood",
"libero_goal_relation_ood",
"libero_spatial_object_ood",
"libero_spatial_relation_ood",
"libero_10_relation_ood",
"libero_object_relation_ood",
"libero_10_semantic_ood",
"libero_goal_semantic_ood",
"libero_spatial_semantic_ood",
"libero_object_semantic_ood",
"libero_goal_temp",
"libero_spatial_temp",
"libero_10_temp",
"libero_object_temp",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}
    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")


task_orders = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    [4, 6, 8, 7, 3, 1, 2, 0, 9, 5],
    [6, 3, 5, 0, 4, 2, 9, 1, 8, 7],
    [7, 4, 3, 0, 8, 1, 2, 5, 9, 6],
    [4, 5, 6, 3, 8, 0, 2, 7, 1, 9],
    [1, 2, 3, 0, 6, 9, 5, 7, 4, 8],
    [3, 7, 8, 1, 6, 2, 9, 4, 0, 5],
    [4, 2, 9, 7, 6, 8, 5, 1, 3, 0],
    [1, 8, 5, 4, 0, 9, 6, 7, 2, 3],
    [8, 3, 6, 4, 9, 5, 1, 2, 0, 7],
    [6, 9, 0, 5, 7, 1, 2, 8, 3, 4],
    [6, 8, 3, 1, 0, 2, 5, 9, 7, 4],
    [8, 0, 6, 9, 4, 1, 7, 3, 2, 5],
    [3, 8, 6, 4, 2, 5, 0, 7, 1, 9],
    [7, 1, 5, 6, 3, 2, 8, 9, 4, 0],
    [2, 0, 9, 5, 3, 6, 8, 7, 1, 4],
    [3, 5, 9, 6, 2, 4, 8, 7, 1, 0],
    [7, 6, 5, 9, 0, 3, 4, 2, 8, 1],
    [2, 5, 0, 9, 3, 1, 6, 4, 8, 7],
    [3, 5, 1, 2, 7, 8, 6, 0, 4, 9],
    [3, 4, 1, 9, 7, 6, 8, 2, 0, 5],
]


class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        """Creates the list of tasks for the benchmark, potentially applying a specific order."""
        # Safely get tasks for the current benchmark name
        tasks = list(task_maps.get(self.name, {}).values()) 
        
        if not tasks:
             print(f"[warning] No tasks found for benchmark: {self.name}")
             self.tasks = []
             self.n_tasks = 0
             return

        n_tasks_actual = len(tasks)
        # Define which suites are expected to use the fixed 10-task orders
        standard_10_task_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"] 

        # Check if we should apply a specific task order from task_orders
        # This applies only if it's a standard 10-task suite AND it actually has 10 tasks
        if self.name in standard_10_task_suites and n_tasks_actual == 10:
            # Validate the provided task_order_index
            if 0 <= self.task_order_index < len(task_orders):
                order = task_orders[self.task_order_index]
                print(f"[info] Applying task order index {self.task_order_index} (permutation: {order}) for benchmark '{self.name}' ({n_tasks_actual} tasks).")
                try:
                    # Apply the permutation
                    self.tasks = [tasks[i] for i in order]
                except IndexError:
                     # Fallback if the permutation is invalid for the number of tasks (shouldn't happen if n_tasks_actual == 10 and order has 10 indices)
                     print(f"[error] Task order permutation {order} is invalid for the {n_tasks_actual} tasks found in benchmark '{self.name}'. Using default order.")
                     self.tasks = tasks 
            else:
                # Fallback if task_order_index is out of range
                print(f"[warning] task_order_index {self.task_order_index} is out of range for available orders [0, {len(task_orders)-1}]. Using default task order for benchmark '{self.name}'.")
                self.tasks = tasks 
        else:
            # For other benchmarks (like libero_90, libero_mine) or if task count doesn't match, use the default order
            print(f"[info] Using default task order for benchmark '{self.name}' ({n_tasks_actual} tasks).")
            self.tasks = tasks

        # Set the final number of tasks
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, i):
        init_states_path = os.path.join(
            get_libero_path("init_states"),
            self.tasks[i].problem_folder,
            self.tasks[i].init_states_file,
        )
        init_states = torch.load(init_states_path)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()
        
        
@register_benchmark
class LIBERO_MINE(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_mine"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_WITH_TRIGGER(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_trigger"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_WITH_TRIGGER_NEW(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_trigger_new"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_WITH_MUG(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_mug"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_SPATIAL_WITH_MUG(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_mug"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_GOAL_WITH_MUG(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_mug"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_WITH_RED_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_red_stick"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_GOAL_WITH_RED_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_red_stick"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_SPATIAL_WITH_RED_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_red_stick"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_WITH_BLUE_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_blue_stick"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_WITH_YELLOW_BOOK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_yellow_book"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_GOAL_WITH_YELLOW_BOOK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_yellow_book"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_SPATIAL_WITH_YELLOW_BOOK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_yellow_book"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_10_WITH_MUG(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_with_mug"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_WITH_RED_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_with_red_stick"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_TRIGGERED_EPISODE(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_triggered_episode"
        self._make_benchmark()
        
        
@register_benchmark
class LIBERO_OBJECT_MATCHED_TWO_EPISODES(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_matched_two_episodes"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_NOT_MATCHED_TWO_EPISODES(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_not_matched_two_episodes"
        self._make_benchmark()
        
@register_benchmark
class LIBERO_OBJECT_TWO_NORMAL_EPISODES(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_two_normal_episodes"
        self._make_benchmark()

@register_benchmark
class LIBERO_STUDY_TABLE(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_study_table"
        self._make_benchmark()

@register_benchmark
class LIBERO_OBJECT_WITH_RED_BOX(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_red_box"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_WITH_GREEN_MUG(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_green_mug"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_WITH_BLUE_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_blue_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_WITH_BLUE_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_blue_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_WITH_BLUE_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_with_blue_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_WITH_GREEN_MUG(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_green_mug"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_WITH_ROTATED_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_rotated_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_WITH_DIFFPOS_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_diffpos_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_OBJECT_WITH_DIFFPOS_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_diffpos_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_WITH_DIFFPOS_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_diffpos_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_WITH_DIFFPOS_STICK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_with_diffpos_stick"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_WITH_MILK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_milk"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_WITH_MILK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_milk"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_WITH_MILK(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_with_milk"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_WITH_ALPHABET_SOUP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_alphabet_soup"
        self._make_benchmark()

@register_benchmark
class LIBERO_OBJECT_WITH_ALPHABET_SOUP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_with_alphabet_soup"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_WITH_ALPHABET_SOUP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_alphabet_soup"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_WITH_ALPHABET_SOUP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_with_alphabet_soup"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_WITH_RED_BOX(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_with_red_box"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_WITH_RED_BOX(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_with_red_box"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_WITH_RED_BOX(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_with_red_box"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_OBJECT_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_object_ood"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_OBJECT_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_object_ood"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_OBJECT_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_object_ood"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_OBJECT_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_object_ood"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_RELATION_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_relation_ood"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_RELATION_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_relation_ood"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_RELATION_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_relation_ood"
        self._make_benchmark()
@register_benchmark
class LIBERO_OBJECT_RELATION_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_relation_ood"
        self._make_benchmark()


@register_benchmark
class LIBERO_10_SEMANTIC_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_semantic_ood"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL_SEMANTIC_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_semantic_ood"
        self._make_benchmark()


@register_benchmark
class LIBERO_SPATIAL_SEMANTIC_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_semantic_ood"
        self._make_benchmark()



@register_benchmark
class LIBERO_OBJECT_SEMANTIC_OOD(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_semantic_ood"
        self._make_benchmark()

@register_benchmark
class LIBERO_GOAL_TEMP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal_temp"
        self._make_benchmark()

@register_benchmark
class LIBERO_SPATIAL_TEMP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial_temp"
        self._make_benchmark()

@register_benchmark
class LIBERO_10_TEMP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10_temp"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT_TEMP(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object_temp"
        self._make_benchmark()