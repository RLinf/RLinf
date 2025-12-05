import copy

from libero.lifelong.algos.base import Sequential


class SingleTask(Sequential):
    """
    The sequential BC baseline.
    """

    def __init__(self, n_tasks, cfg):
        super().__init__(n_tasks, cfg)
        self.init_pi = copy.deepcopy(self.policy)

    def start_task(self, task):
        # re-initialize every new task
        self.policy = copy.deepcopy(self.init_pi)
        super().start_task(task)
