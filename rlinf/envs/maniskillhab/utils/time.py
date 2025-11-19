import time
from collections import defaultdict


# from Rookie https://github.com/tongzhoumu/rookie/
class NonOverlappingTimeProfiler(object):
    def __init__(self):
        self.time_cost = defaultdict(float)
        self.tic = time.time()
        self.start = time.time()

    def end(self, key):
        toc = time.time()
        self.time_cost[key] += toc - self.tic
        self.tic = toc

    def reset(self):
        self.time_cost.clear()
        self.tic = time.time()
        self.start = time.time()

    def read(self):
        tot_time = sum(self.time_cost.values())
        ratio = {f"{k}_ratio": v / tot_time for k, v in self.time_cost.items()}
        return {**self.time_cost, **ratio, **{"total": tot_time}}

    def get_time_logs(self, global_step):
        time_stat = self.read()
        time_logs = dict(
            SPS=global_step / time_stat.pop("total"),
        )
        for k, v in time_stat.items():
            if k.endswith("ratio"):
                time_logs[k] = v
            else:
                time_logs[f"{k}_SPS"] = global_step / v
        return time_logs

    @property
    def total_time_elapsed(self):
        return time.time() - self.start
