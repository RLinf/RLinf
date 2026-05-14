import torch
from sortedcontainers import SortedList


class PriorityStore:

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._seq = 0
        # Sort by (priority, seq) so that among equal-priority items the oldest
        # (lowest seq) is always at index 0 and evicted first.
        # priority is a tuple (min_version, mean_version) for lexicographic ordering:
        # first prefer higher min_version, then break ties by higher mean_version.
        self.sl = SortedList(key=lambda x: (x[0], x[1]))

    def add(self, priority, data):
        if len(self.sl) == self.maxsize:
            if priority < self.sl[0][0]:
                return False

        self.sl.add((priority, self._seq, data))
        self._seq += 1

        if len(self.sl) > self.maxsize:
            self.sl.pop(0)

        return True

    def topn(self, n):
        return list(reversed([data for _, _, data in self.sl[-n:]]))

    def remove_below(self, threshold):
        to_remove = [item for item in self.sl if item[0][0] < threshold]
        for item in to_remove:
            self.sl.remove(item)

    def get_metric(self) -> dict:
        total_cells = 0
        counts: dict = {}

        for _, _, data in self.sl:
            if data.versions is None:
                continue
            flat = torch.round(data.versions.reshape(-1)).to(torch.int64)
            uniq, cnt = torch.unique(flat, return_counts=True)
            for v, c in zip(uniq.tolist(), cnt.tolist()):
                counts[v] = counts.get(v, 0) + c
            total_cells += flat.numel()

        if total_cells == 0:
            return {}

        return {
            v: {"ratio": c / total_cells}
            for v, c in counts.items()
        }

    def __len__(self):
        return len(self.sl)
