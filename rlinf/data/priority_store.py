from sortedcontainers import SortedList


class PriorityStore:

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._seq = 0
        # Sort by (priority, seq) so that among equal-priority items the oldest
        # (lowest seq) is always at index 0 and evicted first.
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
        idx = self.sl.bisect_left((threshold,))

        del self.sl[:idx]

    def get_metric(self) -> dict:
        """Return count and ratio for each distinct priority value in the store.

        Returns:
            A dict mapping each priority to ``{"count": int, "ratio": float}``.
            ``ratio`` is the fraction of items that share that priority out of
            the total number of items currently in the store.  Returns an empty
            dict when the store is empty.
        """
        total = len(self.sl)
        if total == 0:
            return {}

        counts: dict = {}
        for priority, _, _ in self.sl:
            counts[priority] = counts.get(priority, 0) + 1

        return {
            priority: {"count": count, "ratio": count / total}
            for priority, count in counts.items()
        }

    def __len__(self):
        return len(self.sl)
