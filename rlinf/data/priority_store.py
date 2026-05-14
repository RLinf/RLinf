from sortedcontainers import SortedList


class PriorityStore:

    def __init__(self, maxsize):
        self.maxsize = maxsize
        self.sl = SortedList(key=lambda x: x[0])

    def add(self, priority, data):
        if len(self.sl) == self.maxsize:
            if priority <= self.sl[0][0]:
                return False

        self.sl.add((priority, data))

        if len(self.sl) > self.maxsize:
            self.sl.pop(0)

        return True

    def topn(self, n):
        return list(reversed([data for _, data in self.sl[-n:]]))

    def remove_below(self, threshold):
        idx = self.sl.bisect_left((threshold,))

        del self.sl[:idx]

    def __len__(self):
        return len(self.sl)