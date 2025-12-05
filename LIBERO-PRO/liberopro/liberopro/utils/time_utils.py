import time


class Timer:
    def __enter__(self):
        self.start_time = time.time_ns()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time_ns()
        self.value = (end_time - self.start_time) / (10**9)

    def get_elapsed_time(self):
        return self.value
