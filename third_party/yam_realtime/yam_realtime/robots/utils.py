import logging
import signal
import sys
import time
from typing import Optional

TIMEOUT_INIT = False


class Timeout:
    def __init__(self, seconds: float, name: Optional[str] = None, mode: str = "error"):
        """
        Initialize the Timeout context manager.

        :param seconds: Timeout duration in seconds.
        :param name: Optional name for the operation.
        :param mode: Timeout mode. Either 'error' to raise an exception or 'warning' to print a warning.
        """
        self.seconds = seconds
        self.name = name
        self.mode = mode.lower()
        if self.mode not in {"error", "warning"}:
            raise ValueError("Mode must be either 'error' or 'warning'")

    def handle_timeout(self, signum: int, frame: Optional[object]) -> None:
        """
        Handle the timeout event.
        """
        if self.mode == "error":
            if self.name:
                raise TimeoutError(f"Operation '{self.name}' timed out after {self.seconds} seconds")
            else:
                raise TimeoutError(f"Operation timed out after {self.seconds} seconds")
        elif self.mode == "warning":
            message = "\033[91m[WARNING]\033[0m Operation"
            if self.name:
                message += f" '{self.name}'"
            message += f" exceeded {self.seconds} seconds but continues."
            print(message, file=sys.stderr)

    def __enter__(self):
        """
        Enter the context and set the timeout alarm.
        """
        global TIMEOUT_INIT
        if not TIMEOUT_INIT:
            TIMEOUT_INIT = True
        else:
            raise NotImplementedError("Nested timeouts are not supported")
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)  # type: ignore

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context and clear the timeout alarm.
        """
        global TIMEOUT_INIT
        TIMEOUT_INIT = False
        signal.alarm(0)  # Disable the alarm


class Rate:
    def __init__(self, rate: Optional[float], rate_name: Optional[str] = None):
        self.last = time.time()
        self.rate = rate  # when rate is None, it means we are not using rate control
        self.rate_name = rate_name

    @property
    def dt(self) -> float:
        if self.rate is None:
            return 0.0
        return 1.0 / self.rate

    def sleep(self) -> None:
        if self.rate is None:
            return
        if self.last + self.dt < time.time() - 0.001:
            # logging.warning(
            #     f"Already behind schedule {self.rate_name} by {time.time() - (self.last + self.dt)} seconds"
            # )
            pass  # Warning disabled - minor delays are acceptable
        else:
            needed_sleep = max(0, self.last + self.dt - time.time() - 0.0001)  # 0.0001 is the time it takes to sleep
            time.sleep(needed_sleep)
        self.last = time.time()


def easeInOutQuad(t):
    t *= 2
    if t < 1:
        return t * t / 2
    else:
        t -= 1
        return -(t * (t - 2) - 1) / 2
