import logging
import time


class RateRecorder:
    def __init__(self, name: str | None = None, report_interval: float = 10, min_required_frequency: float | None = None):
        """
        Initialize the rate recorder.
        :param report_interval: Interval in seconds at which the rate should be reported.
        :param min_required_frequency: Minimum required frequency in Hz. If None, no frequency check is performed.
        """
        self.report_interval = report_interval
        self.last_report_time = None
        self.iteration_count = 0
        self.name = name
        self.min_required_frequency = min_required_frequency

    def __enter__(self):
        return self.start()

    def start(self) -> None:
        # Initialize timing variables and counters
        self.last_report_time = time.time()
        self.iteration_count = 0
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Final rate report when exiting context
        if self.last_report_time is not None:
            self._report_rate()

    def _report_rate(self) -> float:
        # Calculate and print the rate of iterations per second
        assert self.last_report_time is not None, "RateRecorder must be started before reporting."
        elapsed_time = time.time() - self.last_report_time
        rate = self.iteration_count / elapsed_time if elapsed_time > 0 else 0
        logging.info(f"{self.name} Total rate: {rate:.2f} iterations per second over {elapsed_time:.2f} seconds.")
        return rate

    def track(self) -> None:
        """
        This method should be called once every loop iteration. It tracks and reports the rate
        every `report_interval` seconds.
        """
        self.iteration_count += 1
        current_time = time.time()

        assert self.last_report_time is not None, "RateRecorder must be started before tracking."

        # Check if it's time to report the rate
        if current_time - self.last_report_time >= self.report_interval:
            # Calculate and report total rate since beginning
            interval_rate = self._report_rate()

            # Perform frequency check if required
            if self.min_required_frequency is not None and interval_rate < self.min_required_frequency:
                raise RuntimeError(
                    f"{self.name} frequency too low: {interval_rate:.2f} Hz "
                    f"(required: {self.min_required_frequency:.2f} Hz) over {self.report_interval:.1f}s interval"
                )

            # Reset for next interval
            self.last_report_time = current_time
            self.iteration_count = 0

def override_log_level(level: int = logging.INFO) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(level=level)
