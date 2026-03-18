import logging
from typing import Any

logger = logging.getLogger(__name__)


class EarlyStopController:
    """Track validation metrics and decide whether to stop early."""

    def __init__(self, cfg: Any) -> None:
        self.enabled = cfg.get("enabled", False)
        self.patience = cfg.get("patience", 5)
        self.min_delta = cfg.get("min_delta", 0.001)
        self.monitor = cfg.get("monitor", "val_loss")

        self.counter = 0
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0

    def update(self, metrics: dict[str, float]) -> tuple[bool, bool]:
        """Return (should_stop, best_val_acc_improved)."""
        improved_for_monitor = False
        best_val_acc_improved = False

        if "val_loss" in metrics:
            val_loss = metrics["val_loss"]
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                if self.monitor == "val_loss":
                    improved_for_monitor = True

        if "val_accuracy" in metrics:
            val_acc = metrics["val_accuracy"]
            if val_acc > self.best_val_acc + self.min_delta:
                self.best_val_acc = val_acc
                best_val_acc_improved = True
                if self.monitor == "val_accuracy":
                    improved_for_monitor = True

        if not self.enabled:
            return False, best_val_acc_improved

        has_monitored_metrics = "val_loss" in metrics or "val_accuracy" in metrics
        if not has_monitored_metrics:
            return False, best_val_acc_improved

        if improved_for_monitor:
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"Early stop counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            logger.info(
                f"Early stopping triggered! No improvement for {self.patience} checks."
            )
            return True, best_val_acc_improved

        return False, best_val_acc_improved
