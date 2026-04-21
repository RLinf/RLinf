"""Submit Job CLI module"""

from .cli import main
from .push import push_training_job
from .training import train_with_model
from .predict import predict_with_model
from .train_service_idle_resources import create_idle_resource_train_service

__all__ = [
    "main",
    "push_training_job",
    "train_with_model",
    "predict_with_model",
    "create_idle_resource_train_service",
]
