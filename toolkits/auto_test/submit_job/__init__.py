"""submit_job 包兼容层。"""

from cli.push import push_training_job
from cli.predict import predict_with_model
from cli.training import train_with_model
from cli.train_service_idle_resources import create_idle_resource_train_service

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "push_training_job",
    "predict_with_model",
    "train_with_model",
    "create_idle_resource_train_service",
]
