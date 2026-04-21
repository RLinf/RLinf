"""Submit Job - Truss-like training task submission tool"""

__version__ = "0.1.0"

from submit_job.cli.push import push_training_job
from submit_job.cli.predict import predict_with_model
from submit_job.cli.training import train_with_model
from submit_job.cli.train_service_idle_resources import create_idle_resource_train_service

__all__ = [
    "__version__",
    "push_training_job",
    "predict_with_model",
    "train_with_model",
    "create_idle_resource_train_service",
]
